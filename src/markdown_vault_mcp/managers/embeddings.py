"""Vector-embedding lifecycle manager, composed into IndexManager (#1157).

Orchestrates the vector store's write lifecycle — cold build, boot
convergence, inline re-embedding during reindex and the deferred flush —
whose writes all go through the one
:class:`~markdown_vault_mcp.indexing.embedding_reconciler.EmbeddingReconciler`
(#1768), and keeps status reporting, opening the store and the rule for
which vectors a vanished document may lose (#1785 replaces that rule). Extracted verbatim from
:class:`~markdown_vault_mcp.managers.index.IndexManager` so the FTS pipeline
and the embedding pipeline each have a single reason to change; the shared
helpers that stayed behind (`_is_path_excluded`,
`_discover_indexable_candidates`) are injected as callables (#736).
"""

from __future__ import annotations

import json
import logging
import stat as stat_module
from typing import TYPE_CHECKING, Any

from markdown_vault_mcp.exceptions import EmbeddingsNotConfiguredError
from markdown_vault_mcp.indexing.embedding_reconciler import EmbeddingReconciler
from markdown_vault_mcp.utils.fs import is_directory, path_exists

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from markdown_vault_mcp.embed_text import EmbedTextBuilder
    from markdown_vault_mcp.interfaces import EmbeddingStore, KeywordIndex
    from markdown_vault_mcp.managers._vector_slot import VectorStoreSlot
    from markdown_vault_mcp.providers import EmbeddingProvider
    from markdown_vault_mcp.types import ParsedNote

logger = logging.getLogger(__name__)

# Maximum chunks per embedding provider call.  Keeps memory bounded during
# build_embeddings() — FastEmbed/ONNX can allocate pathologically large buffers
# when the entire corpus is sent in one batch (see issue #159).
_EMBEDDING_BATCH_SIZE = 4

# Disposition codes returned by EmbeddingsManager._embed_note_inline so the
# caller can log an accurate aggregate (#935). A provider failure and a vector
# the store refuses both leave the note's rows untouched (#1778).
_EMBED_OK = 0  # embedded (or nothing to embed)
_EMBED_KEPT = (
    1  # the note's existing vectors were kept: provider failed or write refused
)


class EmbeddingsManager:
    """Manages the vector-embedding lifecycle for one vault.

    Composed into :class:`~markdown_vault_mcp.managers.index.IndexManager`,
    which keeps thin delegations so the split is invisible outside
    ``managers/``. The FTS index is the source of truth for what to embed;
    the shared path helpers are injected as callables so the exclusion and
    discovery semantics stay defined in one place.

    Args:
        fts: The FTS index queried as the source of truth for what to embed.
        source_dir: Absolute path to the vault root directory.
        embeddings_path: Base path for ``.npy`` / ``.json`` sidecar files.
            ``None`` keeps the vectors in memory (#1708).
        embedding_provider: Provider used to generate embeddings.
        embed_text_builder: Shared
            :class:`~markdown_vault_mcp.embed_text.EmbedTextBuilder` used at
            every embedding site so hot, cold, converge, and flush paths all
            produce identical embedding input.
        embedding_batch_size: Maximum number of chunk texts sent to the
            embedding provider per call in the cold-build, convergence, and
            inline-reindex paths. Defaults to the module constant
            ``_EMBEDDING_BATCH_SIZE``.
        is_path_excluded: Injected exclusion check (the owner's
            ``_is_path_excluded``), so both pipelines share one policy.
        discover_candidates: Injected source-tree discovery (the owner's
            ``_discover_indexable_candidates``), used by the authoritative
            empty-build check's source walk.
        vector_slot: Opens, rebuilds and describes the vector store. ``None``
            with a provider builds a numpy store over ``embeddings_path``
            (in memory when that is ``None``), owned by this manager and
            never closed: every write site checkpoints what it changes.
    """

    def __init__(
        self,
        *,
        fts: KeywordIndex,
        source_dir: Path,
        embeddings_path: Path | None,
        embedding_provider: EmbeddingProvider | None,
        embed_text_builder: EmbedTextBuilder,
        embedding_batch_size: int = _EMBEDDING_BATCH_SIZE,
        is_path_excluded: Callable[[str], bool],
        discover_candidates: Callable[..., list[tuple[Path, str]]],
        vector_slot: VectorStoreSlot | None = None,
    ) -> None:
        self._fts = fts
        self._source_dir = source_dir
        self._embeddings_path = embeddings_path
        self._embedding_provider = embedding_provider
        self._embed_builder = embed_text_builder
        self._embedding_batch_size = embedding_batch_size
        self._is_path_excluded = is_path_excluded
        self._discover_candidates = discover_candidates
        if vector_slot is None and embedding_provider is not None:
            from markdown_vault_mcp.managers._vector_slot import VectorStoreSlot
            from markdown_vault_mcp.stores.numpy_sidecar import NumpySidecarBackend

            vector_slot = VectorStoreSlot(
                backend=NumpySidecarBackend(),
                location=embeddings_path,
                provider=embedding_provider,
                text_format=embed_text_builder.format_token(),
            )
        self._slot = vector_slot
        self._conflicts: set[str] = set()
        self._retries: set[str] = set()
        self._reconciler = (
            EmbeddingReconciler(
                fts=fts,
                provider=embedding_provider,
                embed_builder=embed_text_builder,
                batch_size=embedding_batch_size,
                confirm_removal=self._confirm_stale_vector_paths,
            )
            if embedding_provider is not None
            else None
        )

    # ------------------------------------------------------------------
    # Shared-path accessors (read by IndexManager's FTS pipeline)
    # ------------------------------------------------------------------

    @property
    def embeddings_path(self) -> Path | None:
        """Base path for the vector sidecar files (``None`` when held in memory)."""
        return self._embeddings_path

    @property
    def embedding_provider(self) -> EmbeddingProvider | None:
        """The configured embedding provider (``None`` when disabled)."""
        return self._embedding_provider

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_vectors(self) -> None:
        """Raise :class:`EmbeddingsNotConfiguredError` if embeddings are unconfigured."""
        if self._embedding_provider is None:
            raise EmbeddingsNotConfiguredError(
                "Embeddings are not enabled on this server, so there is nothing "
                "to build or query; search with mode='keyword' instead."
            )

    def _empty_embedding_build_is_authoritative(self, indexed_paths: set[str]) -> bool:
        """Return whether an empty FTS-derived embedding build is complete.

        A forced FTS rebuild can omit a source that failed parsing, leaving no
        row for :meth:`build_embeddings` to embed. Before replacing an old
        sidecar with an empty one, inspect only source candidates absent from
        FTS: since #1129, every deliberate skip (any surfaced category) leaves
        a tombstone row, so a candidate is accounted for iff it has one — no
        per-candidate re-parse needed. A candidate with neither a live row
        nor a tombstone is an FTS-population gap, and an incomplete source
        walk can hide candidates entirely (walk incompleteness is not
        representable as tombstones), so both still veto the empty save.
        """
        walk_incomplete = False

        def _record_walk_error(_exc: OSError) -> None:
            nonlocal walk_incomplete
            walk_incomplete = True

        candidates = self._discover_candidates(on_walk_error=_record_walk_error)
        if walk_incomplete:
            logger.warning(
                "build_embeddings_source_walk_incomplete outcome=no_vectors_saved"
            )
            return False

        for _abs_path, path in candidates:
            if path in indexed_paths:
                continue
            if self._fts.get_tombstone(path) is not None:
                # Deliberately skipped — the absence is recorded, not a gap.
                continue
            logger.warning(
                "build_embeddings_unindexed_source path=%s outcome=no_vectors_saved",
                path,
            )
            return False
        return True

    def _confirm_stale_vector_paths(self, stale_paths: list[str]) -> list[str]:
        """Return the stale sidecar paths whose FTS absence is authoritative.

        A path present in the vector sidecar but absent from the FTS index is
        reclaimable only when the absence means "not an index candidate".
        Since #1129 that is a direct check: a deliberately skipped candidate
        has a tombstone row, so a path is reclaimable iff it has a tombstone
        OR its source is no longer a candidate at all (excluded by pattern,
        or gone from disk — one ``stat`` replaces the former full
        re-parse/walk). The conservative default stands (#1130): a source
        still on disk with neither a live row nor a tombstone is a build gap
        — the FTS gap, not the source, is the anomaly — so its vectors are
        kept for the pass that next sees it.

        An unavailable source *root* (unmounted volume, not-yet-populated
        checkout) would make every per-path stat read as "gone" and wipe the
        whole sidecar, so it confirms nothing — the cheap root check replaces
        the former walk-incompleteness guard for this method.

        Args:
            stale_paths: Sidecar paths absent from the FTS chunk listing.

        Returns:
            The subset safe to reclaim, in input order.
        """
        if stale_paths and not _source_root_available(self._source_dir):
            logger.warning(
                "build_embeddings_converge_source_root_unavailable kept=%d "
                "outcome=no_vectors_removed",
                len(stale_paths),
            )
            return []
        confirmed: list[str] = []
        for path in stale_paths:
            if self._fts.get_tombstone(path) is not None:
                confirmed.append(path)
                continue
            if self._is_path_excluded(path) or self._stale_source_gone(path):
                # Deleted or excluded — the reclaim convergence exists for.
                confirmed.append(path)
                continue
            logger.warning(
                "build_embeddings_converge_kept path=%s outcome=vectors_kept",
                path,
            )
        return confirmed

    def _stale_source_gone(self, path: str) -> bool:
        """Return whether a stale sidecar path's source file is gone from disk.

        ``FileNotFoundError`` / ``NotADirectoryError`` mean the source is
        genuinely not there (or no longer a file), so its vectors are
        reclaimable. Any other ``OSError`` — a permission flap, an I/O error
        — is possibly transient and reads as *present*, so the conservative
        keep applies and the next pass re-checks (#1130).
        """
        try:
            st = (self._source_dir / path).stat()
        except (FileNotFoundError, NotADirectoryError):
            return True
        except OSError as exc:
            logger.debug(
                "build_embeddings_converge_stat_failed path=%s err=%s", path, exc
            )
            return False
        return not stat_module.S_ISREG(st.st_mode)

    def _loaded_store(self) -> EmbeddingStore | None:
        """Return the open store if this provider may write to it, opening nothing."""
        return self._slot.writable_current() if self._slot is not None else None

    def _open_store(self) -> EmbeddingStore:
        """Open (installing) the store for writing.

        An owned store that is corrupt or of another identity is rebuilt
        through :meth:`build_embeddings`.

        Returns:
            The writable store.

        Raises:
            RuntimeError: If called without a prior ``_require_vectors()``.
        """
        if self._slot is None or self._embedding_provider is None:
            raise RuntimeError("_require_vectors() must be called before _open_store()")
        rebuild = (
            (lambda: self.build_embeddings(force=True))
            if self._slot.owns_files
            else None
        )
        return self._slot.writable(rebuild=rebuild)

    # ------------------------------------------------------------------
    # Inline reindex embedding
    # ------------------------------------------------------------------

    def _embed_note_inline(self, vectors: EmbeddingStore, note: ParsedNote) -> int:
        """Reconcile one note's vectors with its just-written keyword rows (#930).

        The reconciler embeds in bounded batches, keeps the note's old rows
        when the provider fails or the store refuses a vector (#935, #1778),
        and records a conflicting write for re-queueing (#1768).

        Args:
            vectors: The store to write.
            note: The note whose keyword rows were just written.

        Returns:
            :data:`_EMBED_OK`, or :data:`_EMBED_KEPT` when the note's
            existing vectors were kept.
        """
        if self._reconciler is None:
            return _EMBED_OK
        result = self._reconciler.reconcile(vectors, [note.path], site="reindex")
        self._conflicts |= result.conflicts
        self._retries |= result.retry
        return _EMBED_KEPT if result.kept else _EMBED_OK

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

    def build_embeddings(self, *, force: bool = False) -> int:
        """Build or converge the vector index against the FTS index.

        Without ``force``, a non-empty persisted vector index is diffed
        against the chunks currently in the FTS ``sections`` table and
        reconciled (#665): chunks present in the FTS index but missing
        from the vector index are embedded and added, vectors for
        documents no longer in the FTS index are removed, and documents
        whose indexed content changed are re-embedded.  After every call
        the vector index mirrors the FTS chunk set, so external changes
        picked up by the boot reconciliation reindex (while no vector
        index was loaded) cannot accumulate as permanent semantic-search
        drift.  Work scales with the size of the diff, not the size of
        the vault.  An empty vector index falls through to the full cold
        build below.

        Args:
            force: If ``True``, rebuild from scratch even if a vector index
                already exists on disk (e.g. after changing the embedding
                model).

        Returns:
            Number of chunks successfully embedded. Any provider exception on
            a batch (a token-context rejection being the motivating case, but
            also transient API/network errors) is logged and the batch
            skipped, so this may be less than the total number of chunks
            attempted; if every batch is skipped, ``0`` is returned and no
            vectors are saved. A successful build with no embeddable chunks
            publishes an empty generation so stale vectors cannot survive a
            forced rebuild. On the convergence path this counts only the newly
            embedded chunks — a fully converged index returns ``0``.

        Raises:
            EmbeddingsNotConfiguredError: If ``embedding_provider`` is not
                configured. A ``ValueError``
                subclass, so callers may catch either; narrow to it to let
                genuine internal ``ValueError``s surface (#774).
        """
        self._require_vectors()
        if self._embedding_provider is None or self._slot is None:
            raise RuntimeError(
                "_require_vectors() must be called before build_embeddings()"
            )
        if force:
            store = self._slot.rebuild_target()
        else:
            # An owned store that will not open is rebuilt inside _open_store;
            # the convergence below then finds it up to date and embeds 0.
            store = self._open_store()
            if store.count > 0:
                return self._converge_embeddings(store)
            # Empty store: fall through to the full cold build.

        identity = self._slot.identity
        if self._reconciler is None:  # pragma: no cover - _require_vectors() ran
            raise RuntimeError("No reconciler without an embedding provider.")
        embedded, total = self._reconciler.rebuild(store, identity)

        if embedded > 0:
            logger.info("build_embeddings_saved chunks=%d", embedded)
        elif total > 0:
            # Every batch was skipped (e.g. provider down for the whole build,
            # or a dimension mismatch). Surface it loudly rather than as the
            # benign "nothing to embed" so an operator can tell an empty vault
            # apart from a wholesale embedding failure (#649).
            logger.warning(
                "build_embeddings_all_batches_failed total=%d outcome=no_vectors_saved",
                total,
            )
        elif not self._empty_embedding_build_is_authoritative(
            {row["path"] for row in self._fts.list_notes()}
        ):
            logger.warning(
                "build_embeddings_empty_not_authoritative outcome=no_vectors_saved"
            )
        else:
            store.publish_generation(identity, [])
            logger.info("build_embeddings_saved_empty_index")
        return embedded

    def _converge_embeddings(self, vectors: EmbeddingStore) -> int:
        """Reconcile a non-empty store with the keyword index (#665, #1768).

        Every document goes through the reconciler: one whose keyword rows
        and token match the store costs nothing, one whose input is
        unchanged but whose token moved (or a legacy row with no token) is
        re-tokened without a provider call, one whose input changed is
        re-embedded in full, and one the keyword index holds no rows for
        loses its vectors only when :meth:`_confirm_stale_vector_paths`
        confirms the absence (tombstone, exclusion or a source gone, #1130).
        A provider failure or a refused vector skips that document with its
        vectors kept, and the next run retries it (#649, #935). The store is
        checkpointed only when something changed.

        Thread-safety: runs on the single-owner
        :class:`~markdown_vault_mcp.indexing.IndexWriter` thread via
        :meth:`build_embeddings`, so no internal lock is required.

        Args:
            vectors: The open, non-empty store to reconcile.

        Returns:
            Number of chunks newly embedded (``0`` on a converged store).
        """
        if self._reconciler is None:  # pragma: no cover - _require_vectors() ran
            raise RuntimeError(
                "_require_vectors() must be called before _converge_embeddings()"
            )
        result = self._reconciler.reconcile(vectors, site="converge")
        self._conflicts |= result.conflicts
        self._retries |= result.retry
        if result.embedded or result.removed or result.retokened:
            vectors.checkpoint()
        if result.kept:
            logger.warning(
                "build_embeddings_converge_failed_chunks total=%d outcome=vectors_kept",
                result.kept,
            )
        logger.info(
            "build_embeddings_converged added=%d removed=%d retokened=%d up_to_date=%d",
            result.embedded,
            result.removed,
            result.retokened,
            result.up_to_date,
        )
        return result.embedded

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def embeddings_status(self) -> dict[str, Any]:
        """Return status information about the vector index.

        Returns:
            Dict with keys ``provider``, ``chunk_count``, ``path``,
            ``available``, and the store's ``backend``, ``identity`` and
            ``generation`` (``None`` until it is opened). ``path`` is
            ``None`` when the vectors are held in memory (#1708).

        Raises:
            OSError: If a sidecar file cannot be statted (#1625).
        """
        no_store = {"backend": None, "identity": None, "generation": None}
        if self._embedding_provider is None:
            # No provider, no store: nothing to describe.
            return {
                "available": False,
                "provider": None,
                "chunk_count": 0,
                "path": None,
                **no_store,
            }

        vectors = self._slot.current() if self._slot is not None else None
        count = 0
        if vectors is not None:
            count = vectors.count
        elif self._embeddings_path is not None:
            # Derive sidecar paths the way the store does (Path.with_suffix),
            # so an EMBEDDINGS_PATH that carries an
            # extension resolves to the real files instead of {path}.npy.npy
            # and misreporting chunk_count=0 (#819).
            # Both layouts keep the rows in the JSON: the legacy pair beside
            # {base}.npy and the store's commit file naming {base}.m<N>.npy.
            json_path = self._embeddings_path.with_suffix(".json")
            # A refused stat raises: a status of "no vectors" for a store the
            # server cannot read would be wrong (#1625).
            if path_exists(json_path):
                try:
                    with json_path.open(encoding="utf-8") as fh:
                        loaded_meta = json.load(fh)
                    if isinstance(loaded_meta, list):
                        count = len(loaded_meta)
                    else:
                        count = len(loaded_meta.get("rows", []))
                except (OSError, json.JSONDecodeError) as exc:
                    logger.warning(
                        "embeddings_status_metadata_read_failed path=%s err=%s",
                        json_path,
                        exc,
                    )

        return {
            "available": True,
            "provider": type(self._embedding_provider).__name__,
            "chunk_count": count,
            "path": str(self._embeddings_path) if self._embeddings_path else None,
            **(self._slot.describe() if self._slot is not None else no_store),
        }

    # ------------------------------------------------------------------
    # Deferred embedding flush
    # ------------------------------------------------------------------

    def flush_dirty_embeddings(self, paths: set[str]) -> None:
        """Reconcile each path in the snapshot and checkpoint the store once.

        Called only by the IndexWriter's ``FlushDirtyEmbeddings`` runner,
        after the keyword index was refreshed for these paths: every row
        comes from the keyword index, no file is read (#1768). A path whose
        input and token are unchanged costs nothing, so retrying a failed
        batch does not charge again for its already embedded siblings. A
        path the keyword index holds no rows for (deleted, excluded or
        tombstoned, #1129) loses its vectors when
        :meth:`_confirm_stale_vector_paths` confirms it. A provider failure
        or a refused vector keeps that path's rows and is logged; a
        conflicting write is recorded for re-queueing. Store open and
        rebuild errors propagate to the writer's Future.

        Args:
            paths: Paths to reconcile (relative to source_dir).
        """
        if self._reconciler is None or not paths:
            return
        vectors = self._open_store()
        result = self._reconciler.reconcile(vectors, paths, site="flush")
        self._conflicts |= result.conflicts
        self._retries |= result.retry
        # Checkpoint even when every row was current: a prior checkpoint may
        # have failed after the rows were published, and this retry persists them.
        vectors.checkpoint()
        logger.debug("deferred_embeddings_flushed paths=%d", len(paths))

    def take_embedding_retries(self) -> set[str]:
        """Return and forget the paths the provider failed for, at any site.

        Their rows were kept; the writer keeps them dirty so the next flush
        retries them, without queueing one of its own (#1464, #1768).

        Returns:
            The paths to keep dirty.
        """
        retries, self._retries = self._retries, set()
        return retries

    def take_embedding_conflicts(self) -> set[str]:
        """Return and forget the paths a conflicting write sent back for another pass.

        Returns:
            The paths to re-queue; empty when no write conflicted.
        """
        conflicts, self._conflicts = self._conflicts, set()
        return conflicts


def _source_root_available(source_dir: Path) -> bool:
    """Whether the vault root can be relied on to judge a source "gone".

    A root the server cannot stat is unavailable, like an unmounted one: every
    per-path check under it would read as "gone", so nothing is reclaimed
    (#1130, #1625).
    """
    try:
        return is_directory(source_dir)
    except OSError:
        return False
