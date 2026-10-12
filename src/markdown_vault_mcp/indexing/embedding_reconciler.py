"""The one writer of embedding rows (#1768).

Rows come from the keyword index, never from a source file: each
document's chunks become embedding inputs, tagged with the document's
``content_hash`` as its token. Per document the reconciler compares those
inputs and that token with what the store holds and decides: nothing,
re-token (the input is unchanged and only the token moved, or a legacy row
carries no token: the stored vectors are republished, no provider call),
embed, or remove. It embeds outside any store call and publishes with the
token it observed, so a write that crosses another writer's comes back as
a conflict, which the caller re-queues. Whether a path the keyword index
holds no rows for may lose its vectors stays with the caller
(``confirm_removal``) until #1785.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from markdown_vault_mcp.embed_text import is_embeddable
from markdown_vault_mcp.types import DocumentRows, VectorRow

if TYPE_CHECKING:
    from collections.abc import Callable, Collection

    from markdown_vault_mcp.embed_text import EmbedTextBuilder
    from markdown_vault_mcp.interfaces import EmbeddingStore, KeywordIndex
    from markdown_vault_mcp.providers import EmbeddingProvider
    from markdown_vault_mcp.types import RevisionToken

logger = logging.getLogger(__name__)

#: At or above this many re-tokened documents in one pass, one generation
#: publish (one matrix copy and one file write) replaces the per-document
#: replaces, each of which copies the whole matrix (#1768).
_BULK_RETOKEN = 64

_METADATA_KEYS = ("path", "title", "folder", "heading", "content", "start_line")


@dataclass
class ReconcileResult:
    """What one pass did, for the caller's aggregate log line.

    Attributes:
        embedded: Chunks embedded and published.
        removed: Rows removed.
        retokened: Documents republished under a new token without embedding.
        kept: Chunks left as they were: the provider failed or the store
            refused a vector.
        up_to_date: Chunks already converged.
        conflicts: Paths another writer changed underneath; the caller
            re-queues them.
    """

    embedded: int = 0
    removed: int = 0
    retokened: int = 0
    kept: int = 0
    up_to_date: int = 0
    conflicts: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _Document:
    """One document's embedding inputs, from the keyword index."""

    path: str
    token: RevisionToken
    texts: tuple[str, ...]
    metadata: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _Pass:
    """The state one reconcile pass shares between its steps."""

    store: EmbeddingStore
    stored: dict[str, RevisionToken | None]
    result: ReconcileResult
    site: str


def _signature(rows: Collection[dict[str, Any]]) -> Counter[tuple[Any, ...]]:
    """The chunk identity two row sets are compared by (#665, #1087).

    ``start_line`` participates so that line-shift-only edits (content
    identical, positions moved) still refresh vector metadata; the cost is
    re-embedding that one doc's chunks, and embeddings are deterministic for
    identical content. The preamble participates so an offline summary-only
    edit (content identical, searchable frontmatter changed) still
    re-embeds; ``or ""`` normalisation keeps legacy vector rows without the
    key equal to fresh rows under default (v1) config.
    """
    return Counter(
        (
            r.get("title"),
            r.get("heading"),
            r.get("content"),
            r.get("start_line"),
            r.get("preamble") or "",
        )
        for r in rows
    )


class EmbeddingReconciler:
    """Publishes every embedding row, one document at a time.

    Args:
        fts: The keyword index, the source of every row.
        provider: Embeds the documents that need it.
        embed_builder: Builds each chunk's embedding text.
        batch_size: Most texts per provider call.
        confirm_removal: Given paths the store holds and the keyword index
            has no rows for, returns those whose vectors may go.
    """

    def __init__(
        self,
        *,
        fts: KeywordIndex,
        provider: EmbeddingProvider,
        embed_builder: EmbedTextBuilder,
        batch_size: int,
        confirm_removal: Callable[[list[str]], list[str]],
    ) -> None:
        self._fts = fts
        self._provider = provider
        self._builder = embed_builder
        self._batch_size = batch_size
        self._confirm_removal = confirm_removal

    def documents(self, paths: Collection[str] | None = None) -> dict[str, _Document]:
        """Read *paths*' (or every) document's embedding inputs from the keyword index.

        Args:
            paths: The documents to read; ``None`` for every document.

        Returns:
            Each document's inputs by path. A document the keyword index
            holds rows for but with nothing embeddable (#1087) is present,
            with no texts.
        """
        texts: dict[str, list[str]] = {}
        metadata: dict[str, list[dict[str, Any]]] = {}
        tokens: dict[str, RevisionToken] = {}
        for row in self._fts.list_chunks(paths=paths):
            path = row["path"]
            tokens[path] = row["content_hash"]
            texts.setdefault(path, [])
            metadata.setdefault(path, [])
            preamble = (
                self._builder.fields_text(row.get("frontmatter_json"))
                if row.get("is_first_chunk")
                else ""
            )
            text = self._builder.build(
                title=row["title"],
                heading=row["heading"],
                content=row["content"],
                fields_text=preamble,
                is_first_chunk=bool(row.get("is_first_chunk")),
            )
            if not is_embeddable(text):
                continue  # #1087: a blank input never enters the signature
            texts[path].append(text)
            metadata[path].append(
                {**{k: row.get(k) for k in _METADATA_KEYS}, "preamble": preamble}
            )
        return {
            path: _Document(
                path, tokens[path], tuple(texts[path]), tuple(metadata[path])
            )
            for path in tokens
        }

    def reconcile(
        self,
        store: EmbeddingStore,
        paths: Collection[str] | None = None,
        *,
        site: str,
    ) -> ReconcileResult:
        """Bring *paths* (or every document) in the store in line with the keyword index.

        Removals run first, then each document's re-embed, then the
        re-tokens, so a bulk re-token publish sees the pass's other writes.

        Args:
            store: The store to write.
            paths: The documents to reconcile; ``None`` for a whole-store pass.
            site: Which caller, for the log lines.

        Returns:
            What the pass did; ``conflicts`` holds the paths to re-queue.
        """
        docs = self.documents(paths)
        stored = store.tokens()
        held = set(stored) if paths is None else set(paths) & set(stored)
        state = _Pass(store, stored, ReconcileResult(), site)
        self._remove_unbacked(state, sorted(held - set(docs)))
        retoken: list[_Document] = []
        for path in sorted(docs):
            if self._needs_retoken(state, docs[path]):
                retoken.append(docs[path])
        self._retoken(state, retoken)
        return state.result

    def _remove_unbacked(self, state: _Pass, paths: list[str]) -> None:
        """Remove the vectors of *paths* the caller confirms may go."""
        if not paths:
            return
        for path in self._confirm_removal(paths):
            self._remove(state, path)

    def _remove(self, state: _Pass, path: str) -> None:
        """Remove *path*'s rows, conditional on the token observed."""
        removed = len(state.store.rows_for(path))
        outcome = state.store.remove_document(
            path, expected_token=state.stored.get(path)
        )
        if outcome == "conflict":
            self._conflict(state, path)
            return
        state.result.removed += removed

    def _needs_retoken(self, state: _Pass, doc: _Document) -> bool:
        """Decide *doc*'s case and act on it; return whether it needs a re-token."""
        current = [dict(r.metadata) for r in state.store.rows_for(doc.path)]
        if not doc.texts:
            if current:
                self._remove(state, doc.path)
            return False
        if current and _signature(current) == _signature(doc.metadata):
            if state.stored.get(doc.path) == doc.token:
                state.result.up_to_date += len(doc.texts)
                return False
            return True
        self._embed_and_publish(state, doc)
        return False

    def _embed_and_publish(self, state: _Pass, doc: _Document) -> None:
        """Embed *doc* and replace its rows, keeping them on any failure."""
        try:
            vectors = self._embed(doc)
        except Exception as exc:
            # Broad by design: providers raise heterogeneous types for a
            # timeout, an oversized batch or an outage. The rows stay.
            state.result.kept += len(doc.texts)
            logger.warning(
                "embedding_reconcile_skip_doc site=%s path=%s chunks=%d err=%s",
                state.site,
                doc.path,
                len(doc.texts),
                exc,
                exc_info=True,
            )
            return
        try:
            rows = [
                VectorRow(v, dict(m))
                for v, m in zip(vectors, doc.metadata, strict=True)
            ]
            outcome = state.store.replace_document(
                DocumentRows(doc.path, doc.token, rows),
                expected_token=state.stored.get(doc.path),
            )
        except ValueError as exc:
            # The store refuses another dimension, a NaN or a wrong count
            # before it changes anything, so the old rows stay (#1778).
            state.result.kept += len(doc.texts)
            logger.warning(
                "embedding_reconcile_refused site=%s path=%s err=%s",
                state.site,
                doc.path,
                exc,
            )
            return
        if outcome == "conflict":
            self._conflict(state, doc.path)
            return
        state.result.embedded += len(doc.texts)

    def _embed(self, doc: _Document) -> list[list[float]]:
        """Embed *doc*'s texts in batches of at most ``batch_size``."""
        vectors: list[list[float]] = []
        for start in range(0, len(doc.texts), self._batch_size):
            batch = list(doc.texts[start : start + self._batch_size])
            vectors.extend(self._provider.embed(batch))
        return vectors

    def _retoken(self, state: _Pass, docs: list[_Document]) -> None:
        """Republish *docs*' stored vectors under their new tokens.

        Below :data:`_BULK_RETOKEN` each is a conditional replace; at or
        above it, one generation holding every document's current rows.
        The bulk publish skips the per-document token check, which is safe
        for a store with one writer per location, as the numpy store is.
        """
        if len(docs) < _BULK_RETOKEN:
            for doc in docs:
                rows = state.store.rows_for(doc.path)
                outcome = state.store.replace_document(
                    DocumentRows(doc.path, doc.token, rows),
                    expected_token=state.stored.get(doc.path),
                )
                if outcome == "conflict":
                    self._conflict(state, doc.path)
                else:
                    state.result.retokened += 1
            return
        new_tokens = {doc.path: doc.token for doc in docs}
        state.store.publish_generation(
            state.store.identity,
            [
                DocumentRows(
                    path, new_tokens.get(path, token), state.store.rows_for(path)
                )
                for path, token in state.store.tokens().items()
            ],
        )
        state.result.retokened += len(docs)

    @staticmethod
    def _conflict(state: _Pass, path: str) -> None:
        """Record that another writer changed *path* underneath this pass."""
        state.result.conflicts.add(path)
        logger.info("embedding_reconcile_conflict site=%s path=%s", state.site, path)
