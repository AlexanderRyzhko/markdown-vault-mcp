"""The numpy sidecar vector backend (#1767).

Rows live in memory as one float32 matrix with a parallel metadata list,
held together with the tokens, identity and generation in one immutable
snapshot. A writer builds a new snapshot and replaces the old one with a
single reference assignment, so a reader sees the old rows or the new
ones, never a mix.

On disk a store is ``{base}.json``, the commit point, and the matrix it
names, ``{base}.m{seq}.npy``. A persist writes the matrix under a new name
and then atomically replaces the JSON, so an interrupted persist leaves the
previous state whole. A sidecar written before #1767 (``{base}.npy`` beside
a JSON that names no matrix) opens as it is; its rows carry no token.

``replace_document`` and ``remove_document`` publish in memory at once and
become durable at :meth:`NumpySidecarStore.checkpoint` or ``close()``;
``publish_generation`` is durable before it returns.

Reachable open outcomes: *unavailable* (a parent that is not a directory,
a refused read), *corrupt* (a JSON or matrix that does not parse, a missing
matrix, rows and matrix disagreeing), *unsupported schema* (a
``format_version`` above :data:`FORMAT_VERSION`) and *incompatible
identity*. *Not configured* cannot happen: no location means in memory.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from markdown_vault_mcp.exceptions import ReadOnlyError
from markdown_vault_mcp.stores.registry import register_backend
from markdown_vault_mcp.types import ANY_TOKEN, EmbeddingIdentity, VectorRow
from markdown_vault_mcp.utils import folder_matches

try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _NUMPY_AVAILABLE = False

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence
    from pathlib import Path

    from markdown_vault_mcp.types import (
        DocumentRows,
        ExpectedToken,
        PublicationOutcome,
        RevisionToken,
        StoreFamily,
    )

logger = logging.getLogger(__name__)

#: The on-disk layout this code writes and the highest it reads.
FORMAT_VERSION = 2

_BACKEND_NAME = "numpy"


@dataclass(frozen=True)
class _Snapshot:
    """One committed state: what every read observes, whole."""

    identity: EmbeddingIdentity
    generation: int
    matrix: Any  # np.ndarray (n, dimension), float32, read-only
    rows: tuple[dict[str, Any], ...]
    tokens: Mapping[str, RevisionToken | None]


def _frozen(matrix: Any) -> Any:
    matrix.setflags(write=False)
    return matrix


def _fresh(identity: EmbeddingIdentity) -> _Snapshot:
    empty = _frozen(np.zeros((0, identity.dimension), dtype=np.float32))
    return _Snapshot(identity, 0, empty, (), MappingProxyType({}))


def _normalised(vectors: Any) -> Any:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vectors / norms).astype(np.float32, copy=False)


def _unit(vector: Any) -> Any:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0 else vector


def _row_vectors(rows: Sequence[VectorRow], dimension: int) -> Any:
    """Return *rows*' vectors as a normalised matrix, refusing bad ones."""
    if not rows:
        return np.zeros((0, dimension), dtype=np.float32)
    try:
        vectors = np.asarray([row.vector for row in rows], dtype=np.float32)
    except ValueError as exc:
        raise ValueError(f"Rows of unequal dimension: {exc}") from exc
    if vectors.ndim != 2 or vectors.shape[1] != dimension:
        raise ValueError(
            f"Embedding dimension mismatch: the store holds dimension {dimension}, "
            f"the rows have {vectors.shape[-1]}."
        )
    if not np.isfinite(vectors).all():
        raise ValueError(
            "Embedding values must be finite; a row holds NaN or infinity."
        )
    return _normalised(vectors)


def _metadata(document: DocumentRows) -> list[dict[str, Any]]:
    return [{**row.metadata, "path": document.path} for row in document.rows]


class NumpySidecarStore:
    """A vector store held in memory, optionally persisted beside a base path.

    Construct through :class:`NumpySidecarBackend`.
    """

    def __init__(
        self,
        snapshot: _Snapshot,
        *,
        base: Path | None,
        read_only: bool,
        seq: int,
        matrix_path: Path | None,
    ) -> None:
        self._snap = snapshot
        self._base = base
        self._read_only = read_only
        self._seq = seq
        self._matrix_path = matrix_path
        self._dirty = False
        self._closed = False
        self._lock = threading.Lock()

    # -- reading -------------------------------------------------------------

    def _current(self) -> _Snapshot:
        if self._closed:
            raise RuntimeError("The vector store is closed.")
        return self._snap

    @property
    def identity(self) -> EmbeddingIdentity:
        """The embedding space of the stored rows."""
        return self._current().identity

    @property
    def generation(self) -> int:
        """Bumped by each :meth:`publish_generation`."""
        return self._current().generation

    @property
    def count(self) -> int:
        """The number of stored rows."""
        return len(self._current().rows)

    def search(
        self,
        vector: Sequence[float],
        *,
        limit: int = 10,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
        folder: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return the rows nearest *vector*; see :class:`~markdown_vault_mcp.interfaces.VectorReader`.

        Raises:
            ValueError: If *vector* is not of the store's dimension.
        """
        snap = self._current()
        if not snap.rows or limit <= 0:
            return []
        query = np.asarray(vector, dtype=np.float32)
        if query.shape != (snap.matrix.shape[1],):
            raise ValueError(
                f"Query dimension mismatch: the store holds dimension "
                f"{snap.matrix.shape[1]}, the query has {query.shape[-1]}."
            )
        scores = snap.matrix @ _unit(query)
        results: list[dict[str, Any]] = []
        for index in np.argsort(scores)[::-1]:
            i = int(index)
            row = snap.rows[i]
            if folder is not None and not folder_matches(row.get("folder", ""), folder):
                continue
            if predicate is not None and not predicate(row):
                continue
            results.append({**row, "score": float(scores[i])})
            if len(results) >= limit:
                break
        return results

    def similar_to_document(
        self,
        path: str,
        *,
        limit: int = 10,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the rows nearest the mean of *path*'s rows, excluding *path*."""
        snap = self._current()
        if not snap.rows or limit <= 0:
            return []
        own = [i for i, row in enumerate(snap.rows) if row.get("path") == path]
        if not own:
            return []
        scores = snap.matrix @ _unit(np.mean(snap.matrix[own], axis=0))
        candidates = [
            (float(score), i)
            for i, score in enumerate(scores)
            if snap.rows[i].get("path") != path
            and (predicate is None or predicate(snap.rows[i]))
        ]
        candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        return [{**snap.rows[i], "score": score} for score, i in candidates[:limit]]

    def rows_for(self, path: str) -> list[VectorRow]:
        """Return *path*'s rows, vectors normalised, in storage order."""
        snap = self._current()
        return [
            VectorRow(vector=snap.matrix[i].tolist(), metadata=dict(row))
            for i, row in enumerate(snap.rows)
            if row.get("path") == path
        ]

    def tokens(self) -> dict[str, RevisionToken | None]:
        """Return each stored document's token; ``None`` when unknown."""
        snap = self._current()
        found: dict[str, RevisionToken | None] = {
            row["path"]: None for row in snap.rows if "path" in row
        }
        found.update(snap.tokens)
        return found

    # -- publishing ----------------------------------------------------------

    def _writable(self) -> _Snapshot:
        snap = self._current()
        if self._read_only:
            raise ReadOnlyError(
                "This vector store was opened read-only: another process owns its files."
            )
        return snap

    @staticmethod
    def _conflicts(snap: _Snapshot, path: str, expected: ExpectedToken) -> bool:
        return expected is not ANY_TOKEN and snap.tokens.get(path) != expected

    def _swap(
        self,
        snap: _Snapshot,
        matrix: Any,
        rows: tuple[dict[str, Any], ...],
        tokens: dict[str, RevisionToken | None],
    ) -> None:
        self._snap = _Snapshot(
            snap.identity,
            snap.generation,
            _frozen(matrix),
            rows,
            MappingProxyType(tokens),
        )
        self._dirty = True

    def replace_document(
        self, document: DocumentRows, *, expected_token: ExpectedToken
    ) -> PublicationOutcome:
        """Replace ``document.path``'s rows; see :class:`~markdown_vault_mcp.interfaces.VectorPublisher`.

        Raises:
            ReadOnlyError: If the store was opened read-only.
            ValueError: If a row is of another dimension or not finite.
        """
        with self._lock:
            snap = self._writable()
            if self._conflicts(snap, document.path, expected_token):
                return "conflict"
            vectors = _row_vectors(document.rows, snap.identity.dimension)
            keep = [
                i for i, row in enumerate(snap.rows) if row.get("path") != document.path
            ]
            matrix = np.vstack([snap.matrix[keep], vectors])
            rows = tuple(snap.rows[i] for i in keep) + tuple(_metadata(document))
            tokens = {**snap.tokens, document.path: document.token}
            self._swap(snap, matrix, rows, tokens)
        return "published"

    def remove_document(
        self, path: str, *, expected_token: ExpectedToken
    ) -> PublicationOutcome:
        """Remove *path*'s rows and token, with the same condition.

        Raises:
            ReadOnlyError: If the store was opened read-only.
        """
        with self._lock:
            snap = self._writable()
            if self._conflicts(snap, path, expected_token):
                return "conflict"
            keep = [i for i, row in enumerate(snap.rows) if row.get("path") != path]
            if len(keep) == len(snap.rows) and path not in snap.tokens:
                return "published"
            tokens = {k: v for k, v in snap.tokens.items() if k != path}
            self._swap(
                snap, snap.matrix[keep], tuple(snap.rows[i] for i in keep), tokens
            )
        return "published"

    def publish_generation(
        self, identity: EmbeddingIdentity, documents: Iterable[DocumentRows]
    ) -> None:
        """Replace the store with *documents* under *identity*, durably.

        Raises:
            ReadOnlyError: If the store was opened read-only.
            ValueError: If a path appears twice, or a row is of another
                dimension or not finite.
        """
        docs = list(documents)
        seen: set[str] = set()
        for document in docs:
            if document.path in seen:
                raise ValueError(f"publish_generation got {document.path} twice.")
            seen.add(document.path)
        blocks = [_row_vectors(d.rows, identity.dimension) for d in docs]
        matrix = (
            np.vstack(blocks)
            if blocks
            else np.zeros((0, identity.dimension), dtype=np.float32)
        )
        rows = tuple(meta for d in docs for meta in _metadata(d))
        tokens = MappingProxyType({d.path: d.token for d in docs})
        with self._lock:
            snap = self._writable()
            new = _Snapshot(
                identity, snap.generation + 1, _frozen(matrix), rows, tokens
            )
            self._persist(new)
            self._snap = new
            self._dirty = False
        logger.info(
            "numpy_store_generation_published generation=%d rows=%d",
            new.generation,
            len(rows),
        )

    # -- lifecycle -----------------------------------------------------------

    def _persist(self, snap: _Snapshot) -> None:  # noqa: ARG002
        """Write *snap* to disk; a no-op in memory (Task 4 adds files)."""
        if self._base is None:
            return
        raise NotImplementedError("file locations arrive in Task 4")

    def checkpoint(self) -> None:
        """Persist pending changes; a no-op in memory, read-only or clean.

        Raises:
            RuntimeError: If the store is closed.
        """
        self._current()
        if self._base is None or self._read_only:
            return
        with self._lock:
            if not self._dirty:
                return
            self._persist(self._snap)
            self._dirty = False

    def close(self) -> None:
        """Persist pending changes and release the store. Idempotent.

        The store is closed afterwards even if the final persist raises.
        """
        if self._closed:
            return
        try:
            self.checkpoint()
        finally:
            self._closed = True


class NumpySidecarBackend:
    """Open a :class:`NumpySidecarStore` at a base path, or in memory."""

    family: StoreFamily = "vector"
    name = _BACKEND_NAME

    def open(
        self,
        location: Path | str | None,
        identity: object | None = None,
        *,
        owns_files: bool = True,  # noqa: ARG002
    ) -> NumpySidecarStore:
        """Open the store.

        Args:
            location: The base path of the sidecar files, or ``None`` /
                ``":memory:"`` for a store held in memory.
            identity: The :class:`~markdown_vault_mcp.types.EmbeddingIdentity`
                the rows must belong to.
            owns_files: ``False`` opens files read-only; ignored in memory.

        Returns:
            The open store.

        Raises:
            ImportError: If numpy is not installed.
            ValueError: If *identity* is not an ``EmbeddingIdentity``.
        """
        if not _NUMPY_AVAILABLE:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "NumpySidecarBackend requires 'numpy'. "
                "Install it with: pip install 'markdown-vault-mcp[embeddings]'"
            )
        if not isinstance(identity, EmbeddingIdentity):
            raise ValueError("A vector store opens against an EmbeddingIdentity.")
        if location is None or str(location) == ":memory:":
            return NumpySidecarStore(
                _fresh(identity), base=None, read_only=False, seq=0, matrix_path=None
            )
        raise NotImplementedError("file locations arrive in Task 4")


register_backend("vector", _BACKEND_NAME, NumpySidecarBackend)
