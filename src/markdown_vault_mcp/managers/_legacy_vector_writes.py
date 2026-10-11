"""The released ``VectorStore`` calls on top of an ``EmbeddingStore`` (#1777).

``IndexManager`` and ``EmbeddingsManager`` still mutate vectors through
``add``, ``add_vectors``, ``delete_by_path`` and ``save``. Until #1778 moves
them to the publisher, this adapter maps those calls onto
``replace_document``, ``remove_document`` and ``checkpoint``. Rows it
publishes carry the document's existing token (``None`` for a new one).
Temporary: #1778 deletes this module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from markdown_vault_mcp.embed_text import is_embeddable
from markdown_vault_mcp.types import ANY_TOKEN, DocumentRows, VectorRow

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from markdown_vault_mcp.interfaces import EmbeddingStore
    from markdown_vault_mcp.providers import EmbeddingProvider


class LegacyVectorWrites:
    """``VectorStore``-shaped access to an ``EmbeddingStore``.

    Args:
        store: The store the calls land on.
        provider: Embeds the texts of :meth:`add` and the queries of
            :meth:`search`.
    """

    def __init__(self, store: EmbeddingStore, provider: EmbeddingProvider) -> None:
        self._store = store
        self._provider = provider

    @property
    def store(self) -> EmbeddingStore:
        """The store underneath."""
        return self._store

    @property
    def count(self) -> int:
        """The number of stored rows."""
        return self._store.count

    def chunks_by_path(self) -> dict[str, list[dict[str, Any]]]:
        """Return the stored metadata grouped by path, documents with rows only."""
        grouped: dict[str, list[dict[str, Any]]] = {}
        for path in self._store.tokens():
            rows = self._store.rows_for(path)
            if rows:
                grouped[path] = [row.metadata for row in rows]
        return grouped

    def add(self, texts: list[str], metadata: list[dict[str, Any]]) -> int:
        """Embed *texts* and append them against *metadata*."""
        if not texts:
            return 0
        return self.add_vectors(self._provider.embed(texts), metadata)

    def add_vectors(
        self, raw_vectors: list[list[float]], metadata: list[dict[str, Any]]
    ) -> int:
        """Append *raw_vectors* to their documents' rows.

        Raises:
            ValueError: If the lengths differ or any vector is not of the
                store's dimension; nothing changes then.
        """
        if len(raw_vectors) != len(metadata):
            raise ValueError(
                f"raw_vectors and metadata must have the same length "
                f"(got {len(raw_vectors)} vs {len(metadata)})"
            )
        dimension = self._store.identity.dimension
        wrong = [len(v) for v in raw_vectors if len(v) != dimension]
        if wrong:
            raise ValueError(
                f"Embedding dimension mismatch: the store holds dimension {dimension}, "
                f"a vector has {wrong[0]}."
            )
        by_path: dict[str, list[VectorRow]] = {}
        for vector, meta in zip(raw_vectors, metadata, strict=True):
            by_path.setdefault(meta["path"], []).append(VectorRow(vector, dict(meta)))
        tokens = self._store.tokens()
        for path, new in by_path.items():
            rows = [*self._store.rows_for(path), *new]
            self._store.replace_document(
                DocumentRows(path, tokens.get(path), rows), expected_token=ANY_TOKEN
            )
        return len(raw_vectors)

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[dict[str, Any]]:
        """Embed *query* and return the nearest rows."""
        if not is_embeddable(query) or self._store.count == 0:
            return []
        vector = self._provider.embed_query([query])[0]
        return self._store.search(vector, limit=limit, predicate=predicate)

    def search_by_path(
        self,
        path: str,
        *,
        limit: int = 10,
        predicate: Callable[[dict[str, Any]], bool] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the rows nearest *path*'s rows, excluding *path*."""
        return self._store.similar_to_document(path, limit=limit, predicate=predicate)

    def delete_by_path(self, path: str) -> int:
        """Remove *path*'s rows; return how many there were."""
        removed = len(self._store.rows_for(path))
        self._store.remove_document(path, expected_token=ANY_TOKEN)
        return removed

    def save(self, path: Path) -> None:  # noqa: ARG002 - the store knows its location
        """Make the store durable; *path* is ignored."""
        self._store.checkpoint()
