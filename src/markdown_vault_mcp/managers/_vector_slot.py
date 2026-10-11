"""The vault's one vector store: opened lazily, rebuilt, closed (#1777).

The slot opens the store through its :class:`VectorBackend` on first use,
because the identity it opens against reads ``provider.dimension``, which
some providers answer with a network probe; a provider that is down must
not stop a vault from being built. On a store the vault owns, a corrupt
store or one holding another identity (a model change) is rebuilt into a
fresh generation through the caller's ``rebuild``; without a rebuild (a
process that does not own the files) both raise
:class:`~markdown_vault_mcp.exceptions.VectorIndexUnusableError`. A store
that is unavailable or of an unsupported schema always propagates. An
injected store is the caller's: never rebuilt on a fault, never closed.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Any

from markdown_vault_mcp.exceptions import (
    StoreCorruptError,
    StoreIncompatibleError,
    VectorIndexUnusableError,
)
from markdown_vault_mcp.managers._legacy_vector_writes import LegacyVectorWrites
from markdown_vault_mcp.types import EmbeddingIdentity

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from markdown_vault_mcp.interfaces import EmbeddingStore, VectorBackend
    from markdown_vault_mcp.providers import EmbeddingProvider

logger = logging.getLogger(__name__)


class VectorStoreSlot:
    """Holds the vault's vector store and decides what happens when it will not open.

    Args:
        backend: Opens the store.
        location: Where the store lives; ``None`` keeps it in memory.
        provider: Supplies the identity; ``None`` means no semantic search.
        text_format: The embedding-text format token.
        owns_files: ``False`` opens read-only.
        store: An already open store, owned by the caller.
    """

    def __init__(
        self,
        *,
        backend: VectorBackend,
        location: Path | None,
        provider: EmbeddingProvider | None,
        text_format: str,
        owns_files: bool = True,
        store: EmbeddingStore | None = None,
    ) -> None:
        self._backend = backend
        self._location = location
        self._provider = provider
        self._text_format = text_format
        self._owns_files = owns_files
        self._store = store
        self._injected = store is not None
        self._identity: EmbeddingIdentity | None = None
        self._legacy: LegacyVectorWrites | None = None

    @property
    def location(self) -> Path | None:
        """Where the store lives; ``None`` in memory."""
        return self._location

    @property
    def injected(self) -> bool:
        """Whether the caller passed the store in."""
        return self._injected

    @property
    def backend_name(self) -> str:
        """The backend's registry name."""
        return str(self._backend.name)

    @property
    def identity(self) -> EmbeddingIdentity:
        """The identity the store opens against, computed on first use.

        Raises:
            RuntimeError: If the slot has no provider.
        """
        if self._identity is None:
            if self._provider is None:
                raise RuntimeError("A vector store needs an embedding provider.")
            self._identity = EmbeddingIdentity.of(self._provider, self._text_format)
        return self._identity

    def current(self) -> EmbeddingStore | None:
        """Return the installed store, opening nothing."""
        return self._store

    def get(self, *, rebuild: Callable[[], object] | None) -> EmbeddingStore:
        """Return the store, opening it on first use.

        Args:
            rebuild: Rebuilds an owned store that is corrupt or holds another
                identity, installing the result (through :meth:`fresh`);
                ``None`` forbids it.

        Raises:
            VectorIndexUnusableError: If the store is corrupt or of another
                identity and no rebuild is allowed.
            ValueError: If a rebuild installs no store.
            StoreOpenError: If the store is unavailable or of an unsupported
                schema.
        """
        if self._store is not None:
            return self._store
        try:
            store = self._backend.open(
                self._location, self.identity, owns_files=self._owns_files
            )
        except (StoreCorruptError, StoreIncompatibleError) as exc:
            if rebuild is None:
                raise VectorIndexUnusableError(str(exc)) from exc
            logger.warning(
                "vector_store_rebuild_started location=%s reason=%s",
                self._location,
                type(exc).__name__,
            )
            rebuild()
            if self._store is None:
                raise ValueError(
                    "Failed to rebuild the vector store after it would not open."
                ) from exc
            return self._store
        self._store = store
        return store

    def replace(self, store: EmbeddingStore | None) -> None:
        """Install *store* (or clear the slot)."""
        self._store = store

    def replace_any(self, value: object) -> None:
        """Install a store, or the store under a :class:`LegacyVectorWrites`."""
        store = value.store if isinstance(value, LegacyVectorWrites) else value
        self._store = store  # type: ignore[assignment]

    def fresh(self) -> EmbeddingStore:
        """Install and return an empty store at the identity, to rebuild into.

        An owned slot opens a fresh store over the location and drops the
        old object without closing it, so its unsaved rows are not written
        over the rebuild. An injected store is emptied in place.
        """
        if self._injected and self._store is not None:
            self._store.publish_generation(self.identity, [])
            return self._store
        self._store = self._backend.open_fresh(self._location, self.identity)
        return self._store

    def legacy(self) -> LegacyVectorWrites | None:
        """Return the cached adapter over the installed store, or ``None``."""
        if self._store is None or self._provider is None:
            return None
        if self._legacy is None or self._legacy.store is not self._store:
            self._legacy = LegacyVectorWrites(self._store, self._provider)
        return self._legacy

    def legacy_get(self, *, rebuild: Callable[[], object] | None) -> LegacyVectorWrites:
        """Open (installing) and return the adapter over the store."""
        self.get(rebuild=rebuild)
        adapter = self.legacy()
        if adapter is None:  # pragma: no cover - get() installed a store
            raise RuntimeError("The vector store did not install.")
        return adapter

    def legacy_fresh(self) -> LegacyVectorWrites:
        """Install a fresh store and return the adapter over it."""
        self.fresh()
        adapter = self.legacy()
        if adapter is None:  # pragma: no cover - fresh() installed a store
            raise RuntimeError("The vector store did not install.")
        return adapter

    def describe(self) -> dict[str, Any]:
        """Return backend, identity and generation for ``embeddings_status``."""
        store = self._store
        return {
            "backend": self.backend_name,
            "identity": dataclasses.asdict(store.identity)
            if store is not None
            else None,
            "generation": store.generation if store is not None else None,
        }

    def close(self) -> None:
        """Close the store if the slot opened it; an injected store is left open."""
        if self._store is not None and not self._injected:
            self._store.close()
