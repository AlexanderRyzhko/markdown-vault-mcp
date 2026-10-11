"""The numpy sidecar vector backend (#1767)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from markdown_vault_mcp.exceptions import ReadOnlyError
from markdown_vault_mcp.interfaces import StoreBackend
from markdown_vault_mcp.stores.numpy_sidecar import NumpySidecarBackend
from markdown_vault_mcp.types import ANY_TOKEN
from tests.fixtures.vector_store_contract import IDENT, VectorStoreContract, doc

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


class TestBackend:
    def test_names_itself_and_is_registered(self) -> None:
        from markdown_vault_mcp.stores.registry import get_backend

        backend = NumpySidecarBackend()
        assert isinstance(backend, StoreBackend)
        assert (backend.family, backend.name) == ("vector", "numpy")
        assert get_backend("vector", "numpy") is NumpySidecarBackend

    def test_an_identity_is_required(self) -> None:
        backend = NumpySidecarBackend()
        with pytest.raises(ValueError, match="EmbeddingIdentity"):
            backend.open(None)

    @pytest.mark.parametrize("location", [None, ":memory:"])
    def test_in_memory_writes_no_file(
        self, tmp_path: Path, location: str | None
    ) -> None:
        store = NumpySidecarBackend().open(location, IDENT)
        store.replace_document(doc("a.md", [[1, 0, 0]]), expected_token=ANY_TOKEN)
        store.checkpoint()
        store.close()
        assert list(tmp_path.iterdir()) == []

    def test_in_memory_is_writable_when_files_are_not_owned(self) -> None:
        store = NumpySidecarBackend().open(None, IDENT, owns_files=False)
        assert (
            store.replace_document(doc("a.md", [[1, 0, 0]]), expected_token=ANY_TOKEN)
            == "published"
        )


class TestInMemoryConformance(VectorStoreContract):
    @pytest.fixture
    def make_store(self) -> Callable[..., Any]:
        return lambda identity: NumpySidecarBackend().open(None, identity)


class TestClosed:
    def test_use_after_close_raises(self) -> None:
        store = NumpySidecarBackend().open(None, IDENT)
        store.close()
        store.close()  # idempotent
        with pytest.raises(RuntimeError, match="closed"):
            store.search([1, 0, 0])
        with pytest.raises(RuntimeError, match="closed"):
            store.checkpoint()

    @pytest.mark.skip(reason="file locations arrive in Task 4")
    def test_writes_to_a_read_only_store_are_refused(self, tmp_path: Path) -> None:
        store = NumpySidecarBackend().open(tmp_path / "vec", IDENT, owns_files=False)
        with pytest.raises(ReadOnlyError, match="read-only"):
            store.replace_document(doc("a.md", [[1, 0, 0]]), expected_token=ANY_TOKEN)
