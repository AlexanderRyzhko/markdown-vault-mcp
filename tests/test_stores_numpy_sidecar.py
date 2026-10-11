"""The numpy sidecar vector backend (#1767)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from markdown_vault_mcp.exceptions import (
    ReadOnlyError,
    StoreCorruptError,
    StoreIncompatibleError,
    StoreUnavailableError,
    StoreUnsupportedSchemaError,
)
from markdown_vault_mcp.interfaces import StoreBackend
from markdown_vault_mcp.stores import numpy_sidecar
from markdown_vault_mcp.stores.numpy_sidecar import NumpySidecarBackend
from markdown_vault_mcp.types import ANY_TOKEN, EmbeddingIdentity
from tests.conftest import MockEmbeddingProvider
from tests.fixtures.store_contract import StoreLifecycleContract
from tests.fixtures.vector_store_contract import (
    IDENT,
    PersistentVectorStoreContract,
    VectorStoreContract,
    doc,
)

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

    def test_writes_to_a_read_only_store_are_refused(self, tmp_path: Path) -> None:
        store = NumpySidecarBackend().open(tmp_path / "vec", IDENT, owns_files=False)
        with pytest.raises(ReadOnlyError, match="read-only"):
            store.replace_document(doc("a.md", [[1, 0, 0]]), expected_token=ANY_TOKEN)


def _base(tmp_path: Path) -> Path:
    return tmp_path / "vectors" / "idx-embeddings"


class TestFileConformance(VectorStoreContract, PersistentVectorStoreContract):
    @pytest.fixture
    def make_store(self, tmp_path: Path) -> Callable[..., Any]:
        return lambda identity: NumpySidecarBackend().open(_base(tmp_path), identity)

    @pytest.fixture
    def reopen_store(self, tmp_path: Path) -> Callable[..., Any]:
        return lambda identity, owns: NumpySidecarBackend().open(
            _base(tmp_path), identity, owns_files=owns
        )


class TestFileLifecycle(StoreLifecycleContract):
    @pytest.fixture
    def open_store(self, tmp_path: Path) -> Callable[[], Any]:
        return lambda: NumpySidecarBackend().open(_base(tmp_path), IDENT)

    @pytest.fixture
    def open_store_read_only(self, tmp_path: Path) -> Callable[[], Any]:
        return lambda: NumpySidecarBackend().open(
            _base(tmp_path), IDENT, owns_files=False
        )

    @pytest.fixture
    def touch_store(self) -> Callable[[Any], None]:
        return lambda store: store.count

    @pytest.fixture
    def write_to_store(self) -> Callable[[Any], None]:
        return lambda store: store.replace_document(
            doc("a.md", [[1, 0, 0]]), expected_token=ANY_TOKEN
        )

    @pytest.fixture
    def closed_error(self) -> type[Exception]:
        return RuntimeError


class TestLayout:
    def test_a_checkpoint_writes_the_commit_file_and_one_matrix(
        self, tmp_path: Path
    ) -> None:
        base = _base(tmp_path)
        store = NumpySidecarBackend().open(base, IDENT)
        store.replace_document(
            doc("a.md", [[1, 0, 0]], token="ta"), expected_token=ANY_TOKEN
        )
        store.checkpoint()
        assert sorted(p.name for p in base.parent.iterdir()) == [
            "idx-embeddings.json",
            "idx-embeddings.m1.npy",
        ]
        payload = json.loads(base.with_suffix(".json").read_text())
        assert payload["format_version"] == 2
        assert payload["index_metadata"]["matrix"] == "idx-embeddings.m1.npy"
        assert payload["tokens"] == {"a.md": "ta"}

    def test_the_next_checkpoint_replaces_the_matrix(self, tmp_path: Path) -> None:
        base = _base(tmp_path)
        store = NumpySidecarBackend().open(base, IDENT)
        store.replace_document(doc("a.md", [[1, 0, 0]]), expected_token=ANY_TOKEN)
        store.checkpoint()
        store.replace_document(doc("b.md", [[0, 1, 0]]), expected_token=ANY_TOKEN)
        store.checkpoint()
        assert sorted(p.name for p in base.parent.iterdir()) == [
            "idx-embeddings.json",
            "idx-embeddings.m2.npy",
        ]

    def test_a_clean_checkpoint_writes_nothing(self, tmp_path: Path) -> None:
        base = _base(tmp_path)
        store = NumpySidecarBackend().open(base, IDENT)
        store.checkpoint()
        assert not base.parent.exists()

    def test_a_read_only_store_never_writes(self, tmp_path: Path) -> None:
        base = _base(tmp_path)
        store = NumpySidecarBackend().open(base, IDENT, owns_files=False)
        store.checkpoint()
        store.close()
        assert not base.parent.exists()


class TestOpenOutcomes:
    def _persisted(self, tmp_path: Path) -> Path:
        base = _base(tmp_path)
        store = NumpySidecarBackend().open(base, IDENT)
        store.replace_document(doc("a.md", [[1, 0, 0]]), expected_token=ANY_TOKEN)
        store.close()
        return base

    def test_no_files_open_an_empty_store(self, tmp_path: Path) -> None:
        store = NumpySidecarBackend().open(_base(tmp_path), IDENT, owns_files=False)
        assert (store.count, store.generation) == (0, 0)

    def test_a_parent_that_is_a_file_is_unavailable(self, tmp_path: Path) -> None:
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        backend = NumpySidecarBackend()
        with pytest.raises(StoreUnavailableError) as info:
            backend.open(blocker / "vec", IDENT)
        assert (info.value.backend, info.value.location) == (
            "numpy",
            str(blocker / "vec"),
        )
        assert blocker.read_text() == "not a directory"

    def test_a_newer_format_is_unsupported(self, tmp_path: Path) -> None:
        base = self._persisted(tmp_path)
        json_path = base.with_suffix(".json")
        payload = json.loads(json_path.read_text())
        payload["format_version"] = 99
        json_path.write_text(json.dumps(payload))
        backend = NumpySidecarBackend()
        with pytest.raises(StoreUnsupportedSchemaError):
            backend.open(base, IDENT)

    def test_garbage_json_is_corrupt_and_left_alone(self, tmp_path: Path) -> None:
        base = self._persisted(tmp_path)
        json_path = base.with_suffix(".json")
        json_path.write_text("{not json")
        backend = NumpySidecarBackend()
        with pytest.raises(StoreCorruptError):
            backend.open(base, IDENT)
        assert json_path.read_text() == "{not json"

    def test_a_matrix_without_json_is_corrupt(self, tmp_path: Path) -> None:
        base = _base(tmp_path)
        base.parent.mkdir(parents=True)
        np.save(base.with_suffix(".npy"), np.zeros((1, 3), dtype=np.float32))
        backend = NumpySidecarBackend()
        with pytest.raises(StoreCorruptError):
            backend.open(base, IDENT)

    def test_json_naming_a_missing_matrix_is_corrupt(self, tmp_path: Path) -> None:
        base = self._persisted(tmp_path)
        base.with_suffix(".m1.npy").unlink()
        backend = NumpySidecarBackend()
        with pytest.raises(StoreCorruptError):
            backend.open(base, IDENT)

    def test_rows_and_matrix_disagreeing_is_corrupt(self, tmp_path: Path) -> None:
        base = self._persisted(tmp_path)
        np.save(base.with_suffix(".m1.npy"), np.zeros((2, 3), dtype=np.float32))
        backend = NumpySidecarBackend()
        with pytest.raises(StoreCorruptError, match="rows"):
            backend.open(base, IDENT)

    def test_a_matrix_name_with_a_directory_is_corrupt(self, tmp_path: Path) -> None:
        base = self._persisted(tmp_path)
        json_path = base.with_suffix(".json")
        payload = json.loads(json_path.read_text())
        payload["index_metadata"]["matrix"] = "../elsewhere.npy"
        json_path.write_text(json.dumps(payload))
        backend = NumpySidecarBackend()
        with pytest.raises(StoreCorruptError):
            backend.open(base, IDENT)

    @pytest.mark.skipif(os.geteuid() == 0, reason="root reads through permissions")
    def test_a_refused_read_is_unavailable(self, tmp_path: Path) -> None:
        base = self._persisted(tmp_path)
        json_path = base.with_suffix(".json")
        json_path.chmod(0)
        backend = NumpySidecarBackend()
        try:
            with pytest.raises(StoreUnavailableError):
                backend.open(base, IDENT)
        finally:
            json_path.chmod(0o644)


class TestLegacySidecar:
    def _legacy(self, tmp_path: Path) -> tuple[Path, EmbeddingIdentity]:
        from markdown_vault_mcp.vector_index import VectorIndex

        provider = MockEmbeddingProvider(dim=3)
        base = tmp_path / "legacy-embeddings"
        index = VectorIndex(provider)
        index.add_vectors(
            [[3.0, 4.0, 0.0], [0.0, 1.0, 0.0]],
            [{"path": "a.md", "folder": ""}, {"path": "b.md", "folder": ""}],
        )
        index.save(base)
        return base, EmbeddingIdentity.of(provider, "v1")

    def test_a_vector_index_sidecar_opens_without_tokens(self, tmp_path: Path) -> None:
        base, identity = self._legacy(tmp_path)
        store = NumpySidecarBackend().open(base, identity)
        assert (store.count, store.generation) == (2, 0)
        assert store.tokens() == {"a.md": None, "b.md": None}
        assert list(store.rows_for("a.md")[0].vector) == pytest.approx([0.6, 0.8, 0.0])

    def test_a_legacy_sidecar_of_another_model_is_incompatible(
        self, tmp_path: Path
    ) -> None:
        base, identity = self._legacy(tmp_path)
        other = EmbeddingIdentity(identity.provider, "other", "", "v1", 3)
        backend = NumpySidecarBackend()
        with pytest.raises(StoreIncompatibleError):
            backend.open(base, other)

    def test_the_first_persist_moves_to_the_new_layout(self, tmp_path: Path) -> None:
        base, identity = self._legacy(tmp_path)
        store = NumpySidecarBackend().open(base, identity)
        store.replace_document(
            doc("c.md", [[0, 0, 1]], token="tc"), expected_token=ANY_TOKEN
        )
        store.close()
        assert not base.with_suffix(".npy").exists()
        again = NumpySidecarBackend().open(base, identity)
        assert again.tokens() == {"a.md": None, "b.md": None, "c.md": "tc"}

    def test_a_metadata_list_sidecar_opens_with_a_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        base = tmp_path / "list-embeddings"
        np.save(base.with_suffix(".npy"), np.asarray([[1, 0, 0]], dtype=np.float32))
        base.with_suffix(".json").write_text(json.dumps([{"path": "a.md"}]))
        store = NumpySidecarBackend().open(base, IDENT)
        assert store.count == 1
        assert any(
            r.msg == "numpy_store_legacy_rows_unverified path=%s"
            for r in caplog.records
        )

    def test_an_empty_legacy_sidecar_opens_empty(self, tmp_path: Path) -> None:
        from markdown_vault_mcp.vector_index import VectorIndex

        provider = MockEmbeddingProvider(dim=3)
        base = tmp_path / "empty-embeddings"
        VectorIndex(provider).save(base)
        store = NumpySidecarBackend().open(base, EmbeddingIdentity.of(provider, "v1"))
        assert store.count == 0


_CHILD = textwrap.dedent(
    """
    import os, sys
    from pathlib import Path
    from markdown_vault_mcp.stores import numpy_sidecar as ns
    from markdown_vault_mcp.types import DocumentRows, EmbeddingIdentity, VectorRow

    base, point = Path(sys.argv[1]), sys.argv[2]
    real = ns._commit_file

    def die_before(path, data):
        os._exit(3)

    def die_after(path, data):
        real(path, data)
        os._exit(3)

    ns._commit_file = die_before if point == "before" else die_after
    ident = EmbeddingIdentity("fake", "m", "", "v1", 3)
    store = ns.NumpySidecarBackend().open(base, ident)
    row = VectorRow(vector=[0.0, 1.0, 0.0], metadata={"path": "new.md"})
    store.publish_generation(ident, [DocumentRows("new.md", "tn", [row])])
    """
)


class TestInterruptedPersist:
    def _seeded(self, tmp_path: Path) -> Path:
        base = _base(tmp_path)
        store = NumpySidecarBackend().open(base, IDENT)
        store.replace_document(doc("old.md", [[1, 0, 0]]), expected_token=ANY_TOKEN)
        store.close()
        return base

    def _kill_during_publish(self, base: Path, point: str) -> None:
        done = subprocess.run(
            [sys.executable, "-c", _CHILD, str(base), point], check=False
        )
        assert done.returncode == 3

    def test_a_kill_before_the_commit_keeps_the_previous_generation(
        self, tmp_path: Path
    ) -> None:
        base = self._seeded(tmp_path)
        self._kill_during_publish(base, "before")
        store = NumpySidecarBackend().open(base, IDENT)
        assert (store.generation, store.tokens()) == (0, {"old.md": "t1"})

    def test_a_kill_after_the_commit_keeps_the_new_generation(
        self, tmp_path: Path
    ) -> None:
        base = self._seeded(tmp_path)
        self._kill_during_publish(base, "after")
        store = NumpySidecarBackend().open(base, IDENT)
        assert (store.generation, store.tokens()) == (1, {"new.md": "tn"})

    def test_a_persist_after_an_orphaned_matrix_succeeds(self, tmp_path: Path) -> None:
        base = self._seeded(tmp_path)
        self._kill_during_publish(base, "before")  # leaves m2.npy uncommitted
        store = NumpySidecarBackend().open(base, IDENT)
        store.replace_document(doc("more.md", [[0, 0, 1]]), expected_token=ANY_TOKEN)
        store.close()
        again = NumpySidecarBackend().open(base, IDENT)
        assert set(again.tokens()) == {"old.md", "more.md"}

    def test_a_read_during_the_persist_sees_the_old_generation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = self._seeded(tmp_path)
        store = NumpySidecarBackend().open(base, IDENT)
        seen: list[tuple[int, int]] = []
        real = numpy_sidecar._commit_file

        def watching(path: Path, data: bytes) -> None:
            seen.append((store.generation, store.count))
            real(path, data)

        monkeypatch.setattr(numpy_sidecar, "_commit_file", watching)
        store.publish_generation(
            IDENT, [doc("n1.md", [[0, 1, 0]]), doc("n2.md", [[0, 0, 1]])]
        )
        assert seen == [(0, 1)]
        assert (store.generation, store.count) == (1, 2)


class TestPersistFailures:
    def test_a_failed_checkpoint_keeps_the_store_dirty_and_the_files(
        self, tmp_path: Path
    ) -> None:
        base = _base(tmp_path)
        store = NumpySidecarBackend().open(base, IDENT)
        store.replace_document(doc("a.md", [[1, 0, 0]]), expected_token=ANY_TOKEN)
        store.checkpoint()
        before = base.with_suffix(".json").read_bytes()
        store.replace_document(
            doc("bad.md", [[0, 1, 0]], when=object()), expected_token=ANY_TOKEN
        )
        with pytest.raises(TypeError):
            store.checkpoint()
        assert base.with_suffix(".json").read_bytes() == before
        store.remove_document("bad.md", expected_token=ANY_TOKEN)
        store.checkpoint()
        assert NumpySidecarBackend().open(base, IDENT).tokens() == {"a.md": "t1"}

    def test_close_with_a_failing_persist_still_closes(self, tmp_path: Path) -> None:
        base = _base(tmp_path)
        store = NumpySidecarBackend().open(base, IDENT)
        store.replace_document(
            doc("bad.md", [[0, 1, 0]], when=object()), expected_token=ANY_TOKEN
        )
        with pytest.raises(TypeError):
            store.close()
        with pytest.raises(RuntimeError, match="closed"):
            store.count  # noqa: B018
        assert not base.with_suffix(".json").exists()


def _mutate_commit(base: Path, change: Callable[[dict[str, Any]], None]) -> None:
    json_path = base.with_suffix(".json")
    payload = json.loads(json_path.read_text())
    change(payload)
    json_path.write_text(json.dumps(payload))


class TestOpenRejectsMalformedState:
    def _persisted(self, tmp_path: Path) -> Path:
        base = _base(tmp_path)
        store = NumpySidecarBackend().open(base, IDENT)
        store.replace_document(doc("a.md", [[1, 0, 0]]), expected_token=ANY_TOKEN)
        store.close()
        return base

    @pytest.mark.parametrize(
        "change",
        [
            lambda p: p["index_metadata"].update(generation="x"),
            lambda p: p["index_metadata"].update(generation=None),
            lambda p: p["index_metadata"].update(matrix_seq="x"),
            lambda p: p["index_metadata"].update(dimension="x"),
            lambda p: p.update(index_metadata=[1]),
            lambda p: p.update(tokens=5),
            lambda p: p.update(tokens=["ab"]),
            lambda p: p.update(tokens={"a.md": 7}),
        ],
        ids=[
            "generation-str",
            "generation-none",
            "matrix_seq-str",
            "dimension-str",
            "index_metadata-list",
            "tokens-int",
            "tokens-list",
            "token-value-int",
        ],
    )
    def test_a_malformed_commit_field_is_corrupt(
        self, tmp_path: Path, change: Callable[[dict[str, Any]], None]
    ) -> None:
        base = self._persisted(tmp_path)
        _mutate_commit(base, change)
        backend = NumpySidecarBackend()
        with pytest.raises(StoreCorruptError):
            backend.open(base, IDENT)

    def test_a_matrix_of_strings_is_corrupt(self, tmp_path: Path) -> None:
        base = self._persisted(tmp_path)
        np.save(base.with_suffix(".m1.npy"), np.asarray([["a", "b", "c"]]))
        backend = NumpySidecarBackend()
        with pytest.raises(StoreCorruptError):
            backend.open(base, IDENT)

    def test_a_matrix_wider_than_the_identity_is_corrupt(self, tmp_path: Path) -> None:
        base = self._persisted(tmp_path)
        np.save(base.with_suffix(".m1.npy"), np.zeros((1, 4), dtype=np.float32))
        backend = NumpySidecarBackend()
        with pytest.raises(StoreCorruptError, match="columns"):
            backend.open(base, IDENT)


class TestReaderRacesWriter:
    def test_a_checkpoint_between_reading_the_commit_and_the_matrix_is_survived(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        base = _base(tmp_path)
        writer = NumpySidecarBackend().open(base, IDENT)
        writer.replace_document(doc("a.md", [[1, 0, 0]]), expected_token=ANY_TOKEN)
        writer.checkpoint()  # commits m1
        real = numpy_sidecar._read_json
        calls: list[int] = []

        def stale_then_fresh(path: Path, where: str) -> Any:
            payload = real(path, where)
            if not calls:
                # The writer commits m2 and removes m1 after the reader read
                # the commit file naming m1.
                writer.replace_document(
                    doc("b.md", [[0, 1, 0]]), expected_token=ANY_TOKEN
                )
                writer.checkpoint()
            calls.append(1)
            return payload

        monkeypatch.setattr(numpy_sidecar, "_read_json", stale_then_fresh)
        reader = NumpySidecarBackend().open(base, IDENT, owns_files=False)
        assert set(reader.tokens()) == {"a.md", "b.md"}
