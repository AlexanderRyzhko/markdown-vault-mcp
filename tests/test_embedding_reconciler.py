"""The one writer of embedding rows (#1768)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from markdown_vault_mcp.embed_text import EmbedTextBuilder
from markdown_vault_mcp.fts_index import FTSIndex
from markdown_vault_mcp.indexing.embedding_reconciler import EmbeddingReconciler
from markdown_vault_mcp.scanner import HeadingChunker, parse_note
from markdown_vault_mcp.stores.numpy_sidecar import NumpySidecarBackend
from markdown_vault_mcp.types import (
    ANY_TOKEN,
    DocumentRows,
    EmbeddingIdentity,
)
from tests.conftest import MockEmbeddingProvider
from tests.fixtures.vectors import metadata_by_path

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class _Counting(MockEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__(dim=8)
        self.texts: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return super().embed(texts)


def _index(tmp_path: Path, notes: dict[str, str]) -> FTSIndex:
    fts = FTSIndex(db_path=":memory:")
    for name, text in notes.items():
        (tmp_path / name).write_text(text)
        fts.upsert_note(parse_note(tmp_path / name, tmp_path, HeadingChunker()))
    return fts


def _setup(tmp_path: Path, notes: dict[str, str], removable: list[str] | None = None):
    fts = _index(tmp_path, notes)
    provider = _Counting()
    store = NumpySidecarBackend().open(None, EmbeddingIdentity.of(provider, "v1"))
    reconciler = EmbeddingReconciler(
        fts=fts,
        provider=provider,
        embed_builder=EmbedTextBuilder(),
        batch_size=4,
        confirm_removal=lambda paths: [p for p in paths if p in (removable or [])],
    )
    return fts, provider, store, reconciler


NOTES = {"a.md": "# A\n\nApples.\n", "b.md": "# B\n\nBoats.\n"}


class TestDecisions:
    def test_new_documents_are_embedded_under_their_token(self, tmp_path):
        fts, _provider, store, rec = _setup(tmp_path, NOTES)
        result = rec.reconcile(store, site="test")
        assert result.embedded == 2
        hashes = {r["path"]: r["content_hash"] for r in fts.list_chunks()}
        assert store.tokens() == hashes

    def test_a_converged_store_costs_nothing(self, tmp_path):
        _, provider, store, rec = _setup(tmp_path, NOTES)
        rec.reconcile(store, site="test")
        provider.texts.clear()
        result = rec.reconcile(store, site="test")
        assert (result.embedded, result.up_to_date, provider.texts) == (0, 2, [])

    def test_a_changed_token_with_the_same_input_is_retokened_without_embedding(
        self, tmp_path
    ):
        fts, provider, store, rec = _setup(tmp_path, NOTES)
        rec.reconcile(store, site="test")
        (tmp_path / "a.md").write_text(
            "---\nmodified: 2026-10-12\n---\n# A\n\nApples.\n"
        )
        fts.upsert_note(parse_note(tmp_path / "a.md", tmp_path, HeadingChunker()))
        provider.texts.clear()
        result = rec.reconcile(store, ["a.md"], site="test")
        assert (result.retokened, provider.texts) == (1, [])
        assert (
            store.tokens()["a.md"] == fts.list_chunks(paths=["a.md"])[0]["content_hash"]
        )

    def test_a_changed_input_is_reembedded(self, tmp_path):
        fts, provider, store, rec = _setup(tmp_path, NOTES)
        rec.reconcile(store, site="test")
        (tmp_path / "a.md").write_text("# A\n\nQuinces.\n")
        fts.upsert_note(parse_note(tmp_path / "a.md", tmp_path, HeadingChunker()))
        provider.texts.clear()
        result = rec.reconcile(store, ["a.md"], site="test")
        assert result.embedded == 1
        assert any("Quinces" in t for t in provider.texts)

    def test_legacy_rows_with_the_same_input_are_adopted_without_embedding(
        self, tmp_path
    ):
        _fts, provider, store, rec = _setup(tmp_path, NOTES)
        rec.reconcile(store, site="test")
        legacy = [
            DocumentRows(p, None, store.rows_for(p)) for p in sorted(store.tokens())
        ]
        store.publish_generation(store.identity, legacy)
        provider.texts.clear()
        result = rec.reconcile(store, site="test")
        assert (result.retokened, result.embedded, provider.texts) == (2, 0, [])
        assert None not in store.tokens().values()

    def test_a_body_less_document_loses_its_rows(self, tmp_path):
        fts, _, store, rec = _setup(tmp_path, NOTES)
        rec.reconcile(store, site="test")
        (tmp_path / "a.md").write_text("---\ntitle: A\n---\n")
        fts.upsert_note(parse_note(tmp_path / "a.md", tmp_path, HeadingChunker()))
        result = rec.reconcile(store, ["a.md"], site="test")
        assert result.removed == 1
        assert "a.md" not in metadata_by_path(store)

    def test_a_path_without_rows_is_removed_only_when_confirmed(self, tmp_path):
        fts, _, store, rec = _setup(tmp_path, NOTES, removable=["a.md"])
        rec.reconcile(store, site="test")
        fts.delete_by_path("a.md")
        fts.delete_by_path("b.md")
        rec.reconcile(store, site="test")
        assert sorted(metadata_by_path(store)) == ["b.md"]


class TestFailures:
    def test_a_provider_failure_keeps_the_old_rows_and_the_pass_continues(
        self, tmp_path
    ):
        fts, provider, store, rec = _setup(tmp_path, NOTES)
        rec.reconcile(store, site="test")
        before = metadata_by_path(store)["a.md"]
        for name, body in (("a.md", "# A\n\nQuinces.\n"), ("b.md", "# B\n\nBarges.\n")):
            (tmp_path / name).write_text(body)
            fts.upsert_note(parse_note(tmp_path / name, tmp_path, HeadingChunker()))

        def flaky(texts: list[str]) -> list[list[float]]:
            if any("Quinces" in t for t in texts):
                raise RuntimeError("provider down")
            return MockEmbeddingProvider.embed(provider, texts)

        provider.embed = flaky  # type: ignore[method-assign]
        result = rec.reconcile(store, site="test")
        assert metadata_by_path(store)["a.md"] == before
        assert result.kept == 1
        assert any("Barges" in m["content"] for m in metadata_by_path(store)["b.md"])

    def test_a_refused_vector_keeps_the_old_rows(self, tmp_path):
        fts, provider, store, rec = _setup(tmp_path, NOTES)
        rec.reconcile(store, site="test")
        before = metadata_by_path(store)["a.md"]
        (tmp_path / "a.md").write_text("# A\n\nQuinces.\n")
        fts.upsert_note(parse_note(tmp_path / "a.md", tmp_path, HeadingChunker()))
        provider.embed = lambda texts: [[1.0] * 9 for _ in texts]  # type: ignore[method-assign]
        result = rec.reconcile(store, ["a.md"], site="test")
        assert (result.kept, metadata_by_path(store)["a.md"]) == (1, before)


class TestConflicts:
    def test_a_conflict_is_reported_and_the_other_documents_land(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        _fts, _, store, rec = _setup(tmp_path, NOTES)
        real = store.replace_document
        seen: set[str] = set()

        def once(document: DocumentRows, *, expected_token: Any) -> str:
            if document.path == "a.md" and "a.md" not in seen:
                seen.add("a.md")
                return "conflict"
            return real(document, expected_token=expected_token)

        monkeypatch.setattr(store, "replace_document", once)
        result = rec.reconcile(store, site="test")
        assert result.conflicts == {"a.md"}
        assert sorted(metadata_by_path(store)) == ["b.md"]
        assert rec.reconcile(store, ["a.md"], site="test").conflicts == set()
        assert sorted(metadata_by_path(store)) == ["a.md", "b.md"]

    def test_writes_pass_the_token_they_observed(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        fts, _, store, rec = _setup(tmp_path, NOTES)
        rec.reconcile(store, site="test")
        tokens_before = store.tokens()
        (tmp_path / "a.md").write_text("# A\n\nQuinces.\n")
        fts.upsert_note(parse_note(tmp_path / "a.md", tmp_path, HeadingChunker()))
        expected: list[Any] = []
        real = store.replace_document
        monkeypatch.setattr(
            store,
            "replace_document",
            lambda d, *, expected_token: (
                expected.append(expected_token)
                or real(d, expected_token=expected_token)
            ),
        )
        rec.reconcile(store, ["a.md"], site="test")
        assert expected == [tokens_before["a.md"]]
        assert ANY_TOKEN not in expected


class TestBulkRetoken:
    def test_many_adoptions_publish_one_generation(
        self, tmp_path, monkeypatch: pytest.MonkeyPatch
    ):
        import markdown_vault_mcp.indexing.embedding_reconciler as module

        monkeypatch.setattr(module, "_BULK_RETOKEN", 2)
        _fts, _provider, store, rec = _setup(tmp_path, NOTES)
        rec.reconcile(store, site="test")
        store.publish_generation(
            store.identity,
            [DocumentRows(p, None, store.rows_for(p)) for p in sorted(store.tokens())],
        )
        generation = store.generation
        calls: list[str] = []
        real = store.replace_document
        monkeypatch.setattr(
            store,
            "replace_document",
            lambda d, *, expected_token: (
                calls.append(d.path) or real(d, expected_token=expected_token)
            ),
        )
        rec.reconcile(store, site="test")
        assert (calls, store.generation) == ([], generation + 1)
        assert None not in store.tokens().values()


class TestRebuild:
    def test_it_publishes_one_generation_with_tokens(self, tmp_path):
        fts, _, store, rec = _setup(tmp_path, NOTES)
        assert rec.rebuild(store, store.identity) == (2, 2)
        hashes = {r["path"]: r["content_hash"] for r in fts.list_chunks()}
        assert (store.generation, store.tokens()) == (1, hashes)

    def test_a_failed_batch_is_left_for_convergence(self, tmp_path):
        notes = {f"n{i}.md": f"# N{i}\n\nBody {i}.\n" for i in range(6)}
        _fts, provider, store, rec = _setup(tmp_path, notes)
        real = provider.embed
        calls = 0

        def second_batch_fails(texts: list[str]) -> list[list[float]]:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("provider down")
            return real(texts)

        provider.embed = second_batch_fails  # type: ignore[method-assign]
        embedded, attempted = rec.rebuild(store, store.identity)
        assert (embedded, attempted) == (4, 6)
        provider.embed = real  # type: ignore[method-assign]
        provider.texts.clear()
        result = rec.reconcile(store, site="test")
        assert result.embedded == 2
        assert len(provider.texts) == 2
