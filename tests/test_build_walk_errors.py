"""A keyword build whose walk was incomplete says so (#1785)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from markdown_vault_mcp.config_sections import VaultSettings
from markdown_vault_mcp.vault import Vault

if TYPE_CHECKING:
    from pathlib import Path


def _vault(tmp_path: Path) -> Vault:
    root = tmp_path / "vault"
    root.mkdir(exist_ok=True)
    (root / "ok.md").write_text("# Ok\n\nbody\n")
    return Vault(
        source_dir=root,
        settings=VaultSettings(
            index_path=tmp_path / "index.db",
            state_path=tmp_path / "state.json",  # outside the source: chmod-safe
            read_only=False,
        ),
    )


def test_a_complete_walk_sets_the_sentinel(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    try:
        stats = vault.index.build_index(force=True)
        assert stats.walk_complete
        assert vault._fts.is_build_completed()
        assert vault._fts.is_build_walk_complete()
    finally:
        vault.close()


def test_an_incomplete_walk_is_recorded_and_search_still_works(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    shut = tmp_path / "vault" / "shut"
    shut.mkdir()
    (shut / "hidden.md").write_text("# Hidden\n\nbody\n")
    shut.chmod(0)
    try:
        stats = vault.index.build_index(force=True)
        assert not stats.walk_complete
        # The build completed (warm restarts stay warm) but read not everything.
        assert vault._fts.is_build_completed()
        assert not vault._fts.is_build_walk_complete()
        assert [r.path for r in vault.reader.search("body", mode="keyword")] == [
            "ok.md"
        ]
    finally:
        shut.chmod(0o755)
        vault.close()


def test_an_unreadable_directory_does_not_hide_offline_deletions(
    tmp_path: Path,
) -> None:
    """A permanently unreadable dir (lost+found) must not stop warm restarts.

    Withholding the build sentinel made every boot a non-forced cold build,
    which purges nothing and rewrote the tracker without the deleted note.
    """
    root = tmp_path / "vault"
    root.mkdir()
    (root / "a.md").write_text("# A\n\napples\n")
    (root / "b.md").write_text("# B\n\nboats\n")
    lost = root / "lost+found"
    lost.mkdir()
    lost.chmod(0)
    try:
        vault = _vault(tmp_path)
        vault.index.build_index()
        vault.close()
        (root / "b.md").unlink()
        vault = _vault(tmp_path)
        try:
            vault.index.build_index()
            vault.index.reindex()
            assert "b.md" not in {r["path"] for r in vault._fts.list_notes()}
        finally:
            vault.close()
    finally:
        lost.chmod(0o755)
