"""A keyword build whose walk was incomplete withholds its sentinel (#1785)."""

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
    finally:
        vault.close()


def test_an_incomplete_walk_withholds_it_and_search_still_works(tmp_path: Path) -> None:
    vault = _vault(tmp_path)
    shut = tmp_path / "vault" / "shut"
    shut.mkdir()
    (shut / "hidden.md").write_text("# Hidden\n\nbody\n")
    shut.chmod(0)
    try:
        stats = vault.index.build_index(force=True)
        assert not stats.walk_complete
        assert not vault._fts.is_build_completed()
        assert [r.path for r in vault.reader.search("body", mode="keyword")] == [
            "ok.md"
        ]
    finally:
        shut.chmod(0o755)
        vault.close()
