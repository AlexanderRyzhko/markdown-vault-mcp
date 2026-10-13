"""The file-backed source probe (#1785)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from markdown_vault_mcp.hashing import compute_file_hash
from markdown_vault_mcp.interfaces import SourceProbe
from markdown_vault_mcp.source_probe import FileSourceProbe
from markdown_vault_mcp.types import SourceState

if TYPE_CHECKING:
    from pathlib import Path


def test_it_satisfies_the_protocol(tmp_path: Path) -> None:
    assert isinstance(FileSourceProbe(tmp_path), SourceProbe)


def test_a_file_is_present_with_its_content_hash(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# A\n")
    assert FileSourceProbe(tmp_path).probe("a.md") == SourceState(
        "present", compute_file_hash(tmp_path / "a.md")
    )


@pytest.mark.parametrize("make", ["missing", "under_a_file", "directory"])
def test_no_file_there_is_absent(tmp_path: Path, make: str) -> None:
    if make == "under_a_file":
        (tmp_path / "x").write_text("")
        path = "x/a.md"
    elif make == "directory":
        (tmp_path / "a.md").mkdir()
        path = "a.md"
    else:
        path = "a.md"
    assert FileSourceProbe(tmp_path).probe(path).status == "absent"


def test_a_refused_stat_is_unavailable(tmp_path: Path) -> None:
    shut = tmp_path / "shut"
    shut.mkdir()
    (shut / "a.md").write_text("# A\n")
    shut.chmod(0)
    try:
        assert FileSourceProbe(tmp_path).probe("shut/a.md").status == "unavailable"
    finally:
        shut.chmod(0o755)


@pytest.mark.parametrize("root", ["missing", "file"])
def test_a_root_that_is_not_a_directory_is_unavailable(
    tmp_path: Path, root: str
) -> None:
    base = tmp_path / "root"
    if root == "file":
        base.write_text("")
    assert FileSourceProbe(base).probe("a.md").status == "unavailable"
