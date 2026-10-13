"""The file-backed answer to "is this note still there?" (#1785).

A ``stat`` classified by errno, as the spec's ``SourceProbe`` says: a path
that does not exist, or names something that is not a regular file, is
*absent*; a stat the file system refuses is *unavailable*, as is any path
under a source root that is not a readable directory (an unmounted volume
must remove nothing). *Present* carries the file's content hash, the
keyword index's token, so it reads the file; the deletion rule only asks
about paths the keyword index holds no rows for.
"""

from __future__ import annotations

import logging
import stat as stat_module
from typing import TYPE_CHECKING

from markdown_vault_mcp.hashing import compute_file_hash
from markdown_vault_mcp.types import SourceState

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class FileSourceProbe:
    """Answers :class:`~markdown_vault_mcp.interfaces.SourceProbe` from a directory.

    Args:
        source_dir: The vault's root directory.
    """

    def __init__(self, source_dir: Path) -> None:
        self._source_dir = source_dir

    def probe(self, path: str) -> SourceState:
        """Return whether *path* is present (with its content hash), absent or unavailable.

        Args:
            path: Vault-relative path.

        Returns:
            The source's answer.
        """
        if not self._root_available():
            return SourceState("unavailable")
        target = self._source_dir / path
        try:
            st = target.stat()
        except (FileNotFoundError, NotADirectoryError):
            return SourceState("absent")
        except OSError as exc:
            logger.debug("source_probe_stat_refused path=%s err=%s", path, exc)
            return SourceState("unavailable")
        if not stat_module.S_ISREG(st.st_mode):
            return SourceState("absent")
        try:
            return SourceState("present", compute_file_hash(target))
        except OSError as exc:
            logger.debug("source_probe_read_refused path=%s err=%s", path, exc)
            return SourceState("unavailable")

    def _root_available(self) -> bool:
        try:
            return stat_module.S_ISDIR(self._source_dir.stat().st_mode)
        except OSError:
            return False
