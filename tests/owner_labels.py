"""A snapshot of the owner's labels file, for the session guard in conftest.py.

The snapshot holds the file's bytes, inode, modification time and inode
change time, and the modification time of its directory. So an append that
is later undone (the file's times move), even with the modification time put
back by `utime` (the change time still moves: nothing can set it), a file
created and deleted again (the directory's time moves) and a copy swapped in
(another inode) all change it, even when the bytes end up the same.
"""

from __future__ import annotations

from pathlib import Path

Snapshot = tuple[bytes | None, int | None, int | None, int | None, int | None]


def snapshot(labels_file: Path) -> Snapshot:
    """(bytes, inode, mtime in ns, ctime in ns) of `labels_file`, None each
    when it is absent, and the mtime in ns of its directory (None when absent)."""
    directory = labels_file.parent
    folder = directory.stat().st_mtime_ns if directory.is_dir() else None
    try:
        stat = labels_file.stat()
        content = labels_file.read_bytes()
    except FileNotFoundError:
        return (None, None, None, None, folder)
    return (content, stat.st_ino, stat.st_mtime_ns, stat.st_ctime_ns, folder)
