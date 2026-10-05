"""The session guard on the owner's labels file, tested on temporary files only.

Synthetic data: the label lines below are made up. Every file lives in
pytest's `tmp_path`; the repository's `labels/human.jsonl` is never created,
written or deleted here. Each test first sets old timestamps, so a change
shows in the next snapshot however fast it happens.
"""

import os
import time

from tests.owner_labels import snapshot

OLD = 1_000_000_000_000_000_000  # 2001-09-09, in nanoseconds


def labels_file(tmp_path, content: bytes | None):
    directory = tmp_path / "labels"
    directory.mkdir()
    path = directory / "human.jsonl"
    if content is not None:
        path.write_bytes(content)
        os.utime(path, ns=(OLD, OLD))
    os.utime(directory, ns=(OLD, OLD))
    return path


def test_an_untouched_file_or_a_missing_one_keeps_its_snapshot(tmp_path):
    path = labels_file(tmp_path, b'{"synthetic": 1}\n')
    assert snapshot(path) == snapshot(path)
    missing = tmp_path / "other" / "human.jsonl"
    assert snapshot(missing) == snapshot(missing) == (None, None, None, None, None)


def test_an_append_then_restore_is_caught(tmp_path):
    original = b'{"synthetic": 1}\n'
    path = labels_file(tmp_path, original)
    before = snapshot(path)
    with path.open("ab") as handle:
        handle.write(b'{"synthetic": 2}\n')
    path.write_bytes(original)
    assert path.read_bytes() == original
    assert snapshot(path) != before  # the modification time moved


def test_a_create_then_delete_is_caught(tmp_path):
    path = labels_file(tmp_path, None)
    before = snapshot(path)
    path.write_bytes(b'{"synthetic": 1}\n')
    path.unlink()
    assert not path.exists()
    assert snapshot(path) != before  # the directory's modification time moved


def test_a_copy_swapped_in_with_the_old_time_is_caught(tmp_path):
    original = b'{"synthetic": 1}\n'
    path = labels_file(tmp_path, original)
    before = snapshot(path)
    copy = path.with_name("copy.jsonl")
    copy.write_bytes(original)
    os.utime(copy, ns=(OLD, OLD))
    os.replace(copy, path)
    os.utime(path.parent, ns=(OLD, OLD))
    assert snapshot(path) != before  # another inode


def test_a_restore_that_also_resets_the_modification_time_is_caught(tmp_path):
    original = b'{"synthetic": 1}\n'
    path = labels_file(tmp_path, original)
    before = snapshot(path)
    # The inode change time follows the kernel's coarse clock; let it tick.
    time.sleep(0.05)
    with path.open("ab") as handle:
        handle.write(b'{"synthetic": 2}\n')
    path.write_bytes(original)
    os.utime(path, ns=(OLD, OLD))  # the modification time put back too
    os.utime(path.parent, ns=(OLD, OLD))
    assert snapshot(path) != before  # utime cannot set the inode change time
