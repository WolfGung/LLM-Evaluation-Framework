"""A guard for the whole test session: the owner's labels file stays untouched.

`labels/human.jsonl` holds labels the owner set by hand with `make label`.
No test may create, change or delete it; tests of the labelling tool use
files in `tmp_path`. This fixture takes a snapshot (`tests.owner_labels`:
bytes, inode, modification and change times, and the directory's
modification time) before the session and fails the session's last teardown
if it differs afterwards. The snapshot itself is tested on temporary files
only.
"""

from pathlib import Path

import pytest

from tests.owner_labels import snapshot

HUMAN_LABELS = Path(__file__).resolve().parents[1] / "labels" / "human.jsonl"


@pytest.fixture(scope="session", autouse=True)
def owner_labels_untouched():
    before = snapshot(HUMAN_LABELS)
    yield
    assert snapshot(HUMAN_LABELS) == before, (
        f"a test created, changed or deleted {HUMAN_LABELS} or touched its directory: "
        "only the owner writes it, with make label"
    )
