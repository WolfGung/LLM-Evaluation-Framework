"""A guard for the whole test session: the owner's labels file stays untouched.

`labels/human.jsonl` holds labels the owner set by hand with `make label`.
No test may create, change or delete it; tests of the labelling tool use
files in `tmp_path`. This fixture records the file before the session and
fails the session's last teardown if it differs afterwards.
"""

from pathlib import Path

import pytest

HUMAN_LABELS = Path(__file__).resolve().parents[1] / "labels" / "human.jsonl"


def _snapshot() -> bytes | None:
    return HUMAN_LABELS.read_bytes() if HUMAN_LABELS.is_file() else None


@pytest.fixture(scope="session", autouse=True)
def owner_labels_untouched():
    before = _snapshot()
    yield
    assert _snapshot() == before, (
        f"a test changed {HUMAN_LABELS}: only the owner writes it, with make label"
    )
