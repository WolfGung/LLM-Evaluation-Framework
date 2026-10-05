"""The committed results/ are a replay of the committed cassettes, byte for byte.

Every number in the README comes from results/, and results/ come from
cassettes/: this test keeps that one chain. It replays the recorded run twice,
in two processes with a different hash seed and time zone, into tmp_path,
and runs `llmeval agreement` on each replay as make eval does. The two runs
must be byte-identical (no wall-clock field, no set order, no local time),
and they must equal the committed results/. Without a manifest the replay
writes nothing, so results/ must hold no results either.

`results/judge-agreement.json` is part of the chain: `llmeval agreement`
writes it from the replay, the committed `labels/sample.json` and the
owner's committed `labels/human.jsonl` (read, never written; no file means
"pending human labels"). `results/baseline.json` is not replay output (make
baseline writes it from the results), so it is left out here.

No model is called: replay needs no key and no network.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from llmeval.cassettes import load_manifest

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
NOT_REPLAY_OUTPUT = frozenset({"baseline.json"})
FIX = "run make eval and commit results/"


def replay_into(directory: Path, *, hash_seed: str, time_zone: str) -> dict[str, bytes]:
    """Run `llmeval eval`, then `llmeval agreement`, on the repository's
    recording and labels; return the files they wrote."""
    env = {**os.environ, "PYTHONHASHSEED": hash_seed, "TZ": time_zone}
    for name in ("OPENROUTER_API_KEY", "LLMEVAL_MODE"):
        env.pop(name, None)
    common = [
        "--cassettes-dir",
        str(ROOT / "cassettes"),
        "--rubric",
        str(ROOT / "rubrics" / "judge.md"),
        "--results-dir",
        str(directory),
    ]
    commands = [
        [
            "eval",
            "--config",
            str(ROOT / "config" / "models.yaml"),
            "--datasets-dir",
            str(ROOT / "datasets"),
            *common,
        ],
        [
            "agreement",
            "--sample",
            str(ROOT / "labels" / "sample.json"),
            "--labels",
            str(ROOT / "labels" / "human.jsonl"),
            *common,
        ],
    ]
    for command in commands:
        done = subprocess.run(
            [sys.executable, "-m", "llmeval", *command],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert done.returncode == 0, done.stdout + done.stderr
    return {path.name: path.read_bytes() for path in sorted(directory.glob("*.json"))}


@pytest.fixture(scope="module")
def replays(tmp_path_factory) -> tuple[dict[str, bytes], dict[str, bytes]]:
    first = replay_into(tmp_path_factory.mktemp("replay-a"), hash_seed="1", time_zone="UTC")
    second = replay_into(
        tmp_path_factory.mktemp("replay-b"), hash_seed="2", time_zone="Asia/Tokyo"
    )
    return first, second


def committed_results() -> dict[str, bytes]:
    return {
        path.name: path.read_bytes()
        for path in sorted(RESULTS.glob("*.json"))
        if path.name not in NOT_REPLAY_OUTPUT
    }


def test_a_replay_writes_results_exactly_when_there_is_a_recorded_run(replays):
    fresh, _ = replays
    assert bool(fresh) == (load_manifest(ROOT / "cassettes") is not None)


def test_two_replays_of_the_cassettes_are_byte_identical(replays):
    first, second = replays
    assert first.keys() == second.keys()
    differ = sorted(name for name in first if first[name] != second[name])
    assert not differ, f"two replays of the same cassettes differ in: {', '.join(differ)}"


def test_the_committed_results_are_a_fresh_replay_of_the_cassettes(replays):
    fresh, _ = replays
    committed = committed_results()
    problems = []
    if missing := sorted(fresh.keys() - committed.keys()):
        problems.append(f"missing from results/: {', '.join(missing)}")
    if extra := sorted(committed.keys() - fresh.keys()):
        problems.append(f"in results/ but not written by the replay: {', '.join(extra)}")
    both = fresh.keys() & committed.keys()
    if stale := sorted(name for name in both if fresh[name] != committed[name]):
        problems.append(f"different from a fresh replay: {', '.join(stale)}")
    assert not problems, "; ".join(problems) + f": {FIX}"
