"""The committed baseline is the baseline of the committed results.

`make baseline` writes `results/baseline.json` from `results/` and the
manifest, and nothing else may write it. This test rebuilds the baseline the
same way and compares. A hand edit fails it and the message names what
differs: a lowered or raised rate, a changed model or date, an invented or a
deleted known failure. test_results_fresh.py ties `results/` to the
cassettes, so the chain is cassettes, then results, then baseline. A change
to the baseline still shows as its own diff in a commit: that diff is the
deliberate review point, and only make baseline produces it.

Without a manifest there is no recorded run and nothing to compare. No model
is called and nothing is written.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from llmeval.baseline import (
    Baseline,
    baseline_differences,
    build_baseline,
    load_baseline,
    load_run_results,
)
from llmeval.cassettes import PENDING_RECORDED_RUN, load_manifest

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
BASELINE = RESULTS / "baseline.json"


@pytest.fixture(scope="module")
def rebuilt() -> Baseline:
    manifest = load_manifest(ROOT / "cassettes")
    if manifest is None:
        pytest.skip(PENDING_RECORDED_RUN)
    run = load_run_results(RESULTS, manifest.prompt_versions)
    return build_baseline(run.functions, manifest, run.pairwise)


@pytest.fixture(scope="module")
def committed(rebuilt) -> Baseline:
    baseline = load_baseline(BASELINE)
    assert baseline is not None, "results/baseline.json is missing: run make baseline"
    return baseline


def explain(differences: list[str]) -> str:
    lines = "\n  ".join(differences)
    return (
        f"results/baseline.json is not the baseline of results/ and the manifest:\n  {lines}\n"
        "review make eval, then make baseline"
    )


def test_the_committed_baseline_is_the_baseline_of_the_committed_results(committed, rebuilt):
    differences = baseline_differences(committed, rebuilt)
    assert not differences, explain(differences)
    # Byte for byte too: the file is exactly what make baseline writes.
    assert BASELINE.read_bytes() == (rebuilt.model_dump_json(indent=2) + "\n").encode(), (
        "results/baseline.json is not written as make baseline writes it: run make baseline"
    )


def set_value(*path: str, value: Any) -> Callable[[dict], None]:
    def edit(data: dict) -> None:
        node = data
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    return edit


def add_known_failure(data: dict) -> None:
    case = data["functions"]["rag"]["v1"]["cases"]["rag-021"]
    case["failed_checks"].append(["safety", "no_trap_leak"])


def delete_case(data: dict) -> None:
    del data["functions"]["triage"]["v1"]["cases"]["tri-036"]


def delete_every_triage_v1_failure(data: dict) -> None:
    cases = data["functions"]["triage"]["v1"]["cases"]
    for case_id in [case_id for case_id, case in cases.items() if not case["passed"]]:
        del cases[case_id]


RAG_V1 = ("functions", "rag", "v1", "metrics")


@pytest.mark.parametrize(
    ("edit", "expected"),
    [
        pytest.param(
            set_value(*RAG_V1, "all_checks", value=0.70),
            ["rag v1 metric all_checks: committed 0.7, rebuilt 0.7756"],
            id="a-lower-all-checks",
        ),
        pytest.param(
            set_value(*RAG_V1, "all_checks", value=0.80),
            ["rag v1 metric all_checks: committed 0.8, rebuilt 0.7756"],
            id="b-raise-all-checks",
        ),
        pytest.param(
            set_value(*RAG_V1, "layers", "safety", value=0.90),
            ["rag v1 metric layers.safety: committed 0.9, rebuilt 0.9551"],
            id="c-lower-safety",
        ),
        pytest.param(
            set_value("pairwise", "rag", "v1-vs-v2", "consistent", value=0.40),
            ["rag v1 vs v2 metric consistent: committed 0.4, rebuilt 0.5263"],
            id="d-lower-consistency",
        ),
        pytest.param(
            set_value("provenance", "models", "system", value="synthetic/other:free"),
            [
                "provenance models.system: committed synthetic/other:free, "
                "rebuilt qwen/qwen3.8-27b:free"
            ],
            id="e-change-model",
        ),
        pytest.param(
            set_value("provenance", "recorded_from", value="2026-10-04T07:53:29.210228Z"),
            [
                "provenance recorded_from: committed 2026-10-04T07:53:29.210228Z, "
                "rebuilt 2026-10-05T07:53:29.210228Z"
            ],
            id="e-change-date",
        ),
        pytest.param(
            add_known_failure,
            [
                "rag v1 rag-021: committed fails reference/required_facts, safety/no_trap_leak; "
                "rebuilt fails reference/required_facts"
            ],
            id="f-invent-a-known-failure",
        ),
        pytest.param(
            delete_case,
            ["triage v1 tri-036: not in the committed baseline"],
            id="g-delete-a-case",
        ),
    ],
)
def test_a_hand_edit_of_the_baseline_is_named(rebuilt, edit, expected):
    # Each edit starts from the rebuilt baseline, so these cases test the
    # message of baseline_differences alone: a drift of the committed file
    # fails only the test above.
    data = rebuilt.model_dump(mode="json")
    edit(data)
    differences = baseline_differences(Baseline.model_validate(data), rebuilt)
    assert differences == expected


def test_deleting_every_known_failure_of_a_version_is_named_case_by_case(rebuilt):
    data = rebuilt.model_dump(mode="json")
    delete_every_triage_v1_failure(data)
    differences = baseline_differences(Baseline.model_validate(data), rebuilt)
    assert len(differences) == 15
    assert all(d.endswith(": not in the committed baseline") for d in differences)
    assert "triage v1 tri-036: not in the committed baseline" in differences
