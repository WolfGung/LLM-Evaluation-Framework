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


def delete_every_triage_v1_failure(data: dict) -> None:
    cases = data["functions"]["triage"]["v1"]["cases"]
    for case_id in [case_id for case_id, case in cases.items() if not case["passed"]]:
        del cases[case_id]


RAG_V1 = ("functions", "rag", "v1", "metrics")
PAIRWISE = ("pairwise", "rag", "v1-vs-v2")
# An edit of the rebuilt baseline's data; it returns the differences it expects.
Edit = Callable[[dict], list[str]]


def get(data: dict, *path: str) -> Any:
    for key in path:
        data = data[key]
    return data


def moved(data: dict, path: tuple[str, ...], by: float) -> tuple[float, str]:
    """The value at `path` moved by `by` (rounded as the baseline stores rates),
    and the message naming the edit against the rebuilt value."""
    rebuilt = get(data, *path)
    edited = round(rebuilt + by, 4)
    return edited, f"committed {edited}, rebuilt {rebuilt}"


def metric_edit(path: tuple[str, ...], by: float, label: str) -> Edit:
    def edit(data: dict) -> list[str]:
        edited, message = moved(data, path, by)
        set_value(*path, value=edited)(data)
        return [f"{label}: {message}"]

    return edit


def provenance_edit(key: str, value: str) -> Edit:
    def edit(data: dict) -> list[str]:
        rebuilt = get(data, "provenance", *key.split("."))
        set_value("provenance", *key.split("."), value=value)(data)
        return [f"provenance {key}: committed {value}, rebuilt {rebuilt}"]

    return edit


def invent_known_failure(data: dict) -> list[str]:
    case_id, case = next(
        (case_id, case)
        for case_id, case in get(data, "functions", "rag", "v1", "cases").items()
        if ["safety", "no_trap_leak"] not in case["failed_checks"]
    )
    before = case["failed_checks"]
    case["failed_checks"] = [*before, ["safety", "no_trap_leak"]]
    case["passed"] = False
    named = ", ".join("/".join(check) for check in case["failed_checks"])
    rebuilt = f"fails {', '.join('/'.join(check) for check in before)}" if before else "passes"
    return [f"rag v1 {case_id}: committed fails {named}; rebuilt {rebuilt}"]


def delete_case(data: dict) -> list[str]:
    cases = get(data, "functions", "triage", "v1", "cases")
    case_id = sorted(cases)[-1]
    del cases[case_id]
    return [f"triage v1 {case_id}: not in the committed baseline"]


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(
            metric_edit((*RAG_V1, "all_checks"), -0.05, "rag v1 metric all_checks"),
            id="a-lower-all-checks",
        ),
        pytest.param(
            metric_edit((*RAG_V1, "all_checks"), 0.05, "rag v1 metric all_checks"),
            id="b-raise-all-checks",
        ),
        pytest.param(
            metric_edit((*RAG_V1, "layers", "safety"), -0.05, "rag v1 metric layers.safety"),
            id="c-lower-safety",
        ),
        pytest.param(
            metric_edit((*PAIRWISE, "consistent"), -0.1, "rag v1 vs v2 metric consistent"),
            id="d-lower-consistency",
        ),
        pytest.param(provenance_edit("models.system", "synthetic/other:free"), id="e-change-model"),
        pytest.param(provenance_edit("recorded_from", "2026-01-01T00:00:00Z"), id="e-change-date"),
        pytest.param(invent_known_failure, id="f-invent-a-known-failure"),
        pytest.param(delete_case, id="g-delete-a-case"),
    ],
)
def test_a_hand_edit_of_the_baseline_is_named(rebuilt, edit):
    # Each edit starts from the rebuilt baseline, so these cases test the
    # message of baseline_differences alone: a drift of the committed file
    # fails only the test above. Each edit returns the message it expects,
    # with the rebuilt values read from the rebuilt baseline.
    data = rebuilt.model_dump(mode="json")
    expected = edit(data)
    differences = baseline_differences(Baseline.model_validate(data), rebuilt)
    assert differences == expected


def test_deleting_every_known_failure_of_a_version_is_named_case_by_case(rebuilt):
    data = rebuilt.model_dump(mode="json")
    failing = [
        case_id
        for case_id, case in get(data, "functions", "triage", "v1", "cases").items()
        if not case["passed"]
    ]
    assert failing, "triage v1 has no known failure to delete"
    delete_every_triage_v1_failure(data)
    differences = baseline_differences(Baseline.model_validate(data), rebuilt)
    assert differences == [
        f"triage v1 {case_id}: not in the committed baseline" for case_id in failing
    ]
