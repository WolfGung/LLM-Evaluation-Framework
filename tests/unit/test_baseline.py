"""The baseline: what the recorded run measured, and how a replay compares with it.

Synthetic data: every case record, result and manifest below (and in
`tests/unit/synthetic_results.py`) is made up for the test. No baseline or
result is written into the repository.
"""

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from llmeval.baseline import (
    PENDING_BASELINE,
    UPDATE_HINT,
    Baseline,
    BaselineError,
    CaseBaseline,
    CurrentInputs,
    Metrics,
    PairwiseMetrics,
    baseline_differences,
    build_baseline,
    case_baseline,
    compare,
    explain,
    load_baseline,
    load_run_results,
    stale_reasons,
    write_baseline,
)
from llmeval.cassettes import MANIFEST_FILE, RunManifest
from llmeval.checks.judge import RUBRIC_PATH
from llmeval.cli import app
from llmeval.results import (
    CallRecord,
    CaseRecord,
    CheckRecord,
    FunctionResults,
    RunRecord,
    summarise,
    write_results,
)
from llmeval.runner import prompt_sha256
from tests.unit import synthetic_results as syn

TIME = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
CALL = CallRecord(
    key="k" * 64,
    model_used="synthetic/system:free",
    latency_ms=1.0,
    prompt_tokens=1,
    completion_tokens=1,
    reasoning_tokens=0,
    cost_usd=0.0,
    cost_source="provider",
    finish_reason="stop",
    empty_reason=None,
    recorded_at=TIME,
)


def record(*runs_failing: tuple[str, ...], case_id="rag-001", category="answerable"):
    """A case record with one run per argument; each names the checks that fail."""
    names = ("retrieval_recall", "cites_retrieved", "required_facts")
    layers = {"retrieval_recall": "retrieval", "cites_retrieved": "deterministic"}
    runs = []
    for repeat, failing in enumerate(runs_failing):
        checks = [
            CheckRecord(
                layer=layers.get(name, "reference"),
                name=name,
                passed=name not in failing,
                detail=f"synthetic detail for {name}",
            )
            for name in names
        ]
        runs.append(RunRecord(repeat=repeat, output="synthetic", call=CALL, checks=checks))
    return CaseRecord(
        id=case_id, category=category, input="Synthetic question?", expected={}, runs=runs
    )


PASSING = record(())
FAILING = record(("required_facts",))


def test_a_case_baseline_lists_failed_checks_across_repeats():
    flaky = record((), ("required_facts",), ("cites_retrieved", "required_facts"))
    baseline = case_baseline(flaky)
    assert not baseline.passed
    assert baseline.failed_checks == (
        ("reference", "required_facts"),
        ("deterministic", "cites_retrieved"),
    )
    assert case_baseline(PASSING) == CaseBaseline(passed=True)


def test_passed_and_failed_checks_must_agree():
    with pytest.raises(ValidationError):
        CaseBaseline(passed=True, failed_checks=(("reference", "required_facts"),))
    with pytest.raises(ValidationError):
        CaseBaseline(passed=False)


def test_no_baseline_entry_is_pending():
    verdict = compare(PASSING, None)
    assert (verdict.outcome, verdict.message) == ("pending", PENDING_BASELINE)
    assert PENDING_BASELINE == "pending baseline"


def test_a_passing_case_that_still_passes():
    assert compare(PASSING, CaseBaseline(passed=True)).outcome == "pass"


def test_a_passing_case_that_now_fails_is_a_regression():
    verdict = compare(FAILING, CaseBaseline(passed=True))
    assert verdict.outcome == "fail"
    assert verdict.message == explain(FAILING)
    assert "[reference] required_facts: synthetic detail for required_facts" in verdict.message


def test_a_known_failure_is_an_xfail_naming_the_checks():
    expected = case_baseline(FAILING)
    verdict = compare(FAILING, expected)
    assert verdict.outcome == "xfail"
    assert "reference/required_facts" in verdict.message


def test_a_known_failure_that_now_passes_asks_for_a_deliberate_update():
    verdict = compare(PASSING, case_baseline(FAILING))
    assert verdict.outcome == "xpass"
    assert verdict.message == UPDATE_HINT
    assert UPDATE_HINT == "now passes; update the baseline deliberately (make baseline)"


def test_a_known_failure_with_a_new_failing_check_is_a_regression():
    worse = record(("required_facts", "cites_retrieved"))
    verdict = compare(worse, case_baseline(FAILING))
    assert verdict.outcome == "fail"
    assert "deterministic/cites_retrieved" in verdict.message
    assert "baseline did not" in verdict.message


def test_explain_says_when_retrieval_missed():
    text = explain(record(("retrieval_recall", "required_facts")))
    assert "retrieval miss" in text


MANIFEST = RunManifest(
    models={"system": "synthetic/system:free", "judge": "synthetic/judge:free"},
    prompt_versions={"rag": ("v1",)},
    repeats=1,
    datasets={"rag.jsonl": "0" * 64},
    recorded_from=TIME,
    recorded_to=TIME,
    planned_calls=2,
    recorded_calls=2,
    judge_repeats="all",
)


def results_of(*records):
    return FunctionResults(
        function="rag",
        version="v1",
        mode="replay",
        model="synthetic/system:free",
        prompt_sha256="p" * 64,
        dataset="rag.jsonl",
        dataset_sha256="0" * 64,
        repeats=1,
        summary=summarise("rag", list(records)),
        cases=list(records),
    )


def test_build_baseline_from_replay_results():
    baseline = build_baseline(
        [results_of(PASSING, record(("required_facts",), case_id="rag-002"))], MANIFEST
    )
    assert baseline.case("rag", "v1", "rag-001") == CaseBaseline(passed=True)
    assert baseline.case("rag", "v1", "rag-002").failed_checks == (("reference", "required_facts"),)
    assert baseline.case("rag", "v2", "rag-001") is None
    assert baseline.case("rag", "v1", "rag-999") is None
    metrics = baseline.functions["rag"]["v1"].metrics
    assert metrics.all_checks == 0.5
    assert metrics.layers["reference"] == 0.5
    assert metrics.stable_share is None
    assert baseline.provenance.models == MANIFEST.models
    assert baseline.provenance.recorded_to == TIME
    assert baseline.provenance.judge_repeats == "all"


def test_build_baseline_refuses_live_results():
    live = results_of(PASSING).model_copy(update={"mode": "live"})
    with pytest.raises(BaselineError, match="replay"):
        build_baseline([live], MANIFEST)


def test_load_baseline(tmp_path):
    assert load_baseline(tmp_path / "baseline.json") is None
    baseline = build_baseline([results_of(PASSING)], MANIFEST)
    path = tmp_path / "baseline.json"
    path.write_text(baseline.model_dump_json(indent=2), encoding="utf-8")
    assert load_baseline(path) == baseline
    path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
    with pytest.raises(BaselineError, match="baseline.json"):
        load_baseline(path)


def test_the_baseline_round_trips_through_json():
    baseline = build_baseline(
        [results_of(PASSING, FAILING.model_copy(update={"id": "rag-002"}))], MANIFEST
    )
    assert Baseline.model_validate_json(baseline.model_dump_json()) == baseline


# --- metrics the gate needs, the results files, and stale results ------------
# Built with tests/unit/synthetic_results.py (synthetic data as well).


def graded_rag(version="v1", **options):
    cases = [
        syn.case_record("rag-001", judge="pass"),
        syn.case_record("rag-002", ("reference/required_facts",), judge="fail"),
        syn.case_record("rag-003", judge="invalid"),
        syn.case_record("rag-004", ("safety/no_trap_leak",), category="safety"),
    ]
    return syn.function_results("rag", version, cases, **options)


def test_metrics_keep_the_judge_rule_pass_and_validity():
    metrics = Metrics.of(graded_rag().summary)
    # Three graded runs: two valid verdicts, one of which passes the rule.
    assert metrics.judge_valid == round(2 / 3, 4)
    assert metrics.judge_rule_pass == 0.5
    assert metrics.layers["safety"] == 0.75
    assert metrics.all_checks == 0.25
    triage = syn.function_results(
        "triage", "v1", [syn.case_record("tri-001", checks=syn.TRIAGE_CHECKS)]
    )
    ungraded = Metrics.of(triage.summary)
    assert ungraded.judge_valid is None and ungraded.judge_rule_pass is None
    assert ungraded.accuracy == {"category": 1.0, "priority": 1.0, "order_id": 1.0}


def test_pairwise_metrics_are_position_consistency_and_validity():
    cases = [
        syn.pair_case("rag-001", "A", "B"),  # v1 both times: consistent
        syn.pair_case("rag-002", "A", "A"),  # the position decided: inconsistent
        syn.pair_case("rag-003", "tie", "tie"),  # consistent
        syn.pair_case("rag-004", "A", None),  # invalid: not compared
    ]
    metrics = PairwiseMetrics.of(syn.pairwise_results(cases).summary)
    assert metrics.consistent == round(2 / 3, 4)
    assert metrics.valid == 0.75
    assert PairwiseMetrics.of(syn.pairwise_results([]).summary) == PairwiseMetrics(
        consistent=None, valid=None
    )


def test_build_baseline_keeps_the_pairwise_metrics_and_the_judge_metrics():
    pairwise = syn.pairwise_results([syn.pair_case("rag-001", "A", "B")])
    baseline = build_baseline([graded_rag()], syn.manifest(), pairwise=[pairwise])
    assert baseline.pairwise["rag"]["v1-vs-v2"] == PairwiseMetrics(consistent=1.0, valid=1.0)
    assert baseline.functions["rag"]["v1"].metrics.judge_rule_pass == 0.5
    assert Baseline.model_validate_json(baseline.model_dump_json()) == baseline


def test_build_baseline_refuses_live_pairwise_results():
    live = syn.pairwise_results([syn.pair_case("rag-001", "A", "B")], mode="live")
    with pytest.raises(BaselineError, match="replay"):
        build_baseline([graded_rag()], syn.manifest(), pairwise=[live])


def test_write_baseline_is_deterministic_json(tmp_path):
    baseline = build_baseline([graded_rag()], syn.manifest())
    path = write_baseline(baseline, tmp_path / "out" / "baseline.json")
    first = path.read_bytes()
    write_baseline(build_baseline([graded_rag()], syn.manifest()), path)
    assert path.read_bytes() == first
    assert first.endswith(b"}\n")
    assert load_baseline(path) == baseline


def written_run(tmp_path, *, rag_versions=("v1", "v2")):
    """Results files for rag (graded, with the pairwise comparison) and triage v1."""
    results = tmp_path / "results"
    for version in rag_versions:
        write_results(graded_rag(version), results)
    triage = [syn.case_record("tri-001", checks=syn.TRIAGE_CHECKS)]
    write_results(syn.function_results("triage", "v1", triage), results)
    if len(rag_versions) > 1:
        cases = [syn.pair_case("rag-001", "A", "B", rag_versions[:2])]
        write_results(syn.pairwise_results(cases, versions=rag_versions[:2]), results)
    return results


def test_load_run_results_reads_every_version_and_the_pairwise_comparison(tmp_path):
    results = written_run(tmp_path)
    run = load_run_results(results, {"rag": ("v1", "v2"), "triage": ("v1",)})
    assert [(r.function, r.version) for r in run.functions] == [
        ("rag", "v1"),
        ("rag", "v2"),
        ("triage", "v1"),
    ]
    assert [p.versions for p in run.pairwise] == [("v1", "v2")]


def test_load_run_results_names_every_missing_file(tmp_path):
    results = written_run(tmp_path, rag_versions=("v1",))
    with pytest.raises(BaselineError) as caught:
        load_run_results(results, {"rag": ("v1", "v2"), "triage": ("v1", "v2")})
    message = str(caught.value)
    assert "rag-v2.json" in message and "triage-v2.json" in message
    assert "run make eval" in message
    # The pairwise file is expected once the graded RAG results are there.
    (results / "rag-v2.json").write_text(graded_rag("v2").model_dump_json(), encoding="utf-8")
    with pytest.raises(BaselineError, match="rag-v1-vs-v2.json"):
        load_run_results(results, {"rag": ("v1", "v2"), "triage": ("v1",)})


def test_load_run_results_refuses_a_broken_file(tmp_path):
    results = written_run(tmp_path)
    (results / "triage-v1.json").write_text('{"function": "triage"}', encoding="utf-8")
    with pytest.raises(BaselineError, match="triage-v1.json: not valid results"):
        load_run_results(results, {"rag": ("v1", "v2"), "triage": ("v1",)})


def current_inputs(**changes):
    inputs = {
        "prompts": {
            (function, version): prompt_sha256(function, version)
            for function, version in (("rag", "v1"), ("rag", "v2"), ("triage", "v1"))
        },
        "datasets": {"rag.jsonl": "0" * 64, "triage.jsonl": "0" * 64},
        "rubric": syn.RUBRIC_SHA256,
    }
    return CurrentInputs(**{**inputs, **changes})


def fresh_run(tmp_path):
    return load_run_results(written_run(tmp_path), {"rag": ("v1", "v2"), "triage": ("v1",)})


def test_fresh_results_have_no_stale_reasons(tmp_path):
    assert stale_reasons(fresh_run(tmp_path), syn.manifest(), current_inputs()) == []


@pytest.mark.parametrize(
    ("manifest_changes", "expected"),
    [
        (
            {"models": {"system": "synthetic/other:free", "judge": syn.JUDGE_MODEL}},
            "rag v1: system model synthetic/system:free, the recording used synthetic/other:free",
        ),
        (
            {"models": {"system": syn.SYSTEM_MODEL, "judge": "synthetic/other:free"}},
            "rag v1 vs v2: judge model synthetic/judge:free, "
            "the recording used synthetic/other:free",
        ),
        ({"repeats": 3}, "triage v1: repeats 1, the recording has 3"),
        ({"judge_repeats": "all"}, "rag v2: judge_repeats first, the recording has all"),
        ({"rubric_sha256": "f" * 64}, "rag v1: graded with another rubric than the recording"),
    ],
)
def test_results_that_do_not_match_the_recording_are_stale(tmp_path, manifest_changes, expected):
    recording = syn.manifest().model_copy(update=manifest_changes)
    assert expected in stale_reasons(fresh_run(tmp_path), recording, current_inputs())


@pytest.mark.parametrize(
    ("input_changes", "expected"),
    [
        (
            {"prompts": {("rag", "v1"): "f" * 64}},
            "rag v1: the prompt changed since these results were written",
        ),
        (
            {"datasets": {"rag.jsonl": "f" * 64, "triage.jsonl": "0" * 64}},
            "rag v1 vs v2: rag.jsonl changed since these results were written",
        ),
        ({"rubric": "f" * 64}, "rag v2: the rubric changed since these results were written"),
    ],
)
def test_results_older_than_the_current_files_are_stale(tmp_path, input_changes, expected):
    assert expected in stale_reasons(
        fresh_run(tmp_path), syn.manifest(), current_inputs(**input_changes)
    )


def test_live_results_are_stale_for_a_baseline(tmp_path):
    run = fresh_run(tmp_path)
    live = run.functions[0].model_copy(update={"mode": "live"})
    run = type(run)(functions=(live, *run.functions[1:]), pairwise=run.pairwise)
    assert (
        "rag v1: comes from a live run; a baseline is built from replay results only"
        in stale_reasons(run, syn.manifest(), current_inputs())
    )


# --- llmeval baseline ------------------------------------------------------------

cli_runner = CliRunner()


def baseline_workspace(tmp_path, *, rag_versions=("v1", "v2")):
    """A recorded run in tmp_path: manifest, datasets and replay results that match."""
    datasets = syn.write_datasets(tmp_path / "datasets")
    versions = {"rag": rag_versions, "triage": ("v1",)}
    (tmp_path / "cassettes").mkdir()
    (tmp_path / "cassettes" / MANIFEST_FILE).write_text(
        syn.manifest(versions).model_dump_json(), encoding="utf-8"
    )
    results = tmp_path / "results"
    for version in rag_versions:
        write_results(graded_rag(version, dataset_sha256=datasets["rag"]), results)
    triage = [syn.case_record("tri-001", checks=syn.TRIAGE_CHECKS)]
    write_results(
        syn.function_results("triage", "v1", triage, dataset_sha256=datasets["triage"]), results
    )
    pair = [syn.pair_case("rag-001", "A", "B")]
    write_results(syn.pairwise_results(pair, dataset_sha256=datasets["rag"]), results)
    return tmp_path


def baseline_args(ws, *extra):
    return [
        "baseline",
        "--results-dir",
        str(ws / "results"),
        "--baseline",
        str(ws / "results" / "baseline.json"),
        "--cassettes-dir",
        str(ws / "cassettes"),
        "--datasets-dir",
        str(ws / "datasets"),
        "--rubric",
        str(syn.ROOT / RUBRIC_PATH),
        *extra,
    ]


def test_llmeval_baseline_writes_the_baseline_of_the_results(tmp_path):
    ws = baseline_workspace(tmp_path)
    result = cli_runner.invoke(app, baseline_args(ws))
    assert result.exit_code == 0, result.output
    path = ws / "results" / "baseline.json"
    run = load_run_results(ws / "results", {"rag": ("v1", "v2"), "triage": ("v1",)})
    expected = build_baseline(run.functions, syn.manifest(), run.pairwise)
    assert load_baseline(path) == expected
    assert path.read_bytes() == expected.model_dump_json(indent=2).encode() + b"\n"
    assert "rag v1: 4 cases: 1 pass, 3 known failures" in result.output
    assert "triage v1: 1 case: 1 pass, 0 known failures" in result.output
    assert "rag v1 vs v2: position consistency 100.0%" in result.output
    assert f"wrote {path}" in result.output


def test_llmeval_baseline_without_a_recorded_run_is_pending(tmp_path):
    ws = baseline_workspace(tmp_path)
    (ws / "cassettes" / MANIFEST_FILE).unlink()
    result = cli_runner.invoke(app, baseline_args(ws))
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "pending first recorded run"
    assert not (ws / "results" / "baseline.json").exists()


def test_llmeval_baseline_refuses_missing_results(tmp_path):
    ws = baseline_workspace(tmp_path)
    (ws / "results" / "triage-v1.json").unlink()
    result = cli_runner.invoke(app, baseline_args(ws))
    assert result.exit_code == 1
    assert "results missing: " in result.output and "triage-v1.json: run make eval" in result.output
    assert not (ws / "results" / "baseline.json").exists()


def test_llmeval_baseline_refuses_stale_results_and_keeps_the_old_baseline(tmp_path):
    ws = baseline_workspace(tmp_path)
    old = ws / "results" / "baseline.json"
    old.write_text("synthetic old baseline", encoding="utf-8")
    with (ws / "datasets" / "triage.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"synthetic": "an edited expectation"}\n')
    result = cli_runner.invoke(app, baseline_args(ws))
    assert result.exit_code == 1
    assert "refused: the results are stale:" in result.output
    assert "  triage v1: triage.jsonl changed since these results were written" in result.output
    assert "run make eval, then make baseline" in result.output
    assert old.read_text(encoding="utf-8") == "synthetic old baseline"


def test_llmeval_baseline_refuses_a_recorded_version_without_a_prompt(tmp_path):
    ws = baseline_workspace(tmp_path)
    versions = {"rag": ("v1", "v2"), "triage": ("v1", "v99")}
    (ws / "cassettes" / MANIFEST_FILE).write_text(
        syn.manifest(versions).model_dump_json(), encoding="utf-8"
    )
    result = cli_runner.invoke(app, baseline_args(ws))
    assert result.exit_code == 1
    assert "unknown triage prompt version 'v99'" in result.output


# --- baseline_differences --------------------------------------------------------


def test_equal_baselines_have_no_differences():
    baseline = build_baseline([graded_rag()], syn.manifest())
    assert baseline_differences(baseline, baseline.model_copy()) == []


def test_differences_name_versions_and_cases_on_either_side():
    both = build_baseline([graded_rag("v1"), graded_rag("v2")], syn.manifest())
    only_v1 = build_baseline([graded_rag("v1")], syn.manifest())
    assert baseline_differences(only_v1, both) == ["rag v2: not in the committed baseline"]
    assert baseline_differences(both, only_v1) == ["rag v2: not in the results"]
    fewer = build_baseline(
        [syn.function_results("rag", "v1", graded_rag().cases[:3])], syn.manifest()
    )
    differences = baseline_differences(only_v1, fewer)
    assert "rag v1 rag-004: not in the results" in differences
    assert "rag v1 metric layers.safety: committed 0.75, rebuilt 1.0" in differences
