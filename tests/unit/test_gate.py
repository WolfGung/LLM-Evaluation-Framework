"""The regression gate: key rates of new results against the baseline, within tolerances.

Synthetic data: every result, baseline and tolerance file below (and in
`tests/unit/synthetic_results.py`) is made up for the test. Files are written
only into pytest's `tmp_path`, never into `results/` or `config/`.
"""

import json

import pytest
from typer.testing import CliRunner

from llmeval.baseline import RunResults, build_baseline, write_baseline
from llmeval.cassettes import MANIFEST_FILE
from llmeval.cli import app
from llmeval.gate import (
    GateError,
    GateReport,
    Row,
    Tolerances,
    format_report,
    gate,
    load_tolerances,
    rate_row,
)
from llmeval.results import write_results
from tests.unit import synthetic_results as syn

runner = CliRunner()

TOLERANCES = Tolerances.model_validate(
    {
        "all_checks": 0.05,
        "layers": {
            "retrieval": 0.0,
            "deterministic": 0.05,
            "reference": 0.05,
            "safety": 0.0,
            "judge": 0.10,
        },
        "accuracy": {"category": 0.05, "priority": 0.05},
        "stable_share": 0.10,
        "judge": {"rule_pass": 0.10, "valid": 0.05},
        "pairwise": {"consistent": 0.15, "valid": 0.05},
    }
)


def rag(*cases, version="v1"):
    return syn.function_results("rag", version, list(cases))


def triage(*cases, version="v1"):
    return syn.function_results("triage", version, list(cases))


def tri(case_id, *failing):
    return syn.case_record(case_id, *failing, checks=syn.TRIAGE_CHECKS)


def twenty_rag_cases(failing_safety=()):
    """Twenty graded RAG cases; the ids in `failing_safety` fail a safety check."""
    return [
        syn.case_record(
            f"rag-{n:03d}",
            ("safety/no_trap_leak",) if f"rag-{n:03d}" in failing_safety else (),
            judge="pass",
        )
        for n in range(1, 21)
    ]


def report_of(baseline_run: RunResults, now_run: RunResults):
    baseline = build_baseline(baseline_run.functions, syn.manifest(), baseline_run.pairwise)
    return gate(baseline, now_run, TOLERANCES)


def rows_by_metric(report):
    return {row.metric: row for row in report.rows}


# --- one rate --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("baseline", "now", "verdict"),
    [
        (0.80, 0.80, "ok"),  # equal: what every replay gives
        (0.80, 0.90, "ok"),  # a rise always passes
        (0.80, 0.77, "ok"),  # within the tolerance
        (0.80, 0.75, "ok"),  # exactly at the tolerance
        (0.80, 0.7499, "REGRESSION"),
        (None, 0.5, "n/a"),  # nothing to compare with
        (0.80, None, "missing"),
    ],
)
def test_a_rate_fails_only_when_it_drops_beyond_its_tolerance(baseline, now, verdict):
    assert rate_row("rag v1 all checks", baseline, now, 0.05).verdict == verdict


def test_a_zero_tolerance_fails_any_drop():
    assert rate_row("rag v1 safety layer", 0.9551, 0.9550, 0.0).verdict == "REGRESSION"
    assert rate_row("rag v1 safety layer", 0.9551, 0.9551, 0.0).verdict == "ok"


# --- the gate over results ----------------------------------------------------------


def test_results_equal_to_the_baseline_pass_every_metric():
    pairwise = syn.pairwise_results([syn.pair_case("rag-001", "A", "B")])
    run = RunResults(
        functions=(rag(*twenty_rag_cases()), triage(tri("tri-001"))), pairwise=(pairwise,)
    )
    report = report_of(run, run)
    assert report.passed
    assert {row.verdict for row in report.rows} == {"ok"}
    assert list(rows_by_metric(report)) == [
        "rag v1 cases",
        "rag v1 all checks",
        "rag v1 retrieval layer",
        "rag v1 deterministic layer",
        "rag v1 reference layer",
        "rag v1 safety layer",
        "rag v1 judge layer",
        "rag v1 new safety failures",
        "rag v1 judge rule pass",
        "rag v1 judge valid verdicts",
        "triage v1 cases",
        "triage v1 all checks",
        "triage v1 deterministic layer",
        "triage v1 reference layer",
        "triage v1 category accuracy",
        "triage v1 priority accuracy",
        "rag v1 vs v2 position consistency",
        "rag v1 vs v2 valid pairs",
    ]


def test_a_drop_beyond_the_tolerance_is_a_regression():
    before = RunResults(functions=(rag(*twenty_rag_cases()),))
    cases = twenty_rag_cases()
    cases[:2] = [
        syn.case_record(c.id, ("reference/required_facts",), judge="pass") for c in cases[:2]
    ]
    report = report_of(before, RunResults(functions=(rag(*cases),)))
    rows = rows_by_metric(report)
    # Two of twenty runs is 10 points: beyond 5 for all checks and the reference layer.
    assert rows["rag v1 all checks"].verdict == "REGRESSION"
    assert rows["rag v1 reference layer"].verdict == "REGRESSION"
    assert rows["rag v1 safety layer"].verdict == "ok"
    assert not report.passed
    assert len(report.failed) == 2


def test_cases_missing_from_the_results_fail_even_when_every_rate_rises():
    cases = [tri(f"tri-{n:03d}") for n in range(1, 21)]
    failing = [tri(f"tri-{n:03d}", ("reference/priority_match",)) for n in range(21, 29)]
    before = RunResults(functions=(triage(*cases, *failing),))
    # Only the passing cases are left: every rate is 100%.
    report = report_of(before, RunResults(functions=(triage(*cases),)))
    rows = rows_by_metric(report)
    assert rows["triage v1 all checks"].verdict == "ok"
    coverage = rows["triage v1 cases"]
    assert (coverage.baseline, coverage.now, coverage.allowed) == (28, 20, 0)
    assert (coverage.count, coverage.may, coverage.verdict) == (True, "drop", "REGRESSION")
    assert report.notes == (
        "missing cases: triage v1 tri-021, tri-022, tri-023, tri-024, tri-025 and 3 more",
    )
    assert not report.passed


def test_one_missing_case_is_named():
    cases = [tri(f"tri-{n:03d}") for n in range(1, 4)]
    report = report_of(
        RunResults(functions=(triage(*cases),)), RunResults(functions=(triage(*cases[1:]),))
    )
    assert report.notes == ("missing cases: triage v1 tri-001",)
    assert rows_by_metric(report)["triage v1 cases"].now == 2


def test_a_new_safety_failure_fails_even_when_the_safety_rate_holds():
    before = RunResults(functions=(rag(*twenty_rag_cases(failing_safety={"rag-001"})),))
    # rag-001 now passes, rag-002 newly fails: the same rate, one new failure.
    now = RunResults(functions=(rag(*twenty_rag_cases(failing_safety={"rag-002"})),))
    report = report_of(before, now)
    rows = rows_by_metric(report)
    assert rows["rag v1 safety layer"].verdict == "ok"
    new = rows["rag v1 new safety failures"]
    assert (new.baseline, new.now, new.allowed, new.verdict) == (0, 1, 0, "REGRESSION")
    assert report.notes == ("new safety failure: rag v1 rag-002 safety/no_trap_leak",)
    assert not report.passed


def test_a_known_safety_failure_is_not_new():
    known = RunResults(functions=(rag(*twenty_rag_cases(failing_safety={"rag-001"})),))
    report = report_of(known, known)
    assert rows_by_metric(report)["rag v1 new safety failures"].now == 0
    assert report.passed


def test_triage_category_and_priority_accuracy_are_gated():
    before = RunResults(functions=(triage(*[tri(f"tri-{n:03d}") for n in range(1, 21)]),))
    worse = [
        tri("tri-001", ("reference/category_match",)),
        tri("tri-002", ("reference/category_match",)),
    ]
    worse += [tri(f"tri-{n:03d}") for n in range(3, 21)]
    report = report_of(before, RunResults(functions=(triage(*worse),)))
    rows = rows_by_metric(report)
    assert rows["triage v1 category accuracy"].verdict == "REGRESSION"
    assert (
        rows["triage v1 category accuracy"].baseline,
        rows["triage v1 category accuracy"].now,
    ) == (1.0, 0.9)
    assert rows["triage v1 priority accuracy"].verdict == "ok"


def test_the_stable_share_is_gated_when_cases_repeat():
    steady = [tri(f"tri-{n:03d}", (), ()) for n in range(1, 11)]
    flipping = [tri(f"tri-{n:03d}", ("reference/priority_match",), ()) for n in range(1, 3)]
    flipping += [tri(f"tri-{n:03d}", (), ()) for n in range(3, 11)]
    before = RunResults(functions=(syn.function_results("triage", "v1", steady, repeats=2),))
    now = RunResults(functions=(syn.function_results("triage", "v1", flipping, repeats=2),))
    row = rows_by_metric(report_of(before, now))["triage v1 stable share"]
    # Two of ten cases now flip between repeats: 20 points, beyond 10.
    assert (row.baseline, row.now, row.verdict) == (1.0, 0.8, "REGRESSION")


def test_the_judge_rule_pass_and_validity_are_gated():
    before = RunResults(functions=(rag(*twenty_rag_cases()),))
    cases = twenty_rag_cases()
    cases[0] = syn.case_record("rag-001", judge="invalid")
    cases[1] = syn.case_record("rag-002", judge="invalid")
    cases[2] = syn.case_record("rag-003", judge="fail")
    rows = rows_by_metric(report_of(before, RunResults(functions=(rag(*cases),))))
    valid = rows["rag v1 judge valid verdicts"]
    assert (valid.baseline, valid.now, valid.verdict) == (1.0, 0.9, "REGRESSION")
    rule = rows["rag v1 judge rule pass"]
    # 17 of 18 valid verdicts pass the rule: 5.6 points down, within 10.
    assert (rule.now, rule.verdict) == (round(17 / 18, 4), "ok")


def test_pairwise_position_consistency_is_gated():
    consistent = [syn.pair_case(f"rag-{n:03d}", "A", "B") for n in range(1, 11)]
    biased = [syn.pair_case(f"rag-{n:03d}", "A", "A") for n in range(1, 3)] + consistent[2:]
    before = RunResults(functions=(), pairwise=(syn.pairwise_results(consistent),))
    now = RunResults(functions=(), pairwise=(syn.pairwise_results(biased),))
    row = rows_by_metric(report_of(before, now))["rag v1 vs v2 position consistency"]
    assert (row.baseline, row.now, row.verdict) == (1.0, 0.8, "REGRESSION")


def test_results_the_baseline_has_but_the_run_lacks_are_missing():
    pairwise = syn.pairwise_results([syn.pair_case("rag-001", "A", "B")])
    before = RunResults(
        functions=(rag(*twenty_rag_cases()), triage(tri("tri-001"))), pairwise=(pairwise,)
    )
    report = report_of(before, RunResults(functions=(rag(*twenty_rag_cases()),)))
    rows = rows_by_metric(report)
    assert rows["triage v1 results"].verdict == "missing"
    assert rows["rag v1 vs v2 results"].verdict == "missing"
    assert not report.passed


def test_a_layer_without_a_tolerance_is_refused():
    case = syn.case_record("rag-001", checks=("style/short_answer",))
    run = RunResults(functions=(rag(case),))
    with pytest.raises(GateError, match="no tolerance for the style layer"):
        report_of(run, run)


# --- the table ---------------------------------------------------------------------


def test_the_report_is_a_table_with_a_verdict_line():
    rows = (
        rate_row("rag v1 all checks", 0.7756, 0.7756, 0.05),
        rate_row("rag v1 safety layer", 0.9551, 0.9487, 0.0),
        Row("rag v1 new safety failures", 0, 1, 0, "REGRESSION", count=True, may="rise"),
        rate_row("rag v1 stable share", None, None, 0.10),
    )
    lines = format_report(GateReport(rows=rows, notes=("new safety failure: rag v1 rag-045",)))
    assert lines[0].split() == ["metric", "baseline", "now", "allowed", "verdict"]
    # The allowed column says which way a metric may move: a rate may drop,
    # the count of new safety failures may not rise at all.
    assert lines[1].split() == [
        "rag",
        "v1",
        "all",
        "checks",
        "77.56%",
        "77.56%",
        "drop",
        "5.00",
        "pp",
        "ok",
    ]
    assert lines[2].split()[-5:] == ["94.87%", "drop", "0.00", "pp", "REGRESSION"]
    assert lines[3].split()[-5:] == ["0", "1", "rise", "0", "REGRESSION"]
    assert lines[4].split()[-6:] == ["n/a", "n/a", "drop", "10.00", "pp", "n/a"]
    # Numbers are right-aligned under their heading, verdicts left-aligned.
    assert lines[1].index("77.56%") + len("77.56%") == lines[0].index("baseline") + len("baseline")
    assert {line.rindex(line.split()[-1]) for line in lines[1:5]} == {lines[0].index("verdict")}
    assert lines[-2] == "new safety failure: rag v1 rag-045"
    assert (
        lines[-1]
        == "gate failed: 2 of 3 compared metrics regressed beyond tolerance or are missing"
    )


def test_a_passing_report_says_so():
    lines = format_report(GateReport(rows=(rate_row("rag v1 all checks", 0.5, 0.5, 0.05),)))
    assert lines[-1] == "gate passed: 1 compared metric within tolerance"


# --- the tolerances file -----------------------------------------------------------


def write_tolerances(tmp_path, data):
    path = tmp_path / "gate.yaml"
    path.write_text(json.dumps(data), encoding="utf-8")  # JSON is valid YAML
    return path


def test_load_tolerances(tmp_path):
    path = write_tolerances(tmp_path, TOLERANCES.model_dump())
    assert load_tolerances(path) == TOLERANCES


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.pop("stable_share"), "stable_share"),
        (lambda d: d["layers"].update(style=0.1), "style"),
        (lambda d: d["judge"].update(valid=-0.1), "valid"),
        (lambda d: d["pairwise"].update(consistent=1.5), "consistent"),
    ],
)
def test_load_tolerances_refuses_a_wrong_file(tmp_path, change, message):
    data = TOLERANCES.model_dump()
    change(data)
    with pytest.raises(GateError, match=message) as caught:
        load_tolerances(write_tolerances(tmp_path, data))
    assert "gate.yaml" in str(caught.value)


def test_load_tolerances_refuses_broken_yaml_and_a_missing_file(tmp_path):
    broken = tmp_path / "gate.yaml"
    broken.write_text("all_checks: [0.05\n", encoding="utf-8")
    with pytest.raises(GateError, match="gate.yaml"):
        load_tolerances(broken)
    with pytest.raises(GateError, match="missing.yaml"):
        load_tolerances(tmp_path / "missing.yaml")


# --- llmeval gate ------------------------------------------------------------------


def gate_workspace(tmp_path, now_cases=None):
    """A baseline of RAG v1 and v2 (twenty cases each) and their comparison in
    tmp_path, and results equal to it except for `now_cases` (RAG v1)."""
    pair = syn.pairwise_results([syn.pair_case("rag-001", "A", "B")])
    versions = [rag(*twenty_rag_cases()), rag(*twenty_rag_cases(), version="v2")]
    baseline = build_baseline(versions, syn.manifest({"rag": ("v1", "v2")}), [pair])
    write_baseline(baseline, tmp_path / "baseline.json")
    results = tmp_path / "results"
    write_results(rag(*(now_cases or twenty_rag_cases())), results)
    write_results(versions[1], results)
    write_results(pair, results)
    tolerances = write_tolerances(tmp_path, TOLERANCES.model_dump())
    (tmp_path / "cassettes").mkdir()
    return tmp_path, tolerances


def gate_args(ws, tolerances):
    return [
        "gate",
        "--results-dir",
        str(ws / "results"),
        "--baseline",
        str(ws / "baseline.json"),
        "--tolerances",
        str(tolerances),
        "--cassettes-dir",
        str(ws / "cassettes"),
    ]


def test_llmeval_gate_passes_results_equal_to_the_baseline(tmp_path):
    ws, tolerances = gate_workspace(tmp_path)
    result = runner.invoke(app, gate_args(ws, tolerances))
    assert result.exit_code == 0, result.output
    assert (
        f"gate: {ws / 'results'} (replay results) against {ws / 'baseline.json'}" in result.output
    )
    assert "rag v1 all checks" in result.output
    assert result.output.rstrip().endswith("within tolerance")


def test_llmeval_gate_exits_1_on_a_regression(tmp_path):
    cases = twenty_rag_cases(failing_safety={"rag-003"})
    ws, tolerances = gate_workspace(tmp_path, now_cases=cases)
    result = runner.invoke(app, gate_args(ws, tolerances))
    assert result.exit_code == 1
    assert "REGRESSION" in result.output
    assert "new safety failure: rag v1 rag-003 safety/no_trap_leak" in result.output
    assert "gate failed: " in result.output


def test_llmeval_gate_refuses_missing_results(tmp_path):
    ws, tolerances = gate_workspace(tmp_path)
    (ws / "results" / "rag-v1.json").unlink()
    result = runner.invoke(app, gate_args(ws, tolerances))
    assert result.exit_code == 1
    assert "results missing: " in result.output and "rag-v1.json: run make eval" in result.output


def test_llmeval_gate_without_a_baseline(tmp_path):
    ws, tolerances = gate_workspace(tmp_path)
    (ws / "baseline.json").unlink()
    # No recorded run either: nothing to gate yet.
    result = runner.invoke(app, gate_args(ws, tolerances))
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "pending first recorded run"
    # A recorded run without a baseline cannot pass the gate.
    (ws / "cassettes" / MANIFEST_FILE).write_text(
        syn.manifest().model_dump_json(), encoding="utf-8"
    )
    result = runner.invoke(app, gate_args(ws, tolerances))
    assert result.exit_code == 1
    assert f"no baseline in {ws / 'baseline.json'}: run make baseline" in result.output


def test_llmeval_gate_names_live_results(tmp_path):
    ws, tolerances = gate_workspace(tmp_path)
    for path in (ws / "results").glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["mode"] = "live"
        path.write_text(json.dumps(data), encoding="utf-8")
    result = runner.invoke(app, gate_args(ws, tolerances))
    assert result.exit_code == 0, result.output
    assert "(live results)" in result.output
