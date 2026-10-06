"""The README results block, rendered from results/ and the run manifest.

Synthetic data: every result, manifest and README below is made up for the
test and written only into `tmp_path`, never into the repository.
"""

import time

import pytest

from llmeval.cassettes import write_manifest
from llmeval.results import write_results
from tests.unit import synthetic_results as syn
from tools import render

START, END = render.START_MARKER, render.END_MARKER


def timed(record, *latencies_ms):
    """The record with each run's call taking the given time."""
    runs = [
        run.model_copy(update={"call": run.call.model_copy(update={"latency_ms": ms})})
        for run, ms in zip(record.runs, latencies_ms, strict=True)
    ]
    return record.model_copy(update={"runs": runs})


def priced(record, cost):
    """The record with each run's call costing `cost` (None: unknown)."""
    source = "unknown" if cost is None else "provider"
    update = {"cost_usd": cost, "cost_source": source}
    runs = [
        run.model_copy(update={"call": run.call.model_copy(update=update)}) for run in record.runs
    ]
    return record.model_copy(update={"runs": runs})


@pytest.fixture
def ws(tmp_path):
    """A results directory, a cassette directory and a README with the markers."""
    (tmp_path / "results").mkdir()
    (tmp_path / "cassettes").mkdir()
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\nIntro.\n\n{START}\nold\n{END}\n\nAfter.\n", encoding="utf-8")
    return tmp_path


def record_run(ws, *results, manifest=None):
    write_manifest(ws / "cassettes", manifest or syn.manifest({"rag": ("v1",), "triage": ("v1",)}))
    for result in results:
        write_results(result, ws / "results")


def rag_v1():
    rag_001 = timed(syn.case_record("rag-001", (), (), judge="pass"), 800, 1200)
    attacked = syn.case_record("rag-047", ("safety/no_trap_leak",), (), category="safety")
    # A safety case that fails a fact but no safety check: no safety failure.
    resisted = syn.case_record("rag-048", ("reference/required_facts",), (), category="safety")
    cases = [rag_001, timed(attacked, 1000, 2550), timed(resisted, 900, 1100)]
    return syn.function_results("rag", "v1", cases)


def triage_v1():
    case = syn.case_record("tri-001", (), checks=syn.TRIAGE_CHECKS, category="order_status")
    return syn.function_results("triage", "v1", [timed(case, 400)])


def block(ws):
    return render.render_block(ws / "results", ws / "cassettes")


TABLE = """\
| Metric | rag v1 | triage v1 |
|---|---:|---:|
| All checks | 66.7% | 100.0% |
| Retrieval layer | 100.0% | — |
| Deterministic layer | 100.0% | 100.0% |
| Reference layer | 83.3% | 100.0% |
| Safety layer | 83.3% | — |
| Judge layer | 100.0% | — |
| Safety cases with no safety failure | 1 of 2 | — |
| Stable cases | 33.3% | — |
| Cost (system + judge calls) | $0.00 (free models) | $0.00 (free models) |
| Latency p50 / p95 (system calls) | 1.0 s / 2.6 s | 0.4 s / 0.4 s |"""


def test_the_block_is_the_main_table_and_the_lines_under_it(ws):
    record_run(
        ws,
        rag_v1(),
        triage_v1(),
        manifest=syn.manifest({"rag": ("v1",), "triage": ("v1",)}, repeats=2),
    )
    # The p95 call of rag v1 (rag-047, 2550 ms) has a repeat of 1000 ms; the one
    # call of triage v1 ran once and has none.
    assert block(ws) == (
        f"\n{TABLE}\n\n"
        "Each column is one prompt version. Each case ran 2 times; each layer's rate is over the "
        "runs that layer checks (the judge graded the first run of each judged case). A safety "
        "case has no "
        "safety failure when every safety check passed on every run. Recorded on 2026-01-01 "
        "(UTC) with synthetic/system:free (system) and synthetic/judge:free (judge), "
        "2 calls.\n\n"
        "Latency: the system call at or above its column's p95 whose case ran more than once "
        "took at least twice as long as a repeat of the same request (2.6 times as long) while "
        "writing 1 answer token against the repeat's 1, so the tail is delay at the provider's "
        "shared free endpoint, not time the request needs.\n\n"
    )


def test_without_a_manifest_the_block_says_pending(ws):
    assert block(ws) == "\npending first recorded run\n\n"


def test_a_manifest_without_its_results_is_an_error_not_pending(ws):
    write_manifest(ws / "cassettes", syn.manifest({"triage": ("v1",)}))
    with pytest.raises(render.RenderError, match="results missing"):
        block(ws)


@pytest.mark.parametrize(
    ("count", "total", "text"),
    [
        (1, 8, "12.5%"),
        (13, 16, "81.3%"),  # 81.25: a half rounds up, never to even
        (2, 3, "66.7%"),
        (121, 156, "77.6%"),
        (0, 5, "0.0%"),
        (5, 5, "100.0%"),
        (0, 0, "—"),
    ],
)
def test_percentages_have_one_decimal_and_a_half_rounds_up(count, total, text):
    assert render.percent(count, total) == text


def test_seconds_have_one_decimal_and_a_half_rounds_up():
    assert render.seconds(2550.0) == "2.6 s"
    assert render.seconds(25290.3) == "25.3 s"
    assert render.seconds(None) == "—"


def repeated(case_id, *runs):
    """A triage case with one run per (latency ms, completion tokens)."""
    record = syn.case_record(case_id, *([()] * len(runs)), checks=syn.TRIAGE_CHECKS)
    calls = [
        run.call.model_copy(update={"latency_ms": ms, "completion_tokens": tokens})
        for run, (ms, tokens) in zip(record.runs, runs, strict=True)
    ]
    return record.model_copy(
        update={
            "runs": [
                run.model_copy(update={"call": call})
                for run, call in zip(record.runs, calls, strict=True)
            ]
        }
    )


def latency_line(ws, *cases, manifest=None):
    """The latency line of triage v1 with these cases: the block after the table's line."""
    manifest = manifest or syn.manifest({"triage": ("v1",)}, repeats=3)
    record_run(ws, syn.function_results("triage", "v1", list(cases)), manifest=manifest)
    parts = block(ws).strip().split("\n\n")
    return parts[2] if len(parts) > 2 else None


# Sixteen cases of three repeats at 1 s with 50 tokens. With four slow cases
# that makes 60 calls: p95 is the 57th value, so the tail is the slow call of
# each slow case, and its two repeats (at repeat_ms) stay out of it.
QUICK = [repeated(f"tri-{n:03d}", (1000, 50), (1000, 50), (1000, 50)) for n in range(5, 21)]


def slow(n, ms, tokens=50, repeat_ms=1000):
    return repeated(f"tri-00{n}", (ms, tokens), (repeat_ms, 50), (repeat_ms, 50))


def test_a_tail_slower_than_its_repeats_is_named_as_waiting(ws):
    line = latency_line(
        ws, *QUICK, slow(1, 30000), slow(2, 20000, 40), slow(3, 9000), slow(4, 10000)
    )
    assert line == (
        "Latency: each of the 4 system calls at or above their column's p95 took at least twice "
        "as long as a repeat of the same request (a median of 15 times as long) while writing a "
        "median of 50 answer tokens against the repeats' 50, so the tail is delay at the "
        "provider's shared free endpoint, not time the request needs."
    )


def test_a_tail_slow_on_every_repeat_comes_from_the_requests(ws):
    line = latency_line(ws, *QUICK, *(slow(n, 3000, repeat_ms=2000) for n in (1, 2, 3, 4)))
    assert line == (
        "Latency: of the 4 system calls at or above their column's p95, none of them took even "
        "twice as long as a repeat of the same request would need for the same answer, so the "
        "slow requests were slow on every try: the tail comes from the requests themselves, not "
        "from delay at the provider."
    )


@pytest.mark.parametrize(("waiting", "share"), [(3, "most of the tail"), (1, "part of the tail")])
def test_a_tail_that_partly_waited_says_how_much(ws, waiting, share):
    tail = [slow(n, 9000 if n <= waiting else 3000, repeat_ms=2000) for n in (1, 2, 3, 4)]
    line = latency_line(ws, *QUICK, *tail)
    assert line.startswith(f"Latency: {waiting} of the 4 system calls at or above their column")
    assert f"so {share} is delay at the provider" in line


def test_a_longer_answer_is_not_taken_for_waiting(ws):
    # 2.5 times as long as the repeat, but with 3 times its tokens: the work, not a wait.
    line = latency_line(ws, *QUICK, *(slow(n, 2500, 150) for n in (1, 2, 3, 4)))
    assert "none of them took even twice as long" in line


def test_a_tail_without_repeats_gets_no_latency_line(ws):
    once = [repeated(f"tri-{n:03d}", (1000 * n, 50)) for n in range(1, 4)]
    assert latency_line(ws, *once, manifest=syn.manifest({"triage": ("v1",)})) is None


def test_a_paid_system_model_is_not_called_a_free_endpoint(ws):
    paid = syn.manifest({"triage": ("v1",)}, repeats=3).model_copy(
        update={"models": {"system": "synthetic/system", "judge": "synthetic/judge"}}
    )
    tail = [slow(n, 30000) for n in (1, 2, 3, 4)]
    line = latency_line(ws, *QUICK, *tail, manifest=paid)
    assert line.endswith("so the tail is delay at the provider, not time the request needs.")


def test_a_paid_run_shows_the_provider_cost(ws):
    paid = syn.manifest({"triage": ("v1",)}).model_copy(
        update={"models": {"system": "synthetic/system", "judge": "synthetic/judge"}}
    )
    case = priced(syn.case_record("tri-001", (), checks=syn.TRIAGE_CHECKS), 0.0012)
    record_run(ws, syn.function_results("triage", "v1", [case]), manifest=paid)
    assert "| Cost (system + judge calls) | $0.0012 |" in block(ws)


def test_free_model_ids_with_a_reported_cost_show_the_cost(ws):
    case = priced(syn.case_record("tri-001", (), checks=syn.TRIAGE_CHECKS), 0.0012)
    record_run(
        ws, syn.function_results("triage", "v1", [case]), manifest=syn.manifest({"triage": ("v1",)})
    )
    assert "| Cost (system + judge calls) | $0.0012 |" in block(ws)


def test_unknown_costs_are_named_never_counted_as_zero(ws):
    case = priced(syn.case_record("tri-001", (), (), checks=syn.TRIAGE_CHECKS), None)
    record_run(
        ws, syn.function_results("triage", "v1", [case]), manifest=syn.manifest({"triage": ("v1",)})
    )
    assert "| Cost (system + judge calls) | $0.00 known, 2 calls unknown |" in block(ws)


def test_the_judge_cost_counts_in_the_cost_per_run(ws):
    graded = priced(syn.case_record("rag-001", (), judge="pass"), 0.001)
    judge = graded.runs[0].judge
    judge = judge.model_copy(update={"call": judge.call.model_copy(update={"cost_usd": 0.002})})
    run = graded.runs[0].model_copy(update={"judge": judge})
    graded = graded.model_copy(update={"runs": [run]})
    paid = syn.manifest({"rag": ("v1",)}).model_copy(
        update={"models": {"system": "synthetic/system", "judge": "synthetic/judge"}}
    )
    record_run(ws, syn.function_results("rag", "v1", [graded]), manifest=paid)
    assert "| Cost (system + judge calls) | $0.003 |" in block(ws)


def test_a_stability_subset_is_named_in_the_line(ws):
    subset = syn.manifest({"triage": ("v1",)}, repeats=3).model_copy(
        update={"stability_cases": ("tri-001", "tri-002")}
    )
    case = syn.case_record("tri-001", (), checks=syn.TRIAGE_CHECKS)
    record_run(ws, syn.function_results("triage", "v1", [case]), manifest=subset)
    expected = "2 cases ran 3 times, the others once; each layer's rate is over the runs that layer"
    assert f"{expected} checks." in block(ws)


def test_one_repeat_is_once_and_several_days_are_a_range(ws):
    later = syn.manifest({"triage": ("v1",)}).model_copy(
        update={"recorded_to": syn.TIME.replace(day=3)}
    )
    case = syn.case_record("tri-001", (), checks=syn.TRIAGE_CHECKS)
    record_run(ws, syn.function_results("triage", "v1", [case]), manifest=later)
    text = block(ws)
    assert "Each case ran once; each layer's rate is over the runs that layer checks." in text
    assert "Recorded from 2026-01-01 to 2026-01-03 (UTC)" in text


def test_ungraded_results_name_no_judge(ws):
    case = syn.case_record("tri-001", (), checks=syn.TRIAGE_CHECKS)
    record_run(
        ws, syn.function_results("triage", "v1", [case]), manifest=syn.manifest({"triage": ("v1",)})
    )
    text = block(ws)
    assert "with synthetic/system:free (system), 2 calls." in text
    # Neither the judge model nor its grading is named (the cost row's label still is).
    assert "synthetic/judge" not in text
    assert "the judge graded" not in text
    # No safety case in any column: neither the safety row nor its definition.
    assert "safety" not in text.lower()


def test_the_judge_grading_every_run_is_named_in_the_line(ws):
    every = syn.manifest({"rag": ("v1",)}).model_copy(update={"judge_repeats": "all"})
    record_run(
        ws, syn.function_results("rag", "v1", [syn.case_record("rag-001", ())]), manifest=every
    )
    assert "(the judge graded every run of each judged case)" in block(ws)


def test_the_pairwise_calls_are_said_to_be_in_no_column(ws):
    versions = {"rag": ("v1", "v2")}
    results = [
        syn.function_results("rag", version, [syn.case_record("rag-001", (), judge="pass")])
        for version in versions["rag"]
    ]
    record_run(
        ws,
        *results,
        syn.pairwise_results([syn.pair_case("rag-001", "A", "B")]),
        manifest=syn.manifest(versions),
    )
    assert "The pairwise comparison calls are not counted in any column." in block(ws)


def test_without_pairwise_results_the_line_does_not_mention_them(ws):
    record_run(ws, rag_v1(), triage_v1())
    assert "pairwise" not in block(ws)


@pytest.fixture
def far_time_zone(monkeypatch):
    """Local time UTC+14, where 12:00 UTC on 1 January is already 2 January."""
    monkeypatch.setenv("TZ", "Pacific/Kiritimati")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_the_dates_are_utc_whatever_the_local_time_zone(ws, far_time_zone):
    record_run(ws, rag_v1(), triage_v1())
    assert "Recorded on 2026-01-01 (UTC)" in block(ws)


def run_main(ws, *args):
    return render.main(
        [
            *args,
            str(ws / "README.md"),
            "--results-dir",
            str(ws / "results"),
            "--cassettes-dir",
            str(ws / "cassettes"),
        ]
    )


def test_write_replaces_only_the_block_and_check_then_passes(ws, capsys):
    record_run(ws, rag_v1(), triage_v1())
    assert run_main(ws, "--check") == 1
    assert "python -m tools.render --write" in capsys.readouterr().err
    assert run_main(ws, "--write") == 0
    text = (ws / "README.md").read_text(encoding="utf-8")
    assert text == f"# Title\n\nIntro.\n\n{START}\n{block(ws)}{END}\n\nAfter.\n"
    assert run_main(ws, "--check") == 0
    assert run_main(ws, "--write") == 0
    assert (ws / "README.md").read_text(encoding="utf-8") == text


def test_without_a_flag_the_block_is_printed(ws, capsys):
    assert run_main(ws) == 0
    assert capsys.readouterr().out == f"{START}\n\npending first recorded run\n\n{END}\n"


@pytest.mark.parametrize(
    "text",
    [f"{START}\n{START}\nx\n{END}\n", f"{END}\nx\n{START}\n"],
    ids=["twice", "reversed"],
)
def test_a_readme_with_markers_out_of_order_is_refused(ws, capsys, text):
    (ws / "README.md").write_text(text, encoding="utf-8")
    assert run_main(ws, "--check") == 1
    assert run_main(ws, "--write") == 1
    assert (ws / "README.md").read_text(encoding="utf-8") == text
    assert "results:" in capsys.readouterr().err


def test_broken_results_exit_1_with_one_line(ws, capsys):
    write_manifest(ws / "cassettes", syn.manifest({"triage": ("v1",)}))
    (ws / "results" / "triage-v1.json").write_text("{}", encoding="utf-8")
    assert run_main(ws, "--check") == 1
    assert "triage-v1.json: not valid results" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("rate", "text"), [(0.9551, "95.5%"), (0.8125, "81.3%"), (1.0, "100.0%"), (None, "—")]
)
def test_a_stored_rate_has_one_decimal_and_a_half_rounds_up(rate, text):
    assert render.share(rate) == text
