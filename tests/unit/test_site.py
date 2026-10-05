"""The published page: the README results table and a link to the Allure report.

Synthetic data: every result and manifest below is made up for the test and
written only into `tmp_path`, never into the repository.
"""

from html import escape

import pytest

from llmeval.agreement import AgreementReport, SampleSummary, write_agreement
from llmeval.cassettes import write_manifest
from llmeval.results import write_results
from tests.unit import synthetic_results as syn
from tools import render, sections, site
from tools.formatting import Table


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "cassettes").mkdir()
    return tmp_path


def record_run(ws, manifest=None):
    versions = {"triage": ("v1",)}
    manifest = manifest or syn.manifest(versions)
    write_manifest(ws / "cassettes", manifest)
    case = syn.case_record("tri-001", (), checks=syn.TRIAGE_CHECKS)
    write_results(syn.function_results("triage", "v1", [case]), ws / "results")


def page(ws) -> str:
    return site.build(ws / "results", ws / "cassettes")


def test_the_page_carries_the_readme_table_and_its_line(ws):
    record_run(ws)
    html = page(ws)
    table = render.table(*render.load(ws / "results", ws / "cassettes"))
    assert '<th scope="col">triage v1</th>' in html
    for row in table.rows:
        assert f'<th scope="row">{row[0]}</th>' in html
        for cell in row[1:]:
            assert f"<td>{cell}</td>" in html
    assert f"<p>{table.line}</p>" in html


def test_the_page_links_the_allure_report(ws):
    record_run(ws)
    assert '<a href="report/">' in page(ws)


def test_the_page_is_a_plain_accessible_document(ws):
    record_run(ws)
    html = page(ws)
    assert html.startswith('<!doctype html>\n<html lang="en">')
    assert "<caption>" in html
    assert "<title>LLM Evaluation Framework results</title>" in html


def test_without_a_recorded_run_the_page_says_pending(ws):
    html = page(ws)
    assert "<p>pending first recorded run</p>" in html
    assert "<table>" not in html
    assert '<a href="report/">' in html


def test_text_from_the_results_is_escaped(ws):
    odd = syn.manifest({"triage": ("v1",)}).model_copy(
        update={"models": {"system": "a<b>&c:free", "judge": "synthetic/judge:free"}}
    )
    record_run(ws, manifest=odd)
    html = page(ws)
    assert "a&lt;b&gt;&amp;c:free" in html
    assert "a<b>" not in html


def test_main_writes_the_page_and_the_same_files_give_the_same_bytes(ws, capsys):
    record_run(ws)
    out = ws / "site" / "index.html"
    args = ["--out", str(out), "--results-dir", str(ws / "results")]
    args += ["--cassettes-dir", str(ws / "cassettes")]
    assert site.main(args) == 0
    first = out.read_bytes()
    assert site.main(args) == 0
    assert out.read_bytes() == first
    assert str(out) in capsys.readouterr().out


def test_main_refuses_broken_results_with_one_line(ws, capsys):
    write_manifest(ws / "cassettes", syn.manifest({"triage": ("v1",)}))
    out = ws / "site" / "index.html"
    code = site.main(
        [
            "--out",
            str(out),
            "--results-dir",
            str(ws / "results"),
            "--cassettes-dir",
            str(ws / "cassettes"),
        ]
    )
    assert code == 1
    assert "results missing" in capsys.readouterr().err
    assert not out.exists()


def record_rag_run(ws):
    """A graded RAG run of two versions, its pairwise comparison and a pending agreement."""
    write_manifest(ws / "cassettes", syn.manifest({"rag": ("v1", "v2")}))
    for version in ("v1", "v2"):
        cases = [
            syn.case_record("rag-001", (), judge="pass"),
            syn.case_record("rag-002", (), judge="fail"),
        ]
        write_results(syn.function_results("rag", version, cases), ws / "results")
    pairs = [syn.pair_case("rag-001", "A", "A"), syn.pair_case("rag-002", "tie", "tie")]
    write_results(syn.pairwise_results(pairs), ws / "results")
    report = AgreementReport(
        status="pending human labels",
        standards="Synthetic standards.",
        judge_model=syn.JUDGE_MODEL,
        rubric_sha256=syn.RUBRIC_SHA256,
        sample=SampleSummary(size=2, seed=1, judge_pass=0, judge_fail=2),
        labelled=0,
        agreed=0,
        agreement_rate=None,
        kappa=None,
        kappa_note="no labelled answers",
        confusion={
            "judge_pass": {"human_pass": 0, "human_fail": 0},
            "judge_fail": {"human_pass": 0, "human_fail": 0},
        },
        disagreements=[],
        stale=[],
        unjudged=[],
    )
    write_agreement(report, ws / "results")


def test_the_page_carries_the_pairwise_block_as_the_readme_does(ws):
    record_rag_run(ws)
    html = page(ws)
    recorded = render.recorded(render.Sources(ws / "results", ws / "cassettes"))
    parts = sections.pairwise(recorded)
    tables = [part for part in parts if isinstance(part, Table)]
    assert len(tables) == 1
    for row in tables[0].rows:
        assert f'<th scope="row">{row[0]}</th><td>{row[1]}</td>' in html
    for paragraph in (part for part in parts if isinstance(part, str)):
        assert f"<p>{escape(paragraph)}</p>" in html
    assert "<h2>The two RAG prompt versions, compared by the judge</h2>" in html


def test_the_page_carries_the_judge_agreement(ws):
    record_rag_run(ws)
    html = page(ws)
    assert "<h2>Can the judge be trusted?</h2>" in html
    assert "<p>pending human labels</p>" in html
    assert "all 2 answers the judge failed and 0 it passed" in html


def test_the_page_says_the_behaviors_counts_are_not_layer_pass_rates(ws):
    html = page(ws)
    assert "Behaviors" in html
    assert "not layer pass rates" in html


def test_without_a_recorded_run_every_section_is_pending(ws):
    html = page(ws)
    assert html.count("<p>pending first recorded run</p>") == 3
