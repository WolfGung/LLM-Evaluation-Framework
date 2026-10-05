"""The published page: the README results table and a link to the Allure report.

Synthetic data: every result and manifest below is made up for the test and
written only into `tmp_path`, never into the repository.
"""

import pytest

from llmeval.cassettes import write_manifest
from llmeval.results import write_results
from tests.unit import synthetic_results as syn
from tools import render, site


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
    assert "<title>LLM-Evaluation-Framework results</title>" in html


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
