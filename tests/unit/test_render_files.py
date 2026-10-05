"""Generated blocks in Markdown files: markers, history quotes and the guard.

Synthetic data: every file, result and manifest below is made up for the test
and written only into `tmp_path`, never into the repository.
"""

import pytest

from llmeval.cassettes import write_manifest
from llmeval.results import write_results
from tests.unit import synthetic_results as syn
from tools import render

PENDING = "\npending first recorded run\n\n"


def block(name: str, body: str = "old") -> str:
    return f"<!-- {name}:start -->\n{body}\n<!-- {name}:end -->"


def test_block_names_are_listed_in_order_history_included():
    history = block("history", "Commit abc1234.")
    text = f"# T\n\n{block('results')}\n\n{history}\n{block('pairwise')}\n"
    assert render.block_names(text) == ["results", "history", "pairwise"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("<!-- results:start -->\nx\n", "no <!-- results:end --> line"),
        ("x\n<!-- results:end -->\n", "<!-- results:end --> without its start"),
        (
            "<!-- results:start -->\n<!-- pairwise:start -->\nx\n<!-- pairwise:end -->\n"
            "<!-- results:end -->\n",
            "<!-- pairwise:start --> inside the results block",
        ),
        (
            "<!-- results:start -->\nx\n<!-- pairwise:end -->\n",
            "<!-- pairwise:end --> inside the results block",
        ),
    ],
    ids=["unclosed", "stray-end", "nested", "crossed"],
)
def test_broken_markers_are_refused(text, message):
    with pytest.raises(render.RenderError, match=message):
        render.block_names(text)


def test_replace_blocks_fills_each_block_and_keeps_history_as_written():
    history = block("history", "Commit abc1234 truncated calls.")
    text = f"Intro.\n\n{block('results')}\n\n{history}\n\n{block('results')}\n\nEnd.\n"
    filled = render.replace_blocks(text, {"results": "\nnew\n\n"})
    assert filled == (
        f"Intro.\n\n<!-- results:start -->\n\nnew\n\n<!-- results:end -->\n\n{history}\n\n"
        "<!-- results:start -->\n\nnew\n\n<!-- results:end -->\n\nEnd.\n"
    )
    # Replacing again changes nothing.
    assert render.replace_blocks(filled, {"results": "\nnew\n\n"}) == filled


def test_a_block_name_without_a_renderer_is_refused():
    with pytest.raises(render.RenderError, match="unknown block 'nonsense'"):
        render.replace_blocks(block("nonsense"), {"results": "\nx\n\n"})


def test_outside_blocks_drops_the_content_of_every_block():
    text = f"Before.\n{block('results', '50%')}\nMiddle.\n{block('history', '13 of 154')}\nAfter.\n"
    outside = render.outside_blocks(text)
    assert "50%" not in outside and "13 of 154" not in outside
    assert "Before." in outside and "Middle." in outside and "After." in outside


@pytest.mark.parametrize(
    "text",
    [
        "141 cases pass.",
        "It passed 20 of 38 pairs.",
        "rag v1 passed 118/156 runs.",
        "The judge passed 52.6% of them.",
        "13 judge calls ran out of tokens.",
        "It costs $0.00 per run.",
        "7 answers the judge failed.",
    ],
)
def test_a_number_from_the_results_outside_a_block_is_found(text):
    assert render.bare_results(text), text


@pytest.mark.parametrize(
    "text",
    [
        "Python 3.12 and fastapi 0.142.2.",
        "Free variants allow 20 requests per minute and 50 requests per day.",
        "The judge's max_tokens is 4096.",
        "Run `make test`, then `make eval`.",
        "The case rag-041 is a direct injection.",
        "Released on 2026-10-05.",
    ],
)
def test_text_without_numbers_from_the_results_passes(text):
    assert render.bare_results(text) == []


def test_a_history_quote_must_name_its_commit():
    good = block("history", "In commit 66b4a3a, 13 of 154 judge calls ran out of tokens.")
    bad = block("history", "Once, 13 of 154 judge calls ran out of tokens.")
    assert render.history_without_commit(f"{good}\n") == []
    assert render.history_without_commit(f"{good}\n{bad}\n") == [
        "Once, 13 of 154 judge calls ran out of tokens."
    ]


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "results").mkdir()
    (tmp_path / "cassettes").mkdir()
    (tmp_path / "docs").mkdir()
    return tmp_path


def sources(ws) -> render.Sources:
    return render.Sources(results_dir=ws / "results", cassettes_dir=ws / "cassettes")


def run_main(ws, *args):
    return render.main(
        [*args, "--results-dir", str(ws / "results"), "--cassettes-dir", str(ws / "cassettes")]
    )


def test_every_block_of_a_file_is_pending_without_a_recorded_run(ws):
    history = block("history", "Commit abc1234.")
    text = f"{block('results')}\n\n{history}\n"
    rendered = render.render_text(text, sources(ws))
    assert rendered == f"<!-- results:start -->\n{PENDING}<!-- results:end -->\n\n{history}\n"


def test_check_and_write_cover_every_file_given(ws, capsys):
    readme, doc = ws / "README.md", ws / "docs" / "03-x.md"
    readme.write_text(f"# R\n\n{block('results')}\n", encoding="utf-8")
    doc.write_text(f"# D\n\nNo blocks here.\n\n{block('results')}\n", encoding="utf-8")
    plain = ws / "docs" / "01-plain.md"
    plain.write_text("# Plain\n\nNo blocks.\n", encoding="utf-8")
    files = [str(readme), str(doc), str(plain)]
    assert run_main(ws, "--check", *files) == 1
    err = capsys.readouterr().err
    assert "README.md: the generated blocks differ from results/" in err
    assert "03-x.md: the generated blocks differ from results/" in err
    assert "python -m tools.render --write" in err
    assert run_main(ws, "--write", *files) == 0
    assert readme.read_text(encoding="utf-8") == (
        f"# R\n\n<!-- results:start -->\n{PENDING}<!-- results:end -->\n"
    )
    assert plain.read_text(encoding="utf-8") == "# Plain\n\nNo blocks.\n"
    assert run_main(ws, "--check", *files) == 0
    assert "the generated blocks match results/" in capsys.readouterr().out


def test_an_unknown_block_in_a_file_exits_1_with_one_line(ws, capsys):
    path = ws / "README.md"
    path.write_text(block("nonsense"), encoding="utf-8")
    assert run_main(ws, "--check", str(path)) == 1
    assert "README.md: unknown block 'nonsense'" in capsys.readouterr().err


def test_the_default_files_are_the_readme_and_the_docs():
    files = render.default_files()
    assert files[0] == render.README
    assert all(path.parent == render.DOCS and path.suffix == ".md" for path in files[1:])
    assert files[1:] == sorted(files[1:])


def test_without_a_flag_each_block_is_printed_once(ws, capsys):
    path = ws / "README.md"
    path.write_text(f"{block('results')}\n{block('results')}\n", encoding="utf-8")
    assert run_main(ws, str(path)) == 0
    assert capsys.readouterr().out == f"<!-- results:start -->\n{PENDING}<!-- results:end -->\n"


def test_a_recorded_run_fills_the_results_block(ws):
    write_manifest(ws / "cassettes", syn.manifest({"triage": ("v1",)}))
    case = syn.case_record("tri-001", (), checks=syn.TRIAGE_CHECKS)
    write_results(syn.function_results("triage", "v1", [case]), ws / "results")
    rendered = render.render_text(block("results"), sources(ws))
    assert "| All checks | 100.0% |" in rendered
    assert rendered == render.replace_blocks(
        block("results"), {"results": render.render_block(ws / "results", ws / "cassettes")}
    )
