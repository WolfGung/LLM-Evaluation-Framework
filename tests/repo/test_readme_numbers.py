"""The generated blocks of README.md and docs/ are the ones tools/render.py renders.

Every number in README.md and docs/ comes from results/, the run manifest,
the gate tolerances or the datasets, through a block between
`<!-- NAME:start -->` and `<!-- NAME:end -->`. These tests fail when a block
differs from the rendered one (a new recording, a changed check or a hand
edit), and when a number from the results is written outside a block. The
fix is make readme (python -m tools.render --write), then commit the files.

A fact that is not in results/, such as one from an earlier recording, goes
in a `history` block that names its commit.

No model is called and nothing is written.
"""

import pytest

from tools.render import (
    END_MARKER,
    README,
    START_MARKER,
    Sources,
    bare_results,
    block_names,
    default_files,
    history_without_commit,
    outside_blocks,
    render_text,
)

FIX = "run make readme and commit the files"
FILES = default_files()
# The generated blocks each page must hold, in order: a page that loses one
# would quote its results nowhere, and no other test would notice.
EXPECTED_BLOCKS = {
    "README.md": ["results", "pairwise", "agreement"],
    "docs/01-what-is-evaluated.md": ["scope"],
    "docs/02-eval-strategy.md": ["cost", "gate"],
    "docs/03-judge-validation.md": ["agreement", "judge", "pairwise", "history"],
    "docs/04-safety-cases.md": ["safety"],
    "docs/05-limitations.md": ["scope"],
    "docs/06-tools-and-versions.md": [],
}


def name(path) -> str:
    return path.relative_to(README.parent).as_posix()


def test_the_readme_has_one_results_block():
    text = README.read_text(encoding="utf-8")
    assert text.count(START_MARKER) == 1, f"README.md needs one {START_MARKER} line"
    assert text.count(END_MARKER) == 1, f"README.md needs one {END_MARKER} line"
    assert block_names(text).count("results") == 1


def test_the_readme_shows_the_pairwise_comparison_and_the_judge_agreement():
    names = block_names(README.read_text(encoding="utf-8"))
    assert names.count("pairwise") == 1, "README.md needs one pairwise block"
    assert names.count("agreement") == 1, "README.md needs one agreement block"


@pytest.mark.parametrize("path", FILES, ids=name)
def test_the_generated_blocks_are_rendered_from_results(path):
    text = path.read_text(encoding="utf-8")
    assert render_text(text, Sources()) == text, (
        f"the generated blocks of {name(path)} differ from results/: {FIX}"
    )


@pytest.mark.parametrize("path", FILES, ids=name)
def test_no_number_from_the_results_is_written_outside_a_block(path):
    written = bare_results(outside_blocks(path.read_text(encoding="utf-8")))
    assert not written, (
        f"{name(path)} has numbers from the results outside its blocks: {', '.join(written)}; "
        "numbers come from results/ through tools/render.py, and a historical fact goes "
        "in a history block with its commit"
    )


@pytest.mark.parametrize("path", FILES, ids=name)
def test_every_history_quote_names_its_commit(path):
    unnamed = history_without_commit(path.read_text(encoding="utf-8"))
    assert not unnamed, f"{name(path)}: a history block names no commit: {unnamed[0][:80]}"


def test_each_page_holds_its_generated_blocks():
    assert sorted(name(path) for path in FILES) == sorted(EXPECTED_BLOCKS)
    for path in FILES:
        assert block_names(path.read_text(encoding="utf-8")) == EXPECTED_BLOCKS[name(path)], (
            f"{name(path)} lost or gained a generated block"
        )
