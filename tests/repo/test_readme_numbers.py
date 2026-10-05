"""The README results block is the one tools/render.py renders from results/.

Every number in the README comes from results/ and the run manifest. This
test fails when the block between `<!-- results:start -->` and
`<!-- results:end -->` differs from the rendered one: a new recording, a
changed check or a hand edit all show up here. The fix is make readme
(python -m tools.render --write), then commit README.md.

No model is called and nothing is written.
"""

import re

from tools.render import END_MARKER, README, START_MARKER, render_block, replace_block

FIX = "run make readme and commit README.md"
PERCENTAGE = re.compile(r"\d+(?:\.\d+)?\s?%")


def readme() -> str:
    return README.read_text(encoding="utf-8")


def test_the_readme_has_one_results_block():
    text = readme()
    assert text.count(START_MARKER) == 1, f"README.md needs one {START_MARKER} line"
    assert text.count(END_MARKER) == 1, f"README.md needs one {END_MARKER} line"
    assert text.index(START_MARKER) < text.index(END_MARKER)


def test_the_readme_results_block_is_rendered_from_results():
    text = readme()
    assert replace_block(text, render_block()) == text, (
        f"the README results block differs from results/: {FIX}"
    )


def test_no_percentage_is_written_outside_the_results_block():
    text = readme()
    start, end = text.find(START_MARKER), text.find(END_MARKER)
    outside = text if start < 0 or end < 0 else text[:start] + text[end:]
    written = PERCENTAGE.findall(outside)
    assert not written, (
        f"README.md has percentages outside the results block: {', '.join(written)}; "
        "numbers come from results/ through tools/render.py"
    )
