"""README.md as a page: the first screen, the badges, the links and the closing sections.

The numbers in the README are checked by test_readme_numbers.py; this module
checks the rest of the page the brief asks for. Links to other websites are
not fetched (the tests reach no network); every relative link must point to
a file or directory in the repository, in README.md and in docs/.
"""

import re
from pathlib import Path

import pytest

from tools import site
from tools.render import default_files

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"
REPOSITORY = "https://github.com/WolfGung/LLM-Evaluation-Framework"
PAGES = "https://wolfgung.github.io/LLM-Evaluation-Framework/"
CLIENT_POINTS = (
    "Catch regressions when you change a prompt or a model.",
    "Test a chatbot against prompt injection and data leaks.",
    "Know when an LLM judge can be trusted.",
    "Keep evaluation in CI without paying for every run.",
)
PORTFOLIO = (
    "Toolshop-Test-Automation-Framework",
    "Marketplace-Test-Automation-Framework",
    "Web-Scraping-Automation-Framework",
    "Test-Suite-Rescue",
    "API-Test-Generator",
    "Accessibility-Test-Automation-Framework",
)
HIRE_ME = (
    "I take short, well-defined jobs: a test automation framework from scratch, an API test "
    "suite for an existing backend, end-to-end tests for a critical flow, fixing flaky tests "
    "and reducing run time, setting up CI for existing tests, scrapers and data pipelines. "
    "Profile on Guru: [https://www.guru.com/freelancers/pavel-zhukov-atum]"
    "(https://www.guru.com/freelancers/pavel-zhukov-atum). Time zone: Central European Time "
    "(CET/CEST), so my working hours overlap with Central European business hours. I work in "
    "writing."
)
LINK = re.compile(r"\]\(([^)\s]+)\)")


def lines() -> list[str]:
    return README.read_text(encoding="utf-8").splitlines()


def section(title: str) -> list[str]:
    text = lines()
    start = text.index(f"## {title}") + 1
    end = next((i for i in range(start, len(text)) if text[i].startswith("## ")), len(text))
    return [line for line in text[start:end] if line.strip()]


def test_the_first_screen_is_title_sentence_badges_then_the_main_table():
    text = lines()
    assert text[0] == "# LLM Evaluation Framework"
    assert text[1] == "" and text[2].endswith(".") and text[3] == ""
    badges = text[4:8]
    assert all(line.startswith("[![") for line in badges)
    assert text[8] == "" and text[9] == "<!-- results:start -->"
    assert text.index("## What this shows") > text.index("<!-- results:end -->")


def test_the_badges_are_ci_the_live_report_python_and_the_licence():
    badges = [line for line in lines() if line.startswith("[![")]
    assert badges == [
        f"[![CI]({REPOSITORY}/actions/workflows/ci.yml/badge.svg?branch=main&event=push)]"
        f"({REPOSITORY}/actions/workflows/ci.yml)",
        f"[![live report](https://img.shields.io/badge/live%20report-GitHub%20Pages-brightgreen)]"
        f"({PAGES})",
        "[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](pyproject.toml)",
        "[![license: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)",
    ]
    # The CI badge reflects ci.yml only, and only its push runs on main: live.yml is
    # started by hand, and a run of ci.yml by another event never colours the badge.
    assert "live.yml/badge" not in README.read_text(encoding="utf-8")


def test_the_published_page_opens_with_the_readme_sentence():
    assert lines()[2] == site.SUMMARY


def test_the_readme_legend_follows_the_main_table_and_is_the_page_legend():
    text = lines()
    after = text.index("<!-- results:end -->")
    assert text[after + 1] == "" and text[after + 3] == ""
    assert text[after + 2] == site.LEGEND


def test_the_findings_follow_the_legend_right_after_the_first_screen():
    text = lines()
    legend = text.index(site.LEGEND)
    assert text[legend + 1 : legend + 5] == ["", "## Findings", "", "<!-- findings:start -->"]
    assert text.index("<!-- findings:end -->") < text.index("## What this shows")
    (pairwise,) = section("Findings")[1:-1]  # one bullet, inside the markers
    assert pairwise.startswith("- **")
    assert "compared pairs" in pairwise and "swapped places" in pairwise


def test_what_this_shows_names_the_four_client_tasks():
    bullets = section("What this shows")
    assert len(bullets) == len(CLIENT_POINTS)
    for bullet, point in zip(bullets, CLIENT_POINTS, strict=True):
        assert bullet.startswith(f"- **{point}**")


def test_related_work_lists_the_six_other_portfolio_repositories():
    entries = section("Related work")
    assert entries[0] == "Six more repositories from the same portfolio:"
    names = [
        re.match(r"- \*\*\[([^\]]+)\]\(https://github\.com/WolfGung/\1\)\*\* — ", e)
        for e in entries[1:]
    ]
    assert [match.group(1) if match else None for match in names] == list(PORTFOLIO)


def test_hire_me_is_the_portfolio_text_and_closes_the_page():
    assert section("Hire me") == [HIRE_ME]
    assert lines()[-1] == HIRE_ME


@pytest.mark.parametrize("path", default_files(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_every_relative_link_points_into_the_repository(path):
    broken = []
    for target in LINK.findall(path.read_text(encoding="utf-8")):
        if re.match(r"[a-z]+:", target) or target.startswith("#"):
            continue
        local = (path.parent / target.split("#", 1)[0]).resolve()
        if not local.exists() or ROOT not in (local, *local.parents):
            broken.append(target)
    assert not broken, f"{path.name}: links to nothing in the repository: {', '.join(broken)}"
