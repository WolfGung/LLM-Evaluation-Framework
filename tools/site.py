"""The published page: the README's generated blocks and a link to the Allure report.

    python -m tools.site                  # writes site/index.html
    python -m tools.site --out page.html  # somewhere else

The CI workflow builds this page on a push to main, puts the Allure report
of the same run next to it under `report/`, and publishes both on GitHub
Pages. The page shows the blocks the README shows, from the same parts
(`tools.render` and `tools.sections`): the main table, the pairwise
comparison and the judge's agreement with the owner's labels. They are read
from `results/` and `cassettes/manifest.json` only, so the same files give
the same page. Without a recorded run each section says `pending first
recorded run`.

The page also says how to read the report: Allure's Behaviors tab groups the
per-case tests by layer, so its counts are tests, not layer pass rates.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from html import escape
from pathlib import Path

from llmeval.cassettes import PENDING_RECORDED_RUN
from tools import sections
from tools.formatting import Part, Table
from tools.render import CASSETTES, RESULTS, ROOT, RenderError, Sources, recorded, table

SITE = ROOT / "site" / "index.html"
# Where the Allure report sits next to the page.
REPORT = "report/"
TITLE = "LLM Evaluation Framework"
# The README's one sentence (a repository test keeps them the same).
SUMMARY = (
    "Layered checks for two LLM features, a support assistant and a ticket triage, that show "
    "in CI what a prompt or model change made better or worse, replayed from real recorded "
    "calls so every run is free."
)
MAIN_CAPTION = "Each prompt version of both functions, layer by layer"
BEHAVIORS = (
    "Allure's Behaviors tab groups the per-case tests by function and layer, and lists a "
    "passing case under every layer it was checked on. Its counts are tests, not layer pass "
    'rates: the layer tests (story "pass rate") give each layer\'s rate, as the table above '
    "does."
)

# Colours on white: #1f2933 text 14.8:1, #1a56db links 6.2:1, #cbd2d9 for lines only.
STYLE = """\
    html { font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
      line-height: 1.5; }
    body { margin: 0; color: #1f2933; background: #ffffff; }
    main { max-width: 64rem; margin: 0 auto; padding: 2rem 1rem 3rem; }
    a { color: #1a56db; }
    :focus-visible { outline: 3px solid #1a56db; outline-offset: 2px; }
    .scroll { overflow-x: auto; }
    table { border-collapse: collapse; margin: 0 0 1rem; }
    caption { padding: 0 0 0.5rem; font-weight: 700; text-align: left; }
    th, td { padding: 0.4rem 0.7rem; border: 1px solid #cbd2d9; }
    th { text-align: left; }
    thead th { background: #f5f7fa; }
    td { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }"""


def _table_html(main: Table, caption: str) -> list[str]:
    header = "".join(f'<th scope="col">{escape(cell)}</th>' for cell in main.header)
    rows = [
        f'<tr><th scope="row">{escape(row[0])}</th>'
        + "".join(f"<td>{escape(cell)}</td>" for cell in row[1:])
        + "</tr>"
        for row in main.rows
    ]
    lines = [
        '<div class="scroll">',
        "<table>",
        f"<caption>{escape(caption)}</caption>",
        f"<thead><tr>{header}</tr></thead>",
        "<tbody>",
        *rows,
        "</tbody>",
        "</table>",
        "</div>",
    ]
    if main.line:
        lines.append(f"<p>{escape(main.line)}</p>")
    return lines


def _parts_html(parts: Sequence[Part]) -> list[str]:
    lines: list[str] = []
    for part in parts:
        if isinstance(part, Table):
            lines += _table_html(part, part.caption)
        elif isinstance(part, str):
            lines.append(f"<p>{escape(part)}</p>")
        else:
            level = part.level + 1  # the page's own h1 and h2 come first
            lines.append(f"<h{level}>{escape(part.text)}</h{level}>")
    return lines


def build(results_dir: Path = RESULTS, cassettes_dir: Path = CASSETTES) -> str:
    """The page's HTML, from the results files and the manifest."""
    run = recorded(Sources(results_dir=results_dir, cassettes_dir=cassettes_dir))
    if run is None:
        main = pairwise = agreement = [f"<p>{PENDING_RECORDED_RUN}</p>"]
    else:
        main = _table_html(table(run.manifest, run.run), MAIN_CAPTION)
        pairwise = _parts_html(sections.pairwise(run))
        agreement = _parts_html(sections.agreement(run))
    lines = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{TITLE} results</title>",
        f"<style>\n{STYLE}\n</style>",
        "</head>",
        "<body>",
        "<main>",
        f"<h1>{TITLE}</h1>",
        f"<p>{escape(SUMMARY)}</p>",
        "<h2>Results of the recorded run</h2>",
        *main,
        "<h2>The two RAG prompt versions, compared by the judge</h2>",
        *pairwise,
        "<h2>Can the judge be trusted?</h2>",
        *agreement,
        "<h2>The full report</h2>",
        f'<p><a href="{REPORT}">Allure report</a>: every case by function, layer and category, '
        "with the answers, the failed checks, the judge's verdicts and the pairwise "
        "comparison.</p>",
        f"<p>{escape(BEHAVIORS)}</p>",
        "</main>",
        "</body>",
        "</html>",
    ]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.site", description="Build the published page from results/."
    )
    parser.add_argument("--out", type=Path, default=SITE, help=f"the page (default: {SITE})")
    parser.add_argument("--results-dir", type=Path, default=RESULTS, help="the results files")
    parser.add_argument("--cassettes-dir", type=Path, default=CASSETTES, help="the manifest")
    args = parser.parse_args(argv)
    try:
        html = build(args.results_dir, args.cassettes_dir)
    except RenderError as exc:
        print(str(exc).splitlines()[0], file=sys.stderr)
        return 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"{args.out} written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
