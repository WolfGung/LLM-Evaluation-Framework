"""The published page: the README results table and a link to the Allure report.

    python -m tools.site                  # writes site/index.html
    python -m tools.site --out page.html  # somewhere else

The CI workflow builds this page on a push to main, puts the Allure report
of the same run next to it under `report/`, and publishes both on GitHub
Pages. The table is the one in README.md (`tools.render.table`), read from
`results/` and `cassettes/manifest.json` only, so the same files give the
same page. Without a recorded run the page says `pending first recorded run`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from html import escape
from pathlib import Path

from llmeval.cassettes import PENDING_RECORDED_RUN
from tools.render import CASSETTES, RESULTS, ROOT, RenderError, Table, load, table

SITE = ROOT / "site" / "index.html"
# Where the Allure report sits next to the page.
REPORT = "report/"
TITLE = "LLM-Evaluation-Framework"
SUMMARY = "Evaluate LLM features the way a test engineer evaluates ordinary code."

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


def _table_html(main: Table) -> list[str]:
    header = "".join(f'<th scope="col">{escape(cell)}</th>' for cell in main.header)
    rows = [
        f'<tr><th scope="row">{escape(row[0])}</th>'
        + "".join(f"<td>{escape(cell)}</td>" for cell in row[1:])
        + "</tr>"
        for row in main.rows
    ]
    return [
        '<div class="scroll">',
        "<table>",
        "<caption>Each prompt version of both functions, layer by layer</caption>",
        f"<thead><tr>{header}</tr></thead>",
        "<tbody>",
        *rows,
        "</tbody>",
        "</table>",
        "</div>",
        f"<p>{escape(main.line)}</p>",
    ]


def build(results_dir: Path = RESULTS, cassettes_dir: Path = CASSETTES) -> str:
    """The page's HTML, from the results files and the manifest."""
    loaded = load(results_dir, cassettes_dir)
    body = [f"<p>{PENDING_RECORDED_RUN}</p>"] if loaded is None else _table_html(table(*loaded))
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
        f"<p>{SUMMARY}</p>",
        "<h2>Results of the recorded run</h2>",
        *body,
        "<h2>The full report</h2>",
        f'<p><a href="{REPORT}">Allure report</a>: every case by function, layer and category, '
        "with the answers, the failed checks, the judge's verdicts and the pairwise "
        "comparison.</p>",
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
