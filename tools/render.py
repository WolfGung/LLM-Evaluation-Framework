"""The README results block, rendered from results/ and the run manifest.

    python -m tools.render           # print the block
    python -m tools.render --check   # exit 1 when README.md holds another block
    python -m tools.render --write   # write the block into README.md

The block sits between `<!-- results:start -->` and `<!-- results:end -->`
in README.md. It reads only the results files in `results/` and
`cassettes/manifest.json`, so every number in it comes from the recorded run,
and the same files always give the same block. Without a manifest there is
no recorded run, and the block says `pending first recorded run`. A manifest
without its results files is an error (run make eval), never "pending".

The main table has one column per function and prompt version:

- All checks: the share of runs (every repeat of every case) that pass
  every check.
- Each layer: the share of the runs the layer checks that pass all of its
  checks. Retrieval and reference check only the cases with expected
  documents or facts, and the judge grades the runs `judge_repeats` names
  (the first run of each judged case by default).
- Safety cases with no safety failure: the safety cases whose safety checks
  all passed on every repeat. A safety case that misses a fact or runs too
  long still resisted the attack. The row is left out when no column has a
  safety case.
- Stable cases: the share of repeated cases whose repeats agree on every
  rule-based check (`llmeval.stability`).
- Cost (system + judge calls): the cost the provider reported for the
  system calls and the judge's grades of that version, the total a client
  pays for evaluating it. Free model variants that reported no charge show
  as `$0.00 (free models)`; a call without a reported cost is named, never
  counted as zero. The pairwise judge calls compare two versions, so they
  are in no column, and the line says so.
- Latency p50 / p95 (system calls): of the system calls only.

One line under the table says what a column is, how often each case ran and
which runs a layer counts, what "no safety failure" means, and the
recording: dates, models and calls. Percentages and seconds have one
decimal, and a half rounds up (`percent`, `seconds`); the page built by
`tools.site` uses the same table.
"""

from __future__ import annotations

import argparse
import difflib
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from llmeval.baseline import BaselineError, RunResults, load_run_results
from llmeval.callplan import plural
from llmeval.cassettes import PENDING_RECORDED_RUN, CassetteError, RunManifest, load_manifest
from llmeval.perf import COST_PLACES
from llmeval.pricing import format_usd
from llmeval.results import LAYERS, CaseRecord, FunctionResults

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
RESULTS = ROOT / "results"
CASSETTES = ROOT / "cassettes"

START_MARKER = "<!-- results:start -->"
END_MARKER = "<!-- results:end -->"
# A cell with nothing to show: the layer or measure does not apply.
NONE = "—"
FREE_SUFFIX = ":free"
SAFETY = "safety"  # the safety layer, and the RAG category of the attack cases


class RenderError(ValueError):
    """The block cannot be rendered or placed: broken results, or no markers."""


def _one_decimal(value: Decimal) -> str:
    return str(value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def percent(count: int, total: int) -> str:
    """`count` of `total` as a percentage with one decimal; a half rounds up."""
    if not total:
        return NONE
    return f"{_one_decimal(Decimal(count) * 100 / Decimal(total))}%"


def share(rate: float | None) -> str:
    """A stored rate (a share such as 0.9551) as a percentage with one decimal."""
    if rate is None:
        return NONE
    return f"{_one_decimal(Decimal(str(rate)) * 100)}%"


def seconds(ms: float | None) -> str:
    """Milliseconds as seconds with one decimal; a half rounds up."""
    if ms is None:
        return NONE
    return f"{_one_decimal(Decimal(str(ms)) / 1000)} s"


@dataclass(frozen=True)
class Table:
    """The main table: a header, rows of cells, and the line under it."""

    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    line: str


def load(results_dir: Path, cassettes_dir: Path) -> tuple[RunManifest, RunResults] | None:
    """The manifest and the results of the recorded run, or None without a manifest."""
    try:
        manifest = load_manifest(cassettes_dir)
        if manifest is None:
            return None
        return manifest, load_run_results(results_dir, manifest.prompt_versions)
    except (CassetteError, BaselineError) as exc:
        raise RenderError(str(exc)) from None


def _layers(results: Sequence[FunctionResults]) -> list[str]:
    present = {layer for result in results for layer in result.summary.layers}
    return [layer for layer in LAYERS if layer in present] + sorted(present - set(LAYERS))


def _layer_cell(result: FunctionResults, layer: str) -> str:
    rate = result.summary.layers.get(layer)
    return NONE if rate is None else percent(rate.passed, rate.total)


def _safety_cases(result: FunctionResults) -> list[CaseRecord]:
    return [record for record in result.cases if record.category == SAFETY]


def _no_safety_failure(result: FunctionResults) -> str:
    cases = _safety_cases(result)
    if not cases:
        return NONE
    resisted = sum(
        all(check.passed for run in record.runs for check in run.checks if check.layer == SAFETY)
        for record in cases
    )
    return f"{resisted} of {len(cases)}"


def _stable(result: FunctionResults) -> str:
    stable = result.summary.stability
    return NONE if stable is None else percent(stable.stable, stable.repeated)


def _cost(result: FunctionResults) -> str:
    summary = result.summary
    measured = [summary.performance, *([summary.judge.performance] if summary.judge else [])]
    total = round(sum(perf.cost.total_usd for perf in measured), COST_PLACES)
    unknown = sum(perf.cost.unknown_calls for perf in measured)
    if unknown:
        return f"{format_usd(total)} known, {plural(unknown, 'call')} unknown"
    models = [result.model, *([result.judge_model] if result.judge_model else [])]
    if total == 0 and all(model.endswith(FREE_SUFFIX) for model in models):
        return "$0.00 (free models)"
    return format_usd(total)


def _latency(result: FunctionResults) -> str:
    latency = result.summary.performance.latency
    return f"{seconds(latency.p50_ms)} / {seconds(latency.p95_ms)}"


def _runs(manifest: RunManifest) -> str:
    times = "once" if manifest.repeats == 1 else f"{manifest.repeats} times"
    if manifest.stability_cases is None:
        return f"Each case ran {times}"
    return f"{plural(len(manifest.stability_cases), 'case')} ran {times}, the others once"


def _graded(manifest: RunManifest) -> str:
    runs = "the first run" if manifest.judge_repeats == "first" else "every run"
    return f" (the judge graded {runs} of each judged case)"


def _dates(manifest: RunManifest) -> str:
    first, last = (t.astimezone(UTC).date() for t in (manifest.recorded_from, manifest.recorded_to))
    if first == last:
        return f"on {first.isoformat()} (UTC)"
    return f"from {first.isoformat()} to {last.isoformat()} (UTC)"


def _line(manifest: RunManifest, run: RunResults) -> str:
    results = run.functions
    graded = any(result.judge_model for result in results)
    models = f"{manifest.models['system']} (system)"
    if graded:
        models += f" and {manifest.models['judge']} (judge)"
    judge = _graded(manifest) if graded else ""
    sentences = [
        "Each column is one prompt version.",
        f"{_runs(manifest)}; a layer counts the runs it checks{judge}.",
    ]
    if any(_safety_cases(result) for result in results):
        sentences.append(
            "A safety case has no safety failure when every safety check passed on every run."
        )
    if run.pairwise:
        sentences.append("The pairwise judge calls are in no column.")
    sentences.append(
        f"Recorded {_dates(manifest)} with {models}, {plural(manifest.recorded_calls, 'call')}."
    )
    return " ".join(sentences)


def table(manifest: RunManifest, run: RunResults) -> Table:
    """The main table of a recorded run (see the module docstring)."""
    results = run.functions
    rows = [
        (
            "All checks",
            *(percent(r.summary.all_checks.passed, r.summary.all_checks.total) for r in results),
        )
    ]
    rows += [
        (f"{layer.capitalize()} layer", *(_layer_cell(r, layer) for r in results))
        for layer in _layers(results)
    ]
    if any(_safety_cases(r) for r in results):
        rows.append(
            ("Safety cases with no safety failure", *(_no_safety_failure(r) for r in results))
        )
    rows.append(("Stable cases", *(_stable(r) for r in results)))
    rows.append(("Cost (system + judge calls)", *(_cost(r) for r in results)))
    rows.append(("Latency p50 / p95 (system calls)", *(_latency(r) for r in results)))
    header = ("Metric", *(f"{r.function} {r.version}" for r in results))
    return Table(header=header, rows=tuple(rows), line=_line(manifest, run))


def markdown(main: Table) -> str:
    """The table in Markdown, numbers right-aligned, then the line under it."""
    lines = [
        "| " + " | ".join(main.header) + " |",
        "|---|" + "---:|" * (len(main.header) - 1),
        *("| " + " | ".join(row) + " |" for row in main.rows),
    ]
    return "\n".join(lines) + f"\n\n{main.line}"


def render_block(results_dir: Path = RESULTS, cassettes_dir: Path = CASSETTES) -> str:
    """What goes between the markers: the table and its line, or the pending sentence."""
    loaded = load(results_dir, cassettes_dir)
    body = PENDING_RECORDED_RUN if loaded is None else markdown(table(*loaded))
    return f"\n{body}\n\n"


def replace_block(text: str, block: str) -> str:
    """`text` with `block` between its markers; exactly one ordered pair is required."""
    starts, ends = text.count(START_MARKER), text.count(END_MARKER)
    start, end = text.find(START_MARKER), text.find(END_MARKER)
    if starts != 1 or ends != 1 or end < start:
        raise RenderError(f"needs one {START_MARKER} line, then one {END_MARKER} line")
    head = text[: start + len(START_MARKER)]
    return f"{head}\n{block}{text[end:]}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.render",
        description="Render the README results block from results/ and the run manifest.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", action="store_true", help="exit 1 when the README holds another block"
    )
    mode.add_argument("--write", action="store_true", help="write the block into the README")
    parser.add_argument("--readme", type=Path, default=README, help="the README file")
    parser.add_argument("--results-dir", type=Path, default=RESULTS, help="the results files")
    parser.add_argument("--cassettes-dir", type=Path, default=CASSETTES, help="the manifest")
    args = parser.parse_args(argv)
    try:
        block = render_block(args.results_dir, args.cassettes_dir)
        if not (args.check or args.write):
            print(block, end="")
            return 0
        text = args.readme.read_text(encoding="utf-8")
        updated = replace_block(text, block)
    except RenderError as exc:
        print(f"{args.readme.name}: {str(exc).splitlines()[0]}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"{args.readme.name}: {exc.strerror}", file=sys.stderr)
        return 1
    name = args.readme.name
    if args.check:
        if updated == text:
            print(f"{name}: the results block matches results/")
            return 0
        print(
            f"{name}: the results block differs from results/: run python -m tools.render --write",
            file=sys.stderr,
        )
        diff = difflib.unified_diff(
            text.splitlines(), updated.splitlines(), "README now", "rendered", lineterm=""
        )
        print("\n".join(diff), file=sys.stderr)
        return 1
    if updated != text:
        args.readme.write_text(updated, encoding="utf-8")
        print(f"{name}: results block written")
    else:
        print(f"{name}: the results block is up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
