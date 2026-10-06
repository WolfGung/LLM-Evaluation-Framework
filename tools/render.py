"""Generated blocks in README.md and docs/, rendered from results/ and the run manifest.

    python -m tools.render             # print each block the files use
    python -m tools.render --check     # exit 1 when a file holds another block
    python -m tools.render --write     # write the blocks into the files
    python -m tools.render --write README.md   # only the files named

The files are README.md and every docs/*.md, unless files are named. A block
sits between `<!-- NAME:start -->` and `<!-- NAME:end -->` and is rendered
from the files in `results/`, `cassettes/manifest.json`, the gate tolerances
and the datasets only, so every number in it comes from the recorded run,
and the same files always give the same block. Without a manifest there is
no recorded run, and every block says `pending first recorded run`. A
manifest without its results files is an error (run make eval), never
"pending". The block names (`BLOCKS`):

- `results`: the main table (below);
- `findings`, `pairwise`, `agreement`, `judge`, `safety`, `cost`, `gate`,
  `scope`: see `tools.sections`;
- `history`: never rendered. It quotes a fact that is not in results/, such
  as one from an earlier recording, and names the commit it comes from; the
  repository test checks that it does.

Numbers from the results may appear only inside blocks: `bare_results` finds
a percentage, a count such as "20 of 38" or "118/156", a number of cases,
runs, calls, pairs, answers or the like, or a dollar amount written anywhere
else, and a repository test runs it over README.md and docs/.

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

A second line reads the p95 latency (`latency_line`), from the tail each
column's summary keeps (`llmeval.perf.TailCall`): the system calls at or
above that column's p95, each next to the fastest other repeat of the same
request. A call that took at least twice as long as a repeat of the same
request (`WAITED`), with that repeat's time first scaled up by the token
ratio when the slow call wrote a longer answer, spent at least half its time
on something the same work did not need: waiting for the provider. The line
counts those calls, gives the median of their latency over their repeat's
and their answers' median length next to the repeats', and says what the
tail is, by what it shows: waiting when such calls are there (most or part
of the tail when not every call is one), the requests themselves when none
is. A free system model (its id ends in `:free`) is named as the shared
free endpoint. Without a tail call that has a repeat there is no line.
"""

from __future__ import annotations

import argparse
import difflib
import re
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path

from llmeval.baseline import BaselineError, RunResults, load_run_results
from llmeval.callplan import plural
from llmeval.cassettes import PENDING_RECORDED_RUN, CassetteError, RunManifest, load_manifest
from llmeval.datasets import RAG_PATH
from llmeval.gate import GATE_CONFIG_PATH
from llmeval.perf import COST_PLACES, TailCall
from llmeval.pricing import format_usd
from llmeval.results import LAYERS, CaseRecord, FunctionResults
from tools import sections
from tools.formatting import (
    NONE,
    Part,
    RenderError,
    Table,
    decimal,
    markdown,
    percent,
    seconds,
)
from tools.formatting import share as share
from tools.sections import Recorded

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DOCS = ROOT / "docs"
RESULTS = ROOT / "results"
CASSETTES = ROOT / "cassettes"
GATE_CONFIG = ROOT / GATE_CONFIG_PATH
RAG_DATASET = ROOT / RAG_PATH

START_MARKER = "<!-- results:start -->"
END_MARKER = "<!-- results:end -->"
# A block that quotes a historical fact with its commit; never rendered.
HISTORY = "history"
FREE_SUFFIX = ":free"
SAFETY = "safety"  # the safety layer, and the RAG category of the attack cases
# A tail call that took at least this many times as long as a repeat of the
# same request spent at least half its time waiting, not working.
WAITED = 2


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
        f"{_runs(manifest)}; each layer's rate is over the runs that layer checks{judge}.",
    ]
    if any(_safety_cases(result) for result in results):
        sentences.append(
            "A safety case has no safety failure when every safety check passed on every run."
        )
    if run.pairwise:
        sentences.append("The pairwise comparison calls are not counted in any column.")
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


def _times_as_long(ratio: float) -> str:
    return decimal(ratio, 0 if ratio >= 10 else 1)


def _answer_tokens(count: float) -> str:
    text = decimal(count, 0)
    return f"{text} answer token" + ("" if text == "1" else "s")


def _waited(call: TailCall) -> bool:
    """At least `WAITED` times as long as the repeat would need for the same
    answer: a longer answer first scales the repeat's time up by the ratio of
    completion tokens, so writing more is never taken for waiting."""
    repeat_ms = call.fastest_repeat_latency_ms or 0.0
    longer = call.completion_tokens / max(call.fastest_repeat_completion_tokens or 0, 1)
    return call.latency_ms >= WAITED * repeat_ms * max(1.0, longer)


def latency_line(manifest: RunManifest, run: RunResults) -> str | None:
    """What the p95 latency of the system calls is made of (see the module
    docstring); None when no tail call has a repeat to compare with."""
    tail = [call for result in run.functions for call in result.summary.performance.tail or []]
    compared = [call for call in tail if call.fastest_repeat_latency_ms is not None]
    if not compared:
        return None
    total = len(compared)
    ran_again = " whose case ran more than once" if total < len(tail) else ""
    calls = (
        f"the system call at or above its column's p95{ran_again}"
        if total == 1
        else f"the {total} system calls at or above their column's p95{ran_again}"
    )
    waited = [call for call in compared if _waited(call)]
    if not waited:
        never = "it did not take" if total == 1 else "none of them took"
        return (
            f"Latency: of {calls}, {never} even twice as long as a repeat of the same request "
            "would need for the same answer, so the slow requests were slow on every try: the "
            "tail comes from the requests themselves, not from waiting at the provider."
        )
    ratio = statistics.median(
        call.latency_ms / (call.fastest_repeat_latency_ms or 1.0) for call in waited
    )
    own = statistics.median(call.completion_tokens for call in waited)
    repeats = statistics.median(call.fastest_repeat_completion_tokens or 0 for call in waited)
    if len(waited) == total:
        subject = calls if total == 1 else f"each of {calls}"
        share = "the tail"
    else:
        subject = f"{len(waited)} of {calls}"
        share = "most of the tail" if 2 * len(waited) > total else "part of the tail"
    where = (
        "the provider's shared free endpoint"
        if manifest.models["system"].endswith(FREE_SUFFIX)
        else "the provider"
    )
    times = f"{_times_as_long(ratio)} times as long"
    if len(waited) == 1:
        measured = (
            f"({times}) while writing {_answer_tokens(own)} against the repeat's "
            f"{decimal(repeats, 0)}"
        )
    else:
        measured = (
            f"(a median of {times}) while writing a median of {_answer_tokens(own)} against "
            f"the repeats' {decimal(repeats, 0)}"
        )
    return (
        f"Latency: {subject} took at least twice as long as a repeat of the same request "
        f"{measured}, so {share} is time spent waiting at {where}, not the time the system "
        "needs to answer."
    )


def results_parts(manifest: RunManifest, run: RunResults) -> list[Part]:
    """The main table, then the latency line when there is one."""
    parts: list[Part] = [table(manifest, run)]
    if line := latency_line(manifest, run):
        parts.append(line)
    return parts


# --- blocks ----------------------------------------------------------------------------

BlockFn = Callable[[Recorded], Sequence[Part]]


def _results_block(recorded: Recorded) -> list[Part]:
    return results_parts(recorded.manifest, recorded.run)


BLOCKS: dict[str, BlockFn] = {
    "results": _results_block,
    "findings": sections.findings,
    "pairwise": sections.pairwise,
    "agreement": sections.agreement,
    "judge": sections.judge,
    "safety": sections.safety,
    "cost": sections.cost,
    "gate": sections.gate,
    "scope": sections.scope,
}


@dataclass(frozen=True)
class Sources:
    """Where the blocks read from: the results, the manifest, the gate
    tolerances and the RAG dataset (for the attack descriptions)."""

    results_dir: Path = RESULTS
    cassettes_dir: Path = CASSETTES
    gate_config: Path = GATE_CONFIG
    rag_dataset: Path = RAG_DATASET


def recorded(sources: Sources) -> Recorded | None:
    """The recorded run, or None without a manifest."""
    loaded = load(sources.results_dir, sources.cassettes_dir)
    if loaded is None:
        return None
    manifest, run = loaded
    return Recorded(
        manifest=manifest,
        run=run,
        results_dir=sources.results_dir,
        gate_config=sources.gate_config,
        rag_dataset=sources.rag_dataset,
    )


def render_bodies(names: Sequence[str], sources: Sources) -> dict[str, str]:
    """What goes between the markers of each named block: the block's parts in
    Markdown, or the pending sentence without a recorded run."""
    if unknown := [name for name in names if name not in BLOCKS]:
        raise RenderError(f"unknown block {unknown[0]!r} (known: {', '.join(BLOCKS)})")
    run = recorded(sources)
    bodies = {}
    for name in dict.fromkeys(names):
        body = PENDING_RECORDED_RUN if run is None else markdown(BLOCKS[name](run))
        bodies[name] = f"\n{body}\n\n"
    return bodies


def render_block(results_dir: Path = RESULTS, cassettes_dir: Path = CASSETTES) -> str:
    """What goes between the markers of the main results block."""
    sources = Sources(results_dir=results_dir, cassettes_dir=cassettes_dir)
    return render_bodies(["results"], sources)["results"]


# --- markers ---------------------------------------------------------------------------

MARKER = re.compile(r"<!-- ([a-z][a-z0-9-]*):(start|end) -->")


@dataclass(frozen=True)
class _Span:
    name: str
    start: int  # just after the start marker
    end: int  # where the end marker begins


def _spans(text: str) -> list[_Span]:
    """The blocks of `text` in order; broken markers raise `RenderError`."""
    spans: list[_Span] = []
    open_name: str | None = None
    open_at = 0
    for match in MARKER.finditer(text):
        name, kind = match.groups()
        if open_name is None:
            if kind == "end":
                raise RenderError(f"{match.group(0)} without its start")
            open_name, open_at = name, match.end()
        elif kind == "start" or name != open_name:
            raise RenderError(f"{match.group(0)} inside the {open_name} block")
        else:
            spans.append(_Span(name, open_at, match.start()))
            open_name = None
    if open_name is not None:
        raise RenderError(f"no <!-- {open_name}:end --> line after its start")
    return spans


def block_names(text: str) -> list[str]:
    """The names of the blocks in `text`, in order (history blocks included)."""
    return [span.name for span in _spans(text)]


def replace_blocks(text: str, bodies: Mapping[str, str]) -> str:
    """`text` with each block's content replaced by its body; history blocks
    stay as written. A block without a body raises `RenderError`."""
    out, last = [], 0
    for span in _spans(text):
        if span.name == HISTORY:
            continue
        if span.name not in bodies:
            raise RenderError(f"unknown block {span.name!r} (known: {', '.join(BLOCKS)})")
        out += [text[last : span.start], "\n", bodies[span.name]]
        last = span.end
    out.append(text[last:])
    return "".join(out)


def render_text(text: str, sources: Sources) -> str:
    """`text` with every block rendered."""
    names = [name for name in block_names(text) if name != HISTORY]
    return replace_blocks(text, render_bodies(names, sources))


def outside_blocks(text: str) -> str:
    """`text` without the content of its blocks (the markers stay)."""
    out, last = [], 0
    for span in _spans(text):
        out.append(text[last : span.start])
        last = span.end
    out.append(text[last:])
    return "".join(out)


def history_bodies(text: str) -> list[str]:
    return [text[span.start : span.end].strip() for span in _spans(text) if span.name == HISTORY]


# A commit named by its hash: "commit 66b4a3a", "commits 66b4a3a and 1fc63b9".
COMMIT = re.compile(r"\bcommits?\b.{0,80}?\b[0-9a-f]{7,40}\b", re.IGNORECASE | re.DOTALL)


def history_without_commit(text: str) -> list[str]:
    """The history quotes of `text` that name no commit."""
    return [body for body in history_bodies(text) if not COMMIT.search(body)]


# --- the guard -------------------------------------------------------------------------

RESULT_NOUNS = (
    "cases?|runs?|calls?|pairs?|answers?|checks?|labels?|verdicts?|failures?|"
    "disagreements?|preferences?|tickets?|questions?|grades?|gradings?"
)
BARE_RESULTS = (
    re.compile(r"\d+(?:\.\d+)?\s?%"),
    re.compile(r"\b\d+\s+of\s+(?:the\s+)?\d+\b"),
    re.compile(r"\b\d+\s?/\s?\d+\b"),
    re.compile(rf"\b\d[\d,]*\s+(?:[a-z-]+\s+)?(?:{RESULT_NOUNS})\b", re.IGNORECASE),
    re.compile(r"\$\s?\d"),
)


def bare_results(text: str) -> list[str]:
    """Numbers in `text` that read like results: percentages, "N of M",
    "N/M", a count of cases, runs, calls and the like, and dollar amounts.
    Run it on the text outside the blocks (`outside_blocks`)."""
    return [match.group(0) for pattern in BARE_RESULTS for match in pattern.finditer(text)]


# --- the command line ------------------------------------------------------------------


def default_files() -> list[Path]:
    """README.md, then every docs/*.md in name order."""
    return [README, *sorted(DOCS.glob("*.md"))]


def _print_blocks(files: Sequence[Path], sources: Sources) -> None:
    names: list[str] = []
    for path in files:
        names += [n for n in block_names(path.read_text(encoding="utf-8")) if n != HISTORY]
    bodies = render_bodies(names, sources)
    for name, body in bodies.items():
        print(f"<!-- {name}:start -->\n{body}<!-- {name}:end -->")


def _show(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path.name


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.render",
        description="Render the generated blocks of README.md and docs/ from results/.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="exit 1 when a file holds another block")
    mode.add_argument("--write", action="store_true", help="write the blocks into the files")
    parser.add_argument(
        "files", nargs="*", type=Path, help="the files (default: README.md and docs/*.md)"
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS, help="the results files")
    parser.add_argument("--cassettes-dir", type=Path, default=CASSETTES, help="the manifest")
    parser.add_argument("--gate-config", type=Path, default=GATE_CONFIG, help="the tolerances")
    parser.add_argument("--rag-dataset", type=Path, default=RAG_DATASET, help="the RAG cases")
    args = parser.parse_args(argv)
    sources = Sources(args.results_dir, args.cassettes_dir, args.gate_config, args.rag_dataset)
    files = args.files or default_files()
    if not (args.check or args.write):
        try:
            _print_blocks(files, sources)
        except (RenderError, OSError) as exc:
            print(str(exc).splitlines()[0], file=sys.stderr)
            return 1
        return 0
    failed = False
    for path in files:
        name = _show(path)
        try:
            text = path.read_text(encoding="utf-8")
            updated = render_text(text, sources)
        except RenderError as exc:
            print(f"{name}: {str(exc).splitlines()[0]}", file=sys.stderr)
            failed = True
            continue
        except OSError as exc:
            print(f"{name}: {exc.strerror}", file=sys.stderr)
            failed = True
            continue
        if args.check:
            if updated == text:
                print(f"{name}: the generated blocks match results/")
                continue
            failed = True
            print(
                f"{name}: the generated blocks differ from results/: "
                "run python -m tools.render --write",
                file=sys.stderr,
            )
            diff = difflib.unified_diff(
                text.splitlines(), updated.splitlines(), f"{name} now", "rendered", lineterm=""
            )
            print("\n".join(diff), file=sys.stderr)
        elif updated != text:
            path.write_text(updated, encoding="utf-8")
            print(f"{name}: blocks written")
        else:
            print(f"{name}: the blocks are up to date")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
