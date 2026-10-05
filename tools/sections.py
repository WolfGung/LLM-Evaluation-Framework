"""The generated blocks other than the main table, rendered from results/.

Each block function takes the recorded run (`Recorded`) and returns parts
(`tools.formatting`): tables, headings and paragraphs. `tools.render` puts
them between their markers in README.md and docs/, and `tools.site` shows
some of them on the published page. Every number comes from the results
files, the run manifest, `results/judge-agreement.json`, the gate tolerances
in `config/gate.yaml` or the datasets; the same files give the same text.

- `pairwise`: the judge's choice between two prompt versions. Outcomes are
  counted over every case; position consistency only over the pairs the
  judge really compared (two different answers, two valid verdicts), so an
  identical or invalid pair never counts as consistent.
- `agreement`: the judge against the owner's labels on the label sample:
  `pending human labels` with the sample until there are labels, then
  percent agreement, Cohen's kappa, the confusion and the disagreements,
  always with the note that the sample oversamples judge failures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from llmeval.agreement import AGREEMENT_FILE, PENDING_HUMAN_LABELS, AgreementReport
from llmeval.baseline import RunResults
from llmeval.callplan import plural
from llmeval.cassettes import RunManifest
from llmeval.results import FunctionResults, PairwiseResults
from tools.formatting import Part, RenderError, Table, decimal, percent


@dataclass(frozen=True)
class Recorded:
    """The recorded run and where the blocks read the rest from."""

    manifest: RunManifest
    run: RunResults
    results_dir: Path
    gate_config: Path
    rag_dataset: Path

    def agreement(self) -> AgreementReport:
        """`results/judge-agreement.json`; missing or broken is an error, never pending."""
        path = self.results_dir / AGREEMENT_FILE
        if not path.is_file():
            raise RenderError(f"results missing: {path}: run make eval")
        try:
            return AgreementReport.model_validate_json(path.read_bytes())
        except ValidationError as exc:
            raise RenderError(f"{path.name}: not a valid agreement report: {exc}") from None

    def graded(self) -> list[FunctionResults]:
        return [result for result in self.run.functions if result.judge_model is not None]


# --- pairwise ---------------------------------------------------------------------------

KINDS = {
    "same_position_a": "it chose the answer shown first in both orders",
    "same_position_b": "it chose the answer shown second in both orders",
    "tie_in_one_order": "it called a tie in one order and chose a side in the other",
}


def _outcome_rows(result: PairwiseResults) -> list[tuple[str, str]]:
    first, second = result.versions
    outcomes = result.summary.outcomes
    rows = [
        (f"{first} preferred in both orders", outcomes.get(first, 0)),
        (f"{second} preferred in both orders", outcomes.get(second, 0)),
        ("A tie in both orders", outcomes.get("tie", 0)),
        ("Inconsistent: the two orders disagree", outcomes.get("inconsistent", 0)),
    ]
    if outcomes.get("identical"):
        rows.append(("Identical answers, not compared", outcomes["identical"]))
    if outcomes.get("invalid"):
        rows.append(("An invalid verdict, not compared", outcomes["invalid"]))
    return [(label, str(count)) for label, count in rows]


def _flips(result: PairwiseResults) -> str:
    summary = result.summary
    flipped, compared = summary.inconsistent.count, summary.inconsistent.total
    if not flipped:
        return (
            f"In none of the {compared} compared pairs did the judge's preference change "
            "when the two answers swapped places."
        )
    kinds = [
        f"{plural(count, 'time')} {KINDS.get(kind, kind)}"
        for kind, count in summary.inconsistent_kinds.items()
        if count
    ]
    listed = ", ".join(kinds[:-1]) + f", and {kinds[-1]}" if len(kinds) > 1 else kinds[0]
    return (
        f"In {flipped} of the {compared} compared pairs, the judge's preference changed when "
        f"the two answers swapped places: {listed}. An inconsistent pair is never settled by "
        "picking one order."
    )


def pairwise_parts(result: PairwiseResults) -> list[Part]:
    """One pairwise comparison: the outcomes, position consistency and the flips."""
    first, second = result.versions
    summary = result.summary
    compared = summary.inconsistent.total
    consistent = compared - summary.inconsistent.count
    if compared:
        consistency = (
            f"Position consistency: {consistent} of {compared} compared pairs "
            f"({percent(consistent, compared)}) got the same verdict in both orders."
        )
        flips = [_flips(result)]
    else:
        consistency = "Position consistency: no pair was compared."
        flips = []
    return [
        f"The judge compared the first answers of {result.function} {first} and "
        f"{result.function} {second} case by case, asked twice with the order of the two "
        "answers swapped.",
        Table(
            header=(f"Outcome over {plural(summary.pairs, 'case')}", "Cases"),
            rows=tuple(_outcome_rows(result)),
            caption=f"The judge's choice between {result.function} {first} and {second}",
        ),
        consistency,
        *flips,
    ]


def pairwise(recorded: Recorded) -> list[Part]:
    """Every pairwise comparison of the run."""
    if not recorded.run.pairwise:
        return ["No pairwise comparison in this run."]
    return [part for result in recorded.run.pairwise for part in pairwise_parts(result)]


# --- agreement --------------------------------------------------------------------------


def _judge_failures(recorded: Recorded) -> int:
    """Valid verdicts that fail the rubric rule, over every graded version."""
    total = 0
    for result in recorded.graded():
        judge = result.summary.judge
        if judge is not None:
            total += judge.rule_pass.total - judge.rule_pass.passed
    return total


def _sample_sentence(report: AgreementReport, failures: int) -> str:
    sample = report.sample
    failed = (
        f"all {sample.judge_fail} answers the judge failed"
        if sample.judge_fail == failures
        else f"{sample.judge_fail} of the {failures} answers the judge failed"
    )
    return (
        f"The owner labels {sample.size} judged answers by hand, blind to the judge's verdict "
        f"(make label): {failed} and {sample.judge_pass} it passed. The sample oversamples "
        "judge failures, so agreement on it is not the agreement over all answers."
    )


def agreement_parts(report: AgreementReport, failures: int) -> list[Part]:
    """The judge's agreement with the owner's labels (see the module docstring)."""
    sample = _sample_sentence(report, failures)
    if report.status == PENDING_HUMAN_LABELS:
        return [PENDING_HUMAN_LABELS, sample]
    cells = report.confusion
    confusion = Table(
        header=("Judge's verdict", "Owner: pass", "Owner: fail"),
        rows=tuple(
            (
                f"Judge: {verdict}",
                str(cells[f"judge_{verdict}"]["human_pass"]),
                str(cells[f"judge_{verdict}"]["human_fail"]),
            )
            for verdict in ("pass", "fail")
        ),
        caption="The judge's verdict by the owner's label, on the labelled sample answers",
    )
    disagreements = cells["judge_pass"]["human_fail"] + cells["judge_fail"]["human_pass"]
    kappa = decimal(report.kappa, 2) if report.kappa is not None else report.kappa_note
    listed = (
        f"{disagreements}, listed in results/judge-agreement.json with the judge's reasons "
        "and the owner's comments"
        if disagreements
        else "0"
    )
    parts: list[Part] = [
        confusion,
        f"Percent agreement: {report.agreed} of {report.labelled} "
        f"({percent(report.agreed, report.labelled)}). Cohen's kappa: {kappa}. "
        f"Disagreements: {listed}.",
    ]
    if report.status == "complete":
        parts.append(f"Labelled: {report.labelled} of {report.sample.size} sample answers.")
    else:
        parts.append(
            f"Labelled so far: {report.labelled} of {report.sample.size} sample answers "
            f"({report.status})."
        )
    if unused := len(report.stale) + len(report.unjudged):
        parts.append(f"Labels not used: {unused}, listed in results/judge-agreement.json.")
    parts.append(sample)
    return parts


def agreement(recorded: Recorded) -> list[Part]:
    if not recorded.graded():
        return ["No answer in this run was graded by the judge."]
    return agreement_parts(recorded.agreement(), _judge_failures(recorded))
