"""How the replay evaluation shows in the Allure report.

Each per-case test is labelled epic = function (`rag`, `triage`), feature =
layer and story = the case's category. A case is listed under each layer it
fails, or under every layer it was checked on when it passes, so the status
shown under a layer is that layer's: a case that fails only the reference
facts is not shown as failing retrieval. The parameters are the case id and
the prompt version. The title names the case and its question or ticket, the
description says what the case tests, and the attachments hold the input,
the retrieved documents (RAG), each answer, the failed checks with their
details, the judge's verdict and reasons (RAG) and the whole record.

Attachments are text or JSON built from the replayed record. A record holds
no request headers; the cassette key of each call (a sha256 of the request)
is left out too, so nothing in the report looks like a key.

With `--alluredir`, the suite also writes `categories.json` (a regression
against the baseline, a known failure that now passes, a known failure, and
pending: no recorded run or no baseline) and `environment.properties` (the
models, the recording dates, repeats, judge_repeats, the call count and the
prompt versions, from the run manifest).
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

import allure

from app.retrieval import default_index
from llmeval.baseline import PENDING_BASELINE, explain
from llmeval.cassettes import PENDING_RECORDED_RUN, CassetteError, RunManifest, load_manifest
from llmeval.datasets import RagCase, TriageCase
from llmeval.results import LAYERS, CaseRecord, JudgeRecord

TITLE_WORDS = 12
# The call key of every call in a record: the system call and the judge's.
CALL_KEYS = {"runs": {"__all__": {"call": {"key"}, "judge": {"call": {"key"}}}}}


def short(text: str, words: int = TITLE_WORDS) -> str:
    """The first `words` words of `text`, with an ellipsis when cut."""
    parts = text.split()
    return " ".join(parts[:words]) + (" …" if len(parts) > words else "")


def _attach_text(body: str, name: str) -> None:
    allure.attach(body, name=name, attachment_type=allure.attachment_type.TEXT)


def _facts(case: RagCase) -> str:
    return "; ".join(" or ".join(alternatives) for alternatives in case.fact_alternatives)


def rag_description(case: RagCase) -> str:
    """What one RAG case tests, in a few short sentences."""
    docs = ", ".join(case.expected_docs)
    if case.category == "safety":
        text = f"A safety case ({case.attack_type}). The attack: {case.attack} "
        text += f"A safe answer: {case.expected_behaviour}"
        if case.expected == "answer":
            text += f" It still states {_facts(case)} from {docs}."
    elif case.category == "unanswerable":
        text = (
            "No document answers this question. The assistant must say it does not know "
            "and invent nothing."
        )
    elif case.category == "multi_doc":
        text = f"The answer needs {len(case.expected_docs)} documents ({docs}). "
        text += f"It must state {_facts(case)}."
    else:
        text = f"One document ({docs}) answers this question. The answer must state {_facts(case)}."
    if case.forbidden and case.category != "safety":
        text += f" It must not say: {'; '.join(case.forbidden)}."
    if case.note:
        text += f" Note: {case.note}"
    return text


def triage_description(case: TriageCase) -> str:
    """What one triage case expects, from the guideline."""
    order = case.order_id or "none"
    text = (
        f"A support ticket. By the guideline: category {case.category}, "
        f"priority {case.priority} (rule {case.priority_rule}), order id {order}."
    )
    if case.note:
        text += f" Note: {case.note}"
    return text


def show_case(function: str, case: RagCase | TriageCase, version: str) -> None:
    """What is known before the replay: labels, title, description and the input."""
    question = case.question if isinstance(case, RagCase) else case.text
    allure.dynamic.epic(function)
    allure.dynamic.story(str(case.category))
    # Replaces the case object pytest passes as a parameter with its id.
    allure.dynamic.parameter("case", case.id)
    allure.dynamic.title(f"{case.id} {version}: {short(question)}")
    if isinstance(case, RagCase):
        allure.dynamic.description(rag_description(case))
        _attach_text(question, "question")
    else:
        allure.dynamic.description(triage_description(case))
        _attach_text(question, "ticket")


def shown_layers(record: CaseRecord) -> list[str]:
    """The layers a case is listed under: those it fails, else every one it was checked on."""
    checked = {check.layer for run in record.runs for check in run.checks}
    failed = {check.layer for run in record.runs for check in run.checks if not check.passed}
    chosen = failed or checked
    return [layer for layer in LAYERS if layer in chosen] + sorted(chosen - set(LAYERS))


def judge_text(repeat: int, judge: JudgeRecord) -> str:
    """The judge's verdict on one answer and its reasons, or why it is invalid."""
    if judge.scores is None:
        return (
            f"repeat {repeat}: invalid verdict ({judge.error}): {judge.detail}\n"
            f"reply: {judge.raw}"
        )
    scores = ", ".join(f"{criterion} {score}" for criterion, score in judge.scores.items())
    rule = "passes" if judge.rule_pass else "fails"
    said = "pass" if judge.judge_pass else "fail"
    return (
        f"repeat {repeat}: {scores}; the rubric rule {rule}; the judge said {said}\n"
        f"reasons: {judge.reasons}"
    )


def _retrieved(record: CaseRecord) -> str:
    index = default_index()
    expected = set(record.expected.get("expected_docs") or ())
    lines = []
    for doc_id in record.runs[0].retrieved or []:
        mark = " (expected)" if doc_id in expected else ""
        lines.append(f"{doc_id}: {index.get(doc_id).title}{mark}")
    return "\n".join(lines) or "nothing retrieved"


def show_record(function: str, record: CaseRecord) -> None:
    """What the replay produced: the layers, the answers, failures, verdicts, the record."""
    for layer in shown_layers(record):
        allure.dynamic.feature(layer)
    if function == "rag":
        _attach_text(_retrieved(record), "retrieved documents")
    output = "answer" if function == "rag" else "reply"
    for run in record.runs:
        _attach_text(run.output, f"{output}, repeat {run.repeat}")
    verdicts = [judge_text(run.repeat, run.judge) for run in record.runs if run.judge]
    if verdicts:
        _attach_text("\n\n".join(verdicts), "judge verdict")
    if any(not run.passed for run in record.runs):
        _attach_text(explain(record), "failed checks")
    allure.attach(
        record.model_dump_json(indent=2, exclude=CALL_KEYS),
        name="record",
        attachment_type=allure.attachment_type.JSON,
    )


# --- categories and environment ------------------------------------------------

# How a failure or skip starts (see `tests.eval.support.apply_verdict` and the
# skips of the suite); the categories below match on these words.
REGRESSION = "regression"
STRICT_XPASS = "[XPASS(strict)]"
KNOWN_FAILURE = "known failure in the baseline"
NOT_RECORDED = "is not in the recorded run"
NOTHING_TO_COMPARE = "nothing to compare"


def categories() -> list[dict[str, object]]:
    """Allure categories: a result falls in one when its status is listed and
    the regex matches its whole message (Allure uses Java regex with DOTALL)."""
    pending = "|".join(
        re.escape(reason)
        for reason in (PENDING_RECORDED_RUN, PENDING_BASELINE, NOT_RECORDED, NOTHING_TO_COMPARE)
    )
    return [
        {
            "name": "Regression against the baseline",
            "matchedStatuses": ["failed"],
            "messageRegex": f"Failed: {REGRESSION}: .*",
        },
        {
            "name": "Known failure that now passes: update the baseline",
            "matchedStatuses": ["failed"],
            "messageRegex": f"Failed: {re.escape(STRICT_XPASS)} .*",
        },
        {
            "name": "Known failure in the baseline",
            "matchedStatuses": ["skipped"],
            "messageRegex": f"XFAIL {KNOWN_FAILURE}: .*",
        },
        {
            "name": "Pending: no recorded run or no baseline",
            "matchedStatuses": ["skipped"],
            "messageRegex": f"Skipped: .*({pending}).*",
        },
    ]


def _utc(moment: datetime) -> str:
    return f"{moment.astimezone(UTC):%Y-%m-%d %H:%M} UTC"


def environment(manifest: RunManifest | None) -> str:
    """`environment.properties`: what the recorded run used, from its manifest."""
    if manifest is None:
        return f"recording={PENDING_RECORDED_RUN}\n"
    versions = "; ".join(
        f"{function} {', '.join(chosen)}" for function, chosen in manifest.prompt_versions.items()
    )
    lines = {
        "system.model": manifest.models["system"],
        "judge.model": manifest.models["judge"],
        "recorded.from": _utc(manifest.recorded_from),
        "recorded.to": _utc(manifest.recorded_to),
        "repeats": manifest.repeats,
        "judge_repeats": manifest.judge_repeats,
        "calls": manifest.recorded_calls,
        "prompt.versions": versions,
    }
    return "".join(f"{key}={value}\n" for key, value in lines.items())


def write_report_files(directory: Path, cassettes: Path) -> None:
    """Write `categories.json` and `environment.properties` into the Allure results."""
    directory.mkdir(parents=True, exist_ok=True)
    text = json.dumps(categories(), indent=2) + "\n"
    (directory / "categories.json").write_text(text, encoding="utf-8")
    try:
        props = environment(load_manifest(cassettes))
    except CassetteError as exc:
        props = f"recording=manifest error: {str(exc).splitlines()[0]}\n"
    (directory / "environment.properties").write_text(props, encoding="utf-8")
