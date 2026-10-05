"""Synthetic judged RAG results and a synthetic recording for the label tests.

Synthetic data: every case, answer, cassette key and judge grade built here
is made up for the test that asks for it, never a model output. Answers are
recorded through the real client with an httpx MockTransport. Nothing is
written outside pytest's `tmp_path`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydantic import SecretStr

from app import assistant
from llmeval.cassettes import CassetteStore
from llmeval.client import ModelClient
from llmeval.config import Config, Mode, Settings, load_models_config
from llmeval.labels import Sample, SampleItem
from llmeval.results import CaseRecord, FunctionResults, RunRecord
from tests.unit import synthetic_results as syn
from tests.unit.synthetic_judge import NoWait, SyntheticTransport


def key_of(case_id: str, version: str, repeat: int) -> str:
    return f"{case_id}-{version}-{repeat}".ljust(64, "0")


def rag_case(case_id: str, version: str, category: str, *grades: str | None) -> CaseRecord:
    """A RAG case with one run per grade ("pass", "fail", "invalid" or None:
    not graded), each run with its own synthetic answer key."""
    runs = []
    for repeat, grade in enumerate(grades or ("pass",)):
        runs.append(
            RunRecord(
                repeat=repeat,
                output=f"Synthetic answer {case_id} {version} {repeat}.",
                call=syn.CALL.model_copy(update={"key": key_of(case_id, version, repeat)}),
                checks=[],
                judge=None if grade is None else syn.judge_record(grade),
            )
        )
    return CaseRecord(
        id=case_id, category=category, input="Synthetic question?", expected={}, runs=runs
    )


def population(version: str, counts: dict[str, tuple[int, int]], start: int = 1) -> FunctionResults:
    """Cases per category: (judge passes, judge failures), numbered from `start`."""
    cases = []
    number = start
    for category, (passes, failures) in counts.items():
        for grade in ["pass"] * passes + ["fail"] * failures:
            cases.append(rag_case(f"rag-{number:03d}", version, category, grade))
            number += 1
    return syn.function_results("rag", version, cases)


def two_versions() -> list[FunctionResults]:
    counts = {"answerable": (24, 1), "multi_doc": (6, 1), "unanswerable": (6, 2)}
    later = {"answerable": (24, 1), "multi_doc": (6, 1), "unanswerable": (7, 1)}
    return [population("v1", counts), population("v2", later)]


# --- a synthetic recording for the labelling tool ---------------------------

CONFIG_YAML = """\
system:
  model: synthetic/system:free
  temperature: 0.2
  max_tokens: 200
  structured_output: false
judge:
  model: synthetic/judge:free
  temperature: 0
  seed: 7
  max_tokens: 200
  structured_output: true
repeats: 1
rpm: 18
"""

ROWS = [
    {
        "id": "rag-001",
        "category": "answerable",
        "question": "How many days do I have to return an item?",
        "expected": "answer",
        "required_facts": ["30 days"],
        "expected_docs": ["kb-returns"],
    },
    {
        "id": "rag-029",
        "category": "unanswerable",
        "question": "Do you price match if I find a drill cheaper somewhere else?",
        "expected": "dont_know",
    },
]
# Synthetic answers, by (case, version).
ANSWERS = {
    ("rag-001", "v1"): "Synthetic: you have 30 days [kb-returns].",
    ("rag-001", "v2"): "Synthetic: returns are accepted within 30 days of delivery [kb-returns].",
    ("rag-029", "v2"): "Synthetic: the documents do not say. Please contact Toolshop support.",
}


@dataclass(frozen=True)
class Recording:
    """A workspace with a config, a RAG dataset and recorded synthetic answers."""

    root: Path
    keys: dict[tuple[str, str], str]

    @property
    def config(self) -> Path:
        return self.root / "config.yaml"

    @property
    def datasets(self) -> Path:
        return self.root / "datasets"

    @property
    def cassettes(self) -> Path:
        return self.root / "cassettes"

    def sample(self, *refs: tuple[str, str]) -> Sample:
        """A sample of the recorded answers `refs` (case, version), in that order."""
        categories = {row["id"]: row["category"] for row in ROWS}
        items = [
            SampleItem(
                case=case,
                version=version,
                repeat=0,
                answer_key=self.keys[(case, version)],
                category=categories[case],
            )
            for case, version in refs or tuple(self.keys)
        ]
        return Sample(seed=1, rule="Synthetic rule.", size=len(items), items=items)


def record_answers(root: Path) -> Recording:
    """Record `ANSWERS` through the real assistant and client with a mock
    transport, into `root` (a `tmp_path`)."""
    (root / "config.yaml").write_text(CONFIG_YAML, encoding="utf-8")
    datasets = root / "datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    (datasets / "rag.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in ROWS), encoding="utf-8"
    )
    models = load_models_config(root / "config.yaml")
    config = Config(models=models, settings=Settings(api_key=SecretStr("synthetic-key-123")))
    questions = {row["id"]: row["question"] for row in ROWS}
    pending = list(ANSWERS.values())
    transport = SyntheticTransport(lambda body: pending.pop(0))
    keys = {}
    with ModelClient(
        Mode.RECORD,
        CassetteStore(root / "cassettes"),
        config,
        httpx.MockTransport(transport),
        limiter=NoWait(),
    ) as client:
        for case, version in ANSWERS:
            answer = assistant.answer(
                client, models.system, questions[case], version, repeat=0, case=case
            )
            keys[(case, version)] = answer.call.key
    return Recording(root=root, keys=keys)
