"""`llmeval sample`, `llmeval label` and `llmeval agreement` on synthetic files.

Synthetic data: the results, judge grades, manifests, answers and labels
below are made up for the test. Every file lives in pytest's `tmp_path`;
nothing is written to the repository's `labels/`, `results/` or `cassettes/`.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from llmeval.agreement import SAMPLE_NOTE
from llmeval.cassettes import write_manifest
from llmeval.checks.judge import RUBRIC_PATH
from llmeval.cli import app
from llmeval.labels import (
    LABEL_QUESTION,
    LABELER,
    HumanLabel,
    append_label,
    label_lock,
    load_labels,
    load_sample,
    runs_by_answer,
    write_sample,
)
from llmeval.results import FunctionResults, write_results
from tests.unit import synthetic_results as syn
from tests.unit.synthetic_labels import population, record_answers

runner = CliRunner()
RUBRIC = Path(__file__).resolve().parents[2] / RUBRIC_PATH


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("OPENROUTER_API_KEY", "LLMEVAL_MODE", "MAX_RUN_COST_USD"):
        monkeypatch.delenv(name, raising=False)


def recorded_run(ws, *, rag=None):
    """A manifest and graded RAG results (v1, v2) in the workspace."""
    rag = rag or [
        population("v1", {"answerable": (10, 1), "unanswerable": (4, 2)}),
        population("v2", {"answerable": (10, 1), "unanswerable": (5, 1)}),
    ]
    write_manifest(ws / "cassettes", syn.manifest({"rag": ("v1", "v2")}))
    for result in rag:
        write_results(result, ws / "results")
    write_results(syn.pairwise_results([syn.pair_case("rag-001", "A", "B")]), ws / "results")


def sample_args(ws):
    return [
        "sample",
        "--results-dir",
        str(ws / "results"),
        "--cassettes-dir",
        str(ws / "cassettes"),
        "--sample",
        str(ws / "labels" / "sample.json"),
    ]


def test_sample_without_a_recorded_run_is_pending(tmp_path):
    (tmp_path / "cassettes").mkdir()
    result = runner.invoke(app, sample_args(tmp_path))
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "pending first recorded run"
    assert not (tmp_path / "labels").exists()


def test_sample_writes_the_label_sample_of_the_results(tmp_path):
    recorded_run(tmp_path)
    result = runner.invoke(app, sample_args(tmp_path))
    assert result.exit_code == 0, result.output
    sample = load_sample(tmp_path / "labels" / "sample.json")
    assert sample.size == 30
    assert sample.seed == 2026
    assert result.output.splitlines() == [
        f"wrote {tmp_path / 'labels' / 'sample.json'}: 30 answers of repeat 0, "
        "5 the judge failed and 25 it passed, seed 2026",
    ]
    again = runner.invoke(app, sample_args(tmp_path))
    assert again.exit_code == 0
    assert load_sample(tmp_path / "labels" / "sample.json") == sample


def test_sample_names_missing_results(tmp_path):
    recorded_run(tmp_path)
    (tmp_path / "results" / "rag-v2.json").unlink()
    result = runner.invoke(app, sample_args(tmp_path))
    assert result.exit_code == 1
    assert "results missing: " in result.output and "rag-v2.json" in result.output
    assert not (tmp_path / "labels").exists()


def test_sample_refuses_results_without_judged_answers(tmp_path):
    ungraded = [
        syn.function_results("rag", version, [syn.case_record("rag-001")], graded=False)
        for version in ("v1", "v2")
    ]
    recorded_run(tmp_path, rag=ungraded)
    result = runner.invoke(app, sample_args(tmp_path))
    assert result.exit_code == 1
    assert "no judged answer with a valid verdict in the results" in result.output
    assert not (tmp_path / "labels").exists()


def test_sample_file_is_plain_json(tmp_path):
    recorded_run(tmp_path)
    runner.invoke(app, sample_args(tmp_path))
    data = json.loads((tmp_path / "labels" / "sample.json").read_text(encoding="utf-8"))
    assert data["size"] == len(data["items"]) == 30


# --- llmeval label ------------------------------------------------------------


@pytest.fixture
def recording(tmp_path):
    rec = record_answers(tmp_path)
    write_sample(rec.sample(), tmp_path / "labels" / "sample.json")
    return rec


def label_args(rec, *extra):
    return [
        "label",
        "--sample",
        str(rec.root / "labels" / "sample.json"),
        "--labels",
        str(rec.root / "labels" / "human.jsonl"),
        "--config",
        str(rec.config),
        "--datasets-dir",
        str(rec.datasets),
        "--cassettes-dir",
        str(rec.cassettes),
        "--width",
        "80",
        *extra,
    ]


def test_label_shows_the_question_first_then_each_item_and_saves_labels(recording):
    result = runner.invoke(app, label_args(recording), input="p\nGrounded.\nf\n\nq\n")
    assert result.exit_code == 0, result.output
    out = result.output
    assert " ".join(LABEL_QUESTION.split()) in " ".join(out.split())
    assert out.index("Would you send") < out.index("Item 1 of 3")
    assert "Item 2 of 3" in out and "Item 3 of 3" in out
    assert out.rstrip().endswith(
        "Saved 2 labels this time. Labelled so far: 2 of 3. Run make label to go on."
    )
    labels = load_labels(recording.root / "labels" / "human.jsonl")
    assert [(label.case, label.version, label.label, label.comment) for label in labels] == [
        ("rag-001", "v1", "pass", "Grounded."),
        ("rag-001", "v2", "fail", ""),
    ]
    assert all(label.labeler == "Pavel Zhukov Atum" for label in labels)


def test_label_goes_on_where_it_stopped(recording):
    runner.invoke(app, label_args(recording), input="p\n\nq\n")
    result = runner.invoke(app, label_args(recording), input="f\nNo source.\np\n\n")
    assert result.exit_code == 0, result.output
    assert "Labelled so far: 1 of 3. To label now: 2." in " ".join(result.output.split())
    assert "Item 1 of 3" not in result.output
    assert "Item 2 of 3" in result.output and "Item 3 of 3" in result.output
    labels_path = recording.root / "labels" / "human.jsonl"
    next_step = (
        f"Next: make eval, then commit {labels_path} and results/judge-agreement.json together."
    )
    assert result.output.splitlines()[-2:] == [
        "Saved 2 labels this time. Labelled so far: 3 of 3. Every sample answer is labelled.",
        next_step,
    ]
    again = runner.invoke(app, label_args(recording), input="")
    assert again.exit_code == 0
    assert "Item" not in again.output
    assert again.output.splitlines()[-2:] == [
        "Labelled so far: 3 of 3. Nothing to label now.",
        next_step,
    ]


def test_label_stops_at_the_end_of_the_input_and_keeps_the_saved_labels(recording):
    result = runner.invoke(app, label_args(recording), input="p\nGood.\nf\n")
    assert result.exit_code == 130
    assert result.output.rstrip().endswith(
        "Stopped. Saved 1 label this time. Labelled so far: 1 of 3. Run make label to go on."
    )
    path = recording.root / "labels" / "human.jsonl"
    assert path.read_text(encoding="utf-8").count("\n") == 1


def test_label_never_shows_what_the_judge_said(recording):
    result = runner.invoke(app, label_args(recording), input="s\ns\ns\n")
    assert result.exit_code == 0, result.output
    lowered = result.output.lower()
    for word in ("judge", "groundedness", "helpfulness", "verdict", "rule_pass", "reasons"):
        assert word not in lowered
    assert not (recording.root / "labels" / "human.jsonl").exists()  # skips write nothing


def test_label_names_stale_answers_and_labels_the_rest(recording):
    sample = recording.sample()
    stale = sample.items[0].model_copy(update={"answer_key": "f" * 64})
    write_sample(
        sample.model_copy(update={"items": [stale, *sample.items[1:]]}),
        recording.root / "labels" / "sample.json",
    )
    result = runner.invoke(app, label_args(recording), input="q\n")
    assert result.exit_code == 0, result.output
    text = " ".join(result.output.split())
    assert "1 sample answer cannot be shown: rag-001 v1 repeat 0: the prompt now differs" in text
    assert "Item 1 of 3" not in result.output and "Item 2 of 3" in result.output


def test_label_without_a_sample_or_with_a_broken_labels_file_fails(recording):
    (recording.root / "labels" / "human.jsonl").write_text("{broken\n", encoding="utf-8")
    result = runner.invoke(app, label_args(recording), input="")
    assert result.exit_code == 1
    assert "human.jsonl line 1: not a valid label" in result.output
    (recording.root / "labels" / "sample.json").unlink()
    result = runner.invoke(app, label_args(recording), input="")
    assert result.exit_code == 1
    assert "no sample in " in result.output and "run llmeval sample" in result.output


# --- llmeval agreement --------------------------------------------------------


def agreement_args(ws):
    return [
        "agreement",
        "--sample",
        str(ws / "labels" / "sample.json"),
        "--labels",
        str(ws / "labels" / "human.jsonl"),
        "--results-dir",
        str(ws / "results"),
        "--cassettes-dir",
        str(ws / "cassettes"),
        "--rubric",
        str(RUBRIC),
    ]


def test_agreement_without_a_recorded_run_is_pending_and_writes_nothing(tmp_path):
    (tmp_path / "cassettes").mkdir()
    result = runner.invoke(app, agreement_args(tmp_path))
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "pending first recorded run"
    assert not (tmp_path / "results").exists()


def test_agreement_without_labels_says_pending_human_labels(tmp_path):
    recorded_run(tmp_path)
    runner.invoke(app, sample_args(tmp_path))
    result = runner.invoke(app, agreement_args(tmp_path))
    assert result.exit_code == 0, result.output
    path = tmp_path / "results" / "judge-agreement.json"
    assert result.output.splitlines() == [
        "judge agreement: pending human labels: 0 of 30 sample answers labelled "
        "(5 the judge failed, 25 it passed)",
        f"  {SAMPLE_NOTE}",
        f"  wrote {path}",
    ]
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["status"] == "pending human labels"
    assert report["sample"]["size"] == 30


def test_agreement_with_labels_prints_the_rate_kappa_confusion_and_disagreements(tmp_path):
    recorded_run(tmp_path)
    runner.invoke(app, sample_args(tmp_path))
    sample = load_sample(tmp_path / "labels" / "sample.json")
    runs = runs_by_answer(
        [
            FunctionResults.model_validate_json(
                (tmp_path / "results" / f"rag-{v}.json").read_text()
            )
            for v in ("v1", "v2")
        ]
    )
    first = sample.items[0]
    judged = runs[first.ref].verdict
    for at, item in enumerate(sample.items[:4]):
        passed = runs[item.ref].verdict if at else not judged
        append_label(
            tmp_path / "labels" / "human.jsonl",
            HumanLabel(
                case=item.case,
                version=item.version,
                repeat=0,
                answer_key=item.answer_key,
                label="pass" if passed else "fail",
                comment="Synthetic comment." if not at else "",
                labeler=LABELER,
                labeled_at=datetime(2026, 1, 2, 9, 30, tzinfo=UTC),
            ),
        )
    result = runner.invoke(app, agreement_args(tmp_path))
    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == "judge agreement (partial): 4 of 30 sample answers labelled"
    assert lines[1].startswith("  percent agreement 75.0% (3 of 4), Cohen's kappa ")
    assert lines[2].startswith("  confusion (judge by human): judge pass: human pass ")
    assert lines[3].startswith(f"  disagreement {first.case} {first.version} repeat 0: judge ")
    assert lines[3].endswith(": Synthetic comment.")
    assert "groundedness" in lines[3]


def test_agreement_refuses_a_missing_sample_or_broken_labels(tmp_path):
    recorded_run(tmp_path)
    result = runner.invoke(app, agreement_args(tmp_path))
    assert result.exit_code == 1
    assert "no sample in " in result.output
    runner.invoke(app, sample_args(tmp_path))
    (tmp_path / "labels" / "human.jsonl").write_text("{}\n", encoding="utf-8")
    result = runner.invoke(app, agreement_args(tmp_path))
    assert result.exit_code == 1
    assert "human.jsonl line 1: not a valid label" in result.output
    assert not (tmp_path / "results" / "judge-agreement.json").exists()


def test_a_second_label_session_is_refused_in_one_line(recording):
    with label_lock(recording.root / "labels" / "human.jsonl"):
        result = runner.invoke(app, label_args(recording), input="p\n\n")
    assert result.exit_code == 1
    assert result.output.strip().splitlines() == [
        f"another make label session holds {recording.root / 'labels' / '.label.lock'}: "
        "finish it first"
    ]
    assert not (recording.root / "labels" / "human.jsonl").exists()
