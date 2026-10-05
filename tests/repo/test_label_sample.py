"""The committed label sample is the sample of the committed results.

`llmeval sample` writes `labels/sample.json` from `results/` by a fixed rule
and seed: every judged answer the judge failed, plus a stratified draw of
judge-passed answers. This test draws it again and compares the bytes, so a
hand-picked or edited sample fails. If a new recording changes the results,
the sample changes too: rebuilding it marks the labels of answers that left
it as stale, and `llmeval agreement` says so.

Nothing is written and no model is called. The owner's labels file is never
touched here.
"""

from pathlib import Path

from llmeval.baseline import load_run_results
from llmeval.cassettes import load_manifest
from llmeval.labels import build_sample, runs_by_answer

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "labels" / "sample.json"
FIX = "run llmeval sample and commit labels/sample.json"


def test_the_committed_sample_is_drawn_from_the_committed_results():
    manifest = load_manifest(ROOT / "cassettes")
    if manifest is None:
        assert not SAMPLE.exists(), "a label sample without a recorded run"
        return
    results = load_run_results(ROOT / "results", {"rag": manifest.prompt_versions["rag"]})
    drawn = build_sample(results.functions)
    assert SAMPLE.is_file(), f"labels/sample.json is missing: {FIX}"
    assert SAMPLE.read_text(encoding="utf-8") == drawn.model_dump_json(indent=2) + "\n", FIX


def test_the_sample_holds_both_judge_verdicts():
    manifest = load_manifest(ROOT / "cassettes")
    if manifest is None:
        return
    results = load_run_results(ROOT / "results", {"rag": manifest.prompt_versions["rag"]})
    runs = runs_by_answer(results.functions)
    drawn = build_sample(results.functions)
    verdicts = [runs[item.ref].verdict for item in drawn.items]
    assert True in verdicts and False in verdicts
    failed = [ref for ref, answer in runs.items() if ref[2] == 0 and answer.verdict is False]
    assert {item.ref for item in drawn.items} >= set(failed)
