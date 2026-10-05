"""The label sample: which judged answers the owner labels by hand.

Synthetic data: every case, answer, cassette key and judge grade below is
made up for the test. Nothing is written outside pytest's `tmp_path`.
"""

import json
from collections import Counter

import pytest

from llmeval.labels import (
    SAMPLE_RULE,
    TWIN_GAP,
    LabelError,
    build_sample,
    load_sample,
    stratum_quotas,
    write_sample,
)
from tests.unit import synthetic_results as syn
from tests.unit.synthetic_labels import key_of, population, rag_case, two_versions


def test_largest_remainder_quotas_follow_the_stratum_sizes():
    counts = {
        ("v1", "answerable"): 24,
        ("v1", "multi_doc"): 6,
        ("v1", "unanswerable"): 6,
        ("v2", "answerable"): 24,
        ("v2", "multi_doc"): 6,
        ("v2", "unanswerable"): 7,
    }
    quotas = stratum_quotas(counts, 23)
    # 23 x n / 73: floors 7, 1, 1, 7, 1, 2 (19); the four largest remainders
    # (.89 three times, then .56 twice: the tie goes to the earlier stratum).
    assert quotas == {
        ("v1", "answerable"): 8,
        ("v1", "multi_doc"): 2,
        ("v1", "unanswerable"): 2,
        ("v2", "answerable"): 7,
        ("v2", "multi_doc"): 2,
        ("v2", "unanswerable"): 2,
    }
    assert sum(quotas.values()) == 23


def test_quotas_never_exceed_a_stratum_and_take_everything_when_short():
    assert stratum_quotas({"a": 2, "b": 1}, 5) == {"a": 2, "b": 1}
    assert stratum_quotas({"a": 3, "b": 0}, 2) == {"a": 2, "b": 0}
    assert stratum_quotas({}, 4) == {}


def test_the_sample_holds_every_judge_failure_and_fills_up_with_passes():
    sample = build_sample(two_versions(), size=30, seed=11)
    assert sample.size == 30 and len(sample.items) == 30
    failures = {
        (case.id, result.version)
        for result in two_versions()
        for case in result.cases
        if case.runs[0].judge.rule_pass is False
    }
    assert len(failures) == 7
    chosen = {(item.case, item.version) for item in sample.items}
    assert failures <= chosen
    assert len(chosen) == 30  # no answer twice


def test_the_passes_are_stratified_by_version_and_category():
    sample = build_sample(two_versions(), size=30, seed=11)
    results = {
        (r.version, c.id): c.runs[0].judge.rule_pass for r in two_versions() for c in r.cases
    }
    passes = Counter(
        (item.version, item.category) for item in sample.items if results[(item.version, item.case)]
    )
    assert passes == {
        ("v1", "answerable"): 8,
        ("v1", "multi_doc"): 2,
        ("v1", "unanswerable"): 2,
        ("v2", "answerable"): 7,
        ("v2", "multi_doc"): 2,
        ("v2", "unanswerable"): 2,
    }


def test_each_item_names_the_answer_by_its_cassette_key():
    sample = build_sample(two_versions(), size=30, seed=11)
    for item in sample.items:
        assert item.repeat == 0
        assert item.answer_key == key_of(item.case, item.version, 0)
        assert item.category in {"answerable", "multi_doc", "unanswerable"}


def test_the_same_seed_gives_the_same_sample_and_another_seed_another():
    assert build_sample(two_versions(), seed=11) == build_sample(two_versions(), seed=11)
    other = build_sample(two_versions(), seed=12)
    assert other.seed == 12
    assert {i.answer_key for i in other.items} != {
        i.answer_key for i in build_sample(two_versions(), seed=11).items
    }


def test_the_labelling_order_does_not_group_the_judge_failures():
    sample = build_sample(two_versions(), size=30, seed=11)
    results = {
        (r.version, c.id): c.runs[0].judge.rule_pass for r in two_versions() for c in r.cases
    }
    positions = [
        at for at, item in enumerate(sample.items) if results[(item.version, item.case)] is False
    ]
    assert positions != list(range(7))
    assert positions != list(range(23, 30))


def test_only_repeat_0_of_judged_answers_with_a_valid_verdict_is_drawn():
    results = [
        syn.function_results(
            "rag",
            "v1",
            [
                rag_case("rag-001", "v1", "answerable", "pass", "fail", "fail"),
                rag_case("rag-002", "v1", "answerable", "invalid"),
                rag_case("rag-003", "v1", "unanswerable", "fail"),
                rag_case("rag-041", "v1", "safety", None),
            ],
            repeats=3,
        ),
        syn.function_results(
            "triage", "v1", [syn.case_record("tri-001", checks=syn.TRIAGE_CHECKS)]
        ),
    ]
    sample = build_sample(results, size=30, seed=1)
    assert [(i.case, i.repeat) for i in sorted(sample.items, key=lambda i: i.case)] == [
        ("rag-001", 0),
        ("rag-003", 0),
    ]
    assert sample.size == 2  # fewer candidates than asked: all of them


def test_more_judge_failures_than_the_sample_size_is_refused():
    result = population("v1", {"answerable": (2, 4)})
    with pytest.raises(LabelError, match="4 judge failures do not fit a sample of 3"):
        build_sample([result], size=3, seed=1)


def test_the_sample_file_starts_with_the_seed_and_the_rule(tmp_path):
    sample = build_sample(two_versions(), size=30, seed=11)
    path = write_sample(sample, tmp_path / "labels" / "sample.json")
    text = path.read_text(encoding="utf-8")
    assert text.endswith("}\n")
    assert list(json.loads(text)) == [
        "schema_version",
        "seed",
        "rule",
        "size",
        "items",
    ]
    assert '"seed": 11' in text and SAMPLE_RULE in text
    assert load_sample(path) == sample
    # Nothing about the judge's verdict is in the file.
    for word in ("pass", "fail", "score", "reason", "judge_"):
        assert f'"{word}' not in text


def test_a_missing_or_broken_sample_file_is_named(tmp_path):
    with pytest.raises(LabelError, match="no sample in .*sample.json: run llmeval sample"):
        load_sample(tmp_path / "sample.json")
    (tmp_path / "sample.json").write_text("{}", encoding="utf-8")
    with pytest.raises(LabelError, match="sample.json: not a valid sample"):
        load_sample(tmp_path / "sample.json")


def twin_gaps(sample) -> list[int]:
    """The distance in labelling order between the two versions of each case."""
    places: dict[str, list[int]] = {}
    for at, item in enumerate(sample.items):
        places.setdefault(item.case, []).append(at)
    return [abs(first - second) for first, second in (p for p in places.values() if len(p) == 2)]


@pytest.mark.parametrize("seed", range(12))
def test_the_two_versions_of_a_case_are_at_least_3_apart(seed):
    sample = build_sample(two_versions(), size=30, seed=seed)
    gaps = twin_gaps(sample)
    assert gaps and min(gaps) >= TWIN_GAP == 3


def test_twins_are_kept_apart_even_when_every_case_comes_twice():
    counts = {"answerable": (12, 3)}
    sample = build_sample([population("v1", counts), population("v2", counts)], size=30, seed=5)
    gaps = twin_gaps(sample)
    assert len(gaps) == 15 and min(gaps) >= 3


def test_twins_that_cannot_be_kept_apart_are_refused():
    counts = {"answerable": (1, 1)}
    with pytest.raises(LabelError, match="cannot keep the two versions of each case 3 apart"):
        build_sample([population("v1", counts), population("v2", counts)], size=4, seed=1)
