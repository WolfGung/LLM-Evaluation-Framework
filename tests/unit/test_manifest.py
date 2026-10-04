"""The recorded-run manifest and the "pending first recorded run" rule.

Synthetic data: every manifest here is made up for the test and written only
into `tmp_path`. No manifest is written into the repository's `cassettes/`.
"""

import json
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from llmeval.cassettes import (
    MANIFEST_FILE,
    PENDING_RECORDED_RUN,
    CassetteError,
    RunManifest,
    load_manifest,
)

HASH = "a" * 64


def manifest_data(**changes):
    data = {
        "models": {"system": "vendor-a/small:free", "judge": "vendor-b/large:free"},
        "prompt_versions": {"rag": ["v1", "v2"], "triage": ["v1", "v2"]},
        "repeats": 3,
        "datasets": {"rag.jsonl": HASH, "triage.jsonl": "b" * 64},
        "recorded_from": "2026-01-01T10:00:00Z",
        "recorded_to": "2026-01-02T18:30:00Z",
        "planned_calls": 480,
        "recorded_calls": 480,
    }
    data.update(changes)
    return data


def test_the_pending_reason_is_exact():
    assert PENDING_RECORDED_RUN == "pending first recorded run"


def test_a_complete_manifest_loads(tmp_path):
    (tmp_path / MANIFEST_FILE).write_text(json.dumps(manifest_data()), encoding="utf-8")
    manifest = load_manifest(tmp_path)
    assert manifest is not None
    assert manifest.models == {"system": "vendor-a/small:free", "judge": "vendor-b/large:free"}
    assert manifest.prompt_versions["triage"] == ("v1", "v2")
    assert manifest.repeats == 3
    assert manifest.recorded_to == datetime(2026, 1, 2, 18, 30, tzinfo=UTC)
    assert manifest.schema_version == 1


def test_no_manifest_means_no_recorded_run(tmp_path):
    assert load_manifest(tmp_path) is None


def test_a_directory_with_gitkeep_is_not_a_recording(tmp_path):
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
    assert load_manifest(tmp_path) is None


def test_cassette_files_without_a_manifest_are_not_a_recording(tmp_path):
    # An interrupted recording leaves cassette files but no manifest.
    (tmp_path / "rag-v1.jsonl").write_text("{}\n", encoding="utf-8")
    assert load_manifest(tmp_path) is None


def test_a_missing_directory_is_not_a_recording(tmp_path):
    assert load_manifest(tmp_path / "absent") is None


def test_a_broken_manifest_fails_loudly(tmp_path):
    (tmp_path / MANIFEST_FILE).write_text("{not json", encoding="utf-8")
    with pytest.raises(CassetteError, match="manifest.json"):
        load_manifest(tmp_path)


def test_a_manifest_with_a_wrong_field_fails_loudly(tmp_path):
    (tmp_path / MANIFEST_FILE).write_text(json.dumps(manifest_data(repeats=0)), encoding="utf-8")
    with pytest.raises(CassetteError, match="manifest.json"):
        load_manifest(tmp_path)


@pytest.mark.parametrize(
    "changes",
    [
        {"extra_field": 1},
        {"models": {"system": "vendor-a/small:free"}},
        {"models": {"system": "a", "judge": "b", "critic": "c"}},
        {"datasets": {"rag.jsonl": "not-a-hash"}},
        {"datasets": {}},
        {"prompt_versions": {"rag": []}},
        {"prompt_versions": {}},
        {"recorded_from": "2026-01-03T00:00:00Z"},
        {"recorded_calls": 479},
        {"planned_calls": 0, "recorded_calls": 0},
    ],
    ids=[
        "extra field",
        "judge missing",
        "unknown role",
        "bad hash",
        "no datasets",
        "no versions for a function",
        "no functions",
        "dates reversed",
        "incomplete recording",
        "nothing planned",
    ],
)
def test_invalid_manifests_are_refused(changes):
    with pytest.raises(ValidationError):
        RunManifest.model_validate(manifest_data(**changes))


def test_the_manifest_round_trips_through_json():
    manifest = RunManifest.model_validate(manifest_data())
    assert RunManifest.model_validate_json(manifest.model_dump_json()) == manifest
