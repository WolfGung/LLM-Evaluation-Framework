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
    ManifestMismatch,
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


def test_without_a_stability_subset_every_case_repeats():
    manifest = RunManifest.model_validate(manifest_data())
    assert manifest.stability_cases is None
    assert manifest.repeats_for("rag-001") == 3


def test_a_stability_subset_repeats_only_its_cases():
    manifest = RunManifest.model_validate(manifest_data(stability_cases=["rag-001", "tri-004"]))
    assert manifest.repeats_for("rag-001") == 3
    assert manifest.repeats_for("rag-002") == 1


def test_an_empty_stability_subset_is_refused():
    with pytest.raises(ValidationError):
        RunManifest.model_validate(manifest_data(stability_cases=[]))


def test_matching_models_pass_the_check():
    manifest = RunManifest.model_validate(manifest_data())
    manifest.check_models(system="vendor-a/small:free", judge="vendor-b/large:free")


def test_a_changed_model_is_a_clear_error():
    manifest = RunManifest.model_validate(manifest_data())
    with pytest.raises(ManifestMismatch) as caught:
        manifest.check_models(system="vendor-c/other:free", judge="vendor-b/large:free")
    assert str(caught.value) == (
        "system model: recorded with vendor-a/small:free, config says vendor-c/other:free: "
        "re-record or restore the config"
    )
    assert isinstance(caught.value, CassetteError)


def test_changed_datasets_are_named():
    manifest = RunManifest.model_validate(manifest_data())
    assert manifest.changed_datasets({"rag.jsonl": HASH, "triage.jsonl": "b" * 64}) == []
    assert manifest.changed_datasets({"rag.jsonl": "c" * 64, "triage.jsonl": "b" * 64}) == [
        "rag.jsonl"
    ]
    notice = manifest.dataset_notice({"rag.jsonl": "c" * 64, "triage.jsonl": "b" * 64})
    assert notice is not None and notice.startswith("notice: ")
    assert "rag.jsonl" in notice and "\n" not in notice
    assert manifest.dataset_notice({"rag.jsonl": HASH, "triage.jsonl": "b" * 64}) is None
