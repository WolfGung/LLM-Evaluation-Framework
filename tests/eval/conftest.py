"""Replay evaluation: every case of every dataset, for each recorded prompt version.

- No `cassettes/manifest.json`: every test skips with "pending first recorded run".
- A manifest: each case is replayed; a call missing from the cassettes fails
  its test with the replay-miss message. The prompt versions come from the
  manifest, not from the prompt files; a function the manifest does not list
  skips with "<function> is not in the recorded run". The pairwise tests
  take the compared version pairs (the first version with each later one);
  with one version they skip with "nothing to compare".
- A broken manifest fails collection, and the summary ends with one
  "manifest error: ..." line.
- A rubric that differs from the recorded one adds one "notice: rubric
  changed since the recording" line; its judge calls then miss in replay.
- No baseline (`results/baseline.json`), or no entry for the case: skipped
  with "pending baseline" after the replay.
- A baseline entry: compared with `llmeval.baseline.compare` (pass, xfail,
  strict XPASS, or a failing regression).
- With `--alluredir`, the Allure categories and environment are written
  there at the end (see `tests.eval.report`).

No model is called: the client is in replay mode and needs no key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llmeval.baseline import load_baseline
from llmeval.cassettes import PENDING_RECORDED_RUN, CassetteError, load_manifest
from llmeval.checks.judge import RUBRIC_PATH, RubricError, load_rubric
from llmeval.datasets import file_sha256
from llmeval.runner import pairs_of
from tests.eval.report import NOT_RECORDED, NOTHING_TO_COMPARE, write_report_files
from tests.eval.support import BASELINE, CASSETTES, DATASETS, ROOT, UNRECORDED, Replay


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "case" not in metafunc.fixturenames:
        return
    cases = metafunc.module.CASES
    metafunc.parametrize("case", cases, ids=[case.id for case in cases])
    function = metafunc.module.FUNCTION
    manifest = load_manifest(CASSETTES)
    versions = [UNRECORDED] if manifest is None else manifest.prompt_versions.get(function, [])
    if "version" in metafunc.fixturenames:
        metafunc.parametrize("version", list(versions) or [_skipped(f"{function} {NOT_RECORDED}")])
    if "pair" in metafunc.fixturenames:
        if manifest is None:
            pairs = [pytest.param((UNRECORDED, UNRECORDED), id=UNRECORDED)]
        elif not versions:
            pairs = [_skipped(f"{function} {NOT_RECORDED}")]
        elif len(versions) < 2:
            pairs = [_skipped(f"{function} has one prompt version: {NOTHING_TO_COMPARE}")]
        else:
            pairs = [pytest.param(pair, id="-vs-".join(pair)) for pair in pairs_of(versions)]
        metafunc.parametrize("pair", pairs)


def _skipped(reason: str):
    return pytest.param(UNRECORDED, marks=pytest.mark.skip(reason=reason), id=UNRECORDED)


@pytest.fixture(scope="session")
def replay() -> Replay:
    manifest = load_manifest(CASSETTES)
    if manifest is None:
        pytest.skip(PENDING_RECORDED_RUN)
    return Replay(manifest)


@pytest.fixture(scope="session")
def baseline():
    return load_baseline(BASELINE)


def pytest_terminal_summary(terminalreporter) -> None:
    try:
        manifest = load_manifest(CASSETTES)
    except CassetteError as exc:
        terminalreporter.write_line(f"manifest error: {str(exc).splitlines()[0]}")
        return
    if manifest is None:
        return
    current = {path.name: file_sha256(path) for path in DATASETS.glob("*.jsonl")}
    if notice := manifest.dataset_notice(current):
        terminalreporter.write_line(notice)
    try:
        rubric = load_rubric(ROOT / RUBRIC_PATH)
    except RubricError as exc:
        terminalreporter.write_line(f"rubric error: {exc}")
        return
    if notice := manifest.rubric_notice(rubric.sha256):
        terminalreporter.write_line(notice)


def pytest_sessionfinish(session: pytest.Session) -> None:
    directory = session.config.getoption("allure_report_dir", default=None)
    if directory:
        write_report_files(Path(directory), CASSETTES)
