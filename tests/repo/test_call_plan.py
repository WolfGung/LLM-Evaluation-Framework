"""The call plan of the repository's own config and datasets, and the commands that need no key.

No model is called, no key is used and nothing reaches the network. A paid
model in the config makes the estimate read the published prices, and the
only request answered is that public model list, served here with synthetic
prices; a `:free` id needs no price at all. Status without a key reads
nothing. The cassettes directory is an empty one in `tmp_path`, so the
repository's `cassettes/` and `results/` are never touched.
"""

from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

from llmeval import cli
from llmeval.callplan import PlanInputs, count_plan, full_plan
from llmeval.cassettes import repeats_for
from llmeval.checks.judge import RUBRIC_PATH, load_rubric
from llmeval.config import load_models_config
from llmeval.datasets import RAG_PATH, TRIAGE_PATH, load_rag, load_triage
from llmeval.runner import JUDGED_CATEGORIES, versions_of

ROOT = Path(__file__).resolve().parents[2]
runner = CliRunner()


@pytest.fixture(autouse=True)
def no_key_no_network(monkeypatch):
    for name in ("OPENROUTER_API_KEY", "LLMEVAL_MODE", "MAX_RUN_COST_USD"):
        monkeypatch.delenv(name, raising=False)

    models = load_models_config(ROOT / "config" / "models.yaml")
    # Synthetic prices (USD per token) for the configured models, in the shape
    # of GET /api/v1/models; any other request fails the test.
    listing = {
        "data": [
            {"id": role.model, "pricing": {"prompt": "0.0000001", "completion": "0.0000003"}}
            for role in (models.system, models.judge)
        ]
    }

    def published_prices_only(request):
        if request.method == "GET" and request.url.path == "/api/v1/models":
            assert "authorization" not in request.headers, "the model list needs no key"
            return httpx.Response(200, json=listing)
        raise AssertionError(f"no request expected: {request.url}")

    monkeypatch.setattr(
        cli, "_network", lambda: cli.Network(httpx.MockTransport(published_prices_only))
    )


def test_the_plan_follows_the_config_and_the_datasets():
    models = load_models_config(ROOT / "config" / "models.yaml")
    rag, triage = load_rag(ROOT / RAG_PATH), load_triage(ROOT / TRIAGE_PATH)
    versions = {f: versions_of(f) for f in ("rag", "triage")}
    inputs = PlanInputs(models, rag, triage, load_rubric(ROOT / RUBRIC_PATH), versions)
    counts = count_plan(full_plan(inputs, None), None, models)
    assert models.judge_repeats == "first"
    judged = sum(case.category in JUDGED_CATEGORIES for case in rag)
    rag_versions, triage_versions = len(versions["rag"]), len(versions["triage"])

    def runs(cases):
        return sum(repeats_for(case.id, models.repeats, models.stability_cases) for case in cases)

    assert counts.system == runs(rag) * rag_versions + runs(triage) * triage_versions
    # Repeat 0 graded per version, plus two pairwise orders per later version.
    assert counts.judge == judged * rag_versions + judged * 2 * (rag_versions - 1)
    assert counts.waiting == counts.judge and not counts.exact


@pytest.mark.parametrize("command", ["estimate", "status"])
def test_estimate_and_status_run_without_a_key(tmp_path, command):
    args = [
        command,
        "--config",
        str(ROOT / "config" / "models.yaml"),
        "--datasets-dir",
        str(ROOT / "datasets"),
        "--rubric",
        str(ROOT / RUBRIC_PATH),
        "--cassettes-dir",
        str(tmp_path / "cassettes"),
    ]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 0, result.output
    assert "call plan: repeats" in result.output
    assert not (tmp_path / "cassettes").exists()
