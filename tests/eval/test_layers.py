"""Layer results: the pass rate of each layer, per function and prompt version.

One test per function, recorded prompt version and layer. It replays every
case of the dataset (the per-case tests share these replays), and the Allure
report shows the layer's pass rate in the title, each check's pass rate and
the failing runs (epic = function, feature = layer, story = "pass rate").

It fails as the gate does: when the rate drops below the baseline's by more
than the layer's tolerance in `config/gate.yaml`. A rise passes; the
per-case tests name the cases that changed. Without a baseline entry for the
layer it is skipped as "pending baseline", after the rate is shown.
"""

import pytest

from llmeval.baseline import PENDING_BASELINE
from llmeval.cassettes import load_manifest
from llmeval.datasets import RAG_PATH, TRIAGE_PATH, load_rag, load_triage
from llmeval.gate import FAILING, GATE_CONFIG_PATH, load_tolerances, rate_row
from llmeval.results import summarise
from tests.eval.report import REGRESSION, show_layer
from tests.eval.support import CASSETTES, ROOT, UNRECORDED
from tools.render import percent, share

CASES = {"rag": load_rag(ROOT / RAG_PATH), "triage": load_triage(ROOT / TRIAGE_PATH)}
LAYERS = {
    "rag": ("retrieval", "deterministic", "reference", "safety", "judge"),
    "triage": ("deterministic", "reference"),
}


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    manifest = load_manifest(CASSETTES)
    params = []
    for function, layers in LAYERS.items():
        if manifest is None:
            versions: tuple[str, ...] = (UNRECORDED,)
        else:
            versions = tuple(manifest.prompt_versions.get(function, ()))
        params += [
            pytest.param(function, version, layer, id=f"{function}-{version}-{layer}")
            for version in versions
            for layer in layers
        ]
    metafunc.parametrize(("function", "version", "layer"), params)


def test_layer_pass_rate(replay, baseline, function, version, layer):
    records = [replay.case(function, case, version) for case in CASES[function]]
    now = summarise(function, records).layers.get(layer)
    if now is None:
        pytest.skip(f"no {layer} layer in the recorded run")
    show_layer(function, version, layer, records, now)
    try:
        expected = baseline.functions[function][version].metrics.layers[layer]
    except (AttributeError, KeyError):
        pytest.skip(PENDING_BASELINE)
    allowed = getattr(load_tolerances(ROOT / GATE_CONFIG_PATH).layers, layer)
    row = rate_row(f"{function} {version} {layer} layer", expected, now.rate, allowed)
    if row.verdict in FAILING:
        pytest.fail(
            f"{REGRESSION}: {row.metric} {percent(now.passed, now.total)} "
            f"({now.passed} of {now.total} runs), baseline {share(expected)}, "
            f"allowed drop {share(allowed).removesuffix('%')} pp",
            pytrace=False,
        )
