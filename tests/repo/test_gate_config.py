"""The repository's gate tolerances, and the gate on the committed results.

`config/gate.yaml` gives every tolerance a one-line reason, keeps safety and
retrieval at 0, and says that a replay always equals the baseline. The gate
then passes on the committed results against the committed baseline (they
are one replay of the cassettes, see test_results_fresh.py). No model is
called and nothing is written.
"""

from pathlib import Path

from typer.testing import CliRunner

from llmeval.cli import app
from llmeval.gate import load_tolerances

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "gate.yaml"


def test_the_gate_tolerances_load():
    tolerances = load_tolerances(CONFIG)
    assert tolerances.layers.safety == 0  # any new safety failure fails
    assert tolerances.layers.retrieval == 0  # the search does not use the model


def tolerance_lines(text: str) -> list[str]:
    """The lines that set a value (not a section heading, not a comment)."""
    values = [line for line in text.splitlines() if ":" in line]
    values = [line for line in values if not line.lstrip().startswith("#")]
    return [line for line in values if line.split(":", 1)[1].split("#")[0].strip()]


def unexplained(text: str) -> list[str]:
    """Value lines without a comment of at least ten characters after `#`."""
    return [
        line
        for line in tolerance_lines(text)
        if "#" not in line or len(line.split("#", 1)[1].strip()) < 10
    ]


def test_every_tolerance_has_a_reason():
    text = CONFIG.read_text(encoding="utf-8")
    assert len(tolerance_lines(text)) == 13  # every key of llmeval.gate.Tolerances
    missing = unexplained(text)
    assert not missing, f"a tolerance without a reason: {missing}"


def test_a_tolerance_without_its_comment_counts_as_unexplained():
    text = CONFIG.read_text(encoding="utf-8")
    line = next(line for line in text.splitlines() if line.startswith("all_checks:"))
    without = text.replace(line, line.split("#", 1)[0].rstrip())
    assert unexplained(without) == ["all_checks: 0.05"]
    short = text.replace(line, "all_checks: 0.05  # short")
    assert unexplained(short) == ["all_checks: 0.05  # short"]


def test_the_tolerances_say_that_a_replay_always_equals_the_baseline():
    text = " ".join(CONFIG.read_text(encoding="utf-8").split())
    assert "deterministic" in text
    assert "live (drift) runs" in text
    assert "docs/02" in text


def test_the_gate_passes_on_the_committed_results():
    result = CliRunner().invoke(
        app,
        [
            "gate",
            "--results-dir",
            str(ROOT / "results"),
            "--baseline",
            str(ROOT / "results" / "baseline.json"),
            "--tolerances",
            str(CONFIG),
            "--cassettes-dir",
            str(ROOT / "cassettes"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "(replay results)" in result.output
    assert "REGRESSION" not in result.output and "missing" not in result.output
    assert result.output.rstrip().endswith("within tolerance")
