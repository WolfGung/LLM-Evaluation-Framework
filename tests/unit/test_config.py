"""Configuration loading and validation.

Synthetic data: the YAML documents and environment mappings below are made up
for the tests; only `test_repository_config_*` reads the real config file.
"""

from pathlib import Path

import pytest

from llmeval.config import (
    ConfigError,
    Mode,
    ReasoningConfig,
    load_config,
    load_models_config,
    load_settings,
)

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config" / "models.yaml"

VALID_YAML = """
system:
  model: vendor-a/small:free
  temperature: 0.2
  max_tokens: 600
  reasoning: {effort: none}
  structured_output: false
judge:
  model: vendor-b/large:free
  temperature: 0
  seed: 7
  max_tokens: 1500
  reasoning: {effort: low}
  structured_output: true
repeats: 3
rpm: 18
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_valid_config_loads(tmp_path):
    config = load_models_config(write(tmp_path, VALID_YAML))

    assert config.system.model == "vendor-a/small:free"
    assert config.system.seed is None
    assert config.system.reasoning.effort == "none"
    assert config.system.structured_output is False
    assert config.judge.temperature == 0
    assert config.judge.reasoning.effort == "low"
    assert config.judge.structured_output is True
    assert config.repeats == 3
    assert config.rpm == 18


def test_judge_must_differ_from_system(tmp_path):
    same = VALID_YAML.replace("vendor-b/large:free", "vendor-a/small:free")

    with pytest.raises(ConfigError, match="self-preference"):
        load_models_config(write(tmp_path, same))


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("repeats: 3", "repeats: 0"),
        ("rpm: 18", "rpm: 0"),
        ("max_tokens: 1500", "max_tokens: 0"),
        ("temperature: 0\n", "temperature: 3\n"),
        ("rpm: 18", "rpm: 18\nunknown_field: 1"),
        ("{effort: low}", "{effort: extreme}"),
        ("{effort: low}", "{effort: low, max_tokens: 500}"),
        ("{effort: low}", "{max_tokens: 0}"),
        ("{effort: low}", "{effort: low, budget: 1}"),
        ("  structured_output: true\n", ""),
    ],
)
def test_invalid_values_are_rejected(tmp_path, old, new):
    with pytest.raises(ConfigError):
        load_models_config(write(tmp_path, VALID_YAML.replace(old, new)))


def test_missing_file_names_the_path(tmp_path):
    missing = tmp_path / "nope.yaml"

    with pytest.raises(ConfigError, match="nope.yaml"):
        load_models_config(missing)


def test_reasoning_settings_become_the_request_option():
    assert ReasoningConfig(effort="none").to_request() == {"effort": "none"}
    assert ReasoningConfig(max_tokens=800, exclude=True).to_request() == {
        "max_tokens": 800,
        "exclude": True,
    }


def test_repository_config_is_valid_and_free():
    config = load_models_config(REPO_CONFIG)

    assert config.system.model.endswith(":free")
    assert config.judge.model.endswith(":free")
    # The system model lists no response_format or seed on OpenRouter, so it
    # gets the schema in the prompt; the judge uses enforced structured output.
    assert config.system.structured_output is False
    assert config.system.seed is None
    assert config.judge.structured_output is True
    assert config.judge.seed is not None
    assert config.judge.reasoning is not None and config.judge.reasoning.effort == "low"
    assert config.judge.max_tokens >= 1000
    assert config.judge.temperature == 0
    assert config.repeats == 3
    assert config.rpm == 18


def test_settings_defaults():
    settings = load_settings({})

    assert settings.mode is Mode.REPLAY
    assert settings.max_run_cost_usd == 1.00
    assert settings.api_key is None


def test_settings_from_environment():
    settings = load_settings(
        {"LLMEVAL_MODE": "record", "MAX_RUN_COST_USD": "0.25", "OPENROUTER_API_KEY": "k-123"}
    )

    assert settings.mode is Mode.RECORD
    assert settings.max_run_cost_usd == 0.25
    assert settings.api_key is not None
    assert settings.api_key.get_secret_value() == "k-123"


def test_settings_never_show_the_key():
    settings = load_settings({"OPENROUTER_API_KEY": "k-very-secret"})

    assert "k-very-secret" not in repr(settings)
    assert "k-very-secret" not in str(settings)


def test_empty_key_counts_as_missing():
    assert load_settings({"OPENROUTER_API_KEY": "  "}).api_key is None


@pytest.mark.parametrize(
    "bad_key",
    [
        "k-123\u0444abc",  # a Cyrillic letter typed on the wrong keyboard layout
        "k-123\x01abc",  # an embedded control character
        "k-123 abc",  # an embedded space
    ],
)
def test_key_must_be_printable_ascii(bad_key):
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY") as caught:
        load_settings({"OPENROUTER_API_KEY": bad_key})

    assert bad_key not in str(caught.value)
    assert "abc" not in str(caught.value)


@pytest.mark.parametrize(
    "env",
    [{"LLMEVAL_MODE": "fake"}, {"MAX_RUN_COST_USD": "-1"}, {"MAX_RUN_COST_USD": "lots"}],
)
def test_invalid_settings_are_rejected(env):
    with pytest.raises(ConfigError):
        load_settings(env)


def test_load_config_combines_file_and_environment(tmp_path):
    config = load_config(write(tmp_path, VALID_YAML), {"LLMEVAL_MODE": "live"})

    assert config.models.judge.model == "vendor-b/large:free"
    assert config.settings.mode is Mode.LIVE
