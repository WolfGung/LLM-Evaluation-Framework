"""Configuration loading and validation.

Synthetic data: the YAML documents and environment mappings below are made up
for the tests; only `test_repository_config_*` reads the real config file.
"""

import traceback
from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from llmeval.config import (
    ConfigError,
    Mode,
    ReasoningConfig,
    Settings,
    check_stability_cases,
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


def test_the_judge_grades_the_first_repeat_and_every_case_repeats_by_default(tmp_path):
    config = load_models_config(write(tmp_path, VALID_YAML))

    assert config.judge_repeats == "first"
    assert config.stability_cases is None


def test_the_judge_can_grade_every_repeat(tmp_path):
    config = load_models_config(write(tmp_path, VALID_YAML + "judge_repeats: all\n"))

    assert config.judge_repeats == "all"


def test_a_stability_subset_is_read_in_order(tmp_path):
    text = VALID_YAML + "stability_cases: [rag-003, tri-001]\n"
    config = load_models_config(write(tmp_path, text))

    assert config.stability_cases == ("rag-003", "tri-001")


@pytest.mark.parametrize(
    "extra",
    [
        "judge_repeats: some\n",
        "judge_repeats: 2\n",
        "stability_cases: []\n",
        "stability_cases: [rag-001, rag-001]\n",
        "stability_cases: rag-001\n",
    ],
    ids=["unknown judge_repeats", "number", "empty subset", "repeated id", "not a list"],
)
def test_invalid_judge_repeats_and_subsets_are_rejected(tmp_path, extra):
    with pytest.raises(ConfigError):
        load_models_config(write(tmp_path, VALID_YAML + extra))


def test_a_subset_must_name_cases_of_the_datasets(tmp_path):
    text = VALID_YAML + "stability_cases: [rag-001, rag-999]\n"
    config = load_models_config(write(tmp_path, text))

    check_stability_cases(config, {"rag-001", "rag-999"})
    with pytest.raises(ConfigError) as caught:
        check_stability_cases(config, {"rag-001", "tri-001"})
    assert str(caught.value) == "stability_cases names cases that are not in the datasets: rag-999"


def test_no_subset_needs_no_check(tmp_path):
    check_stability_cases(load_models_config(write(tmp_path, VALID_YAML)), set())


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
    # The first recording (2026-10-05) ran out of budget at 1500 in 13 of 154
    # judge calls; finished calls used up to 1416 reasoning tokens.
    assert config.judge.max_tokens >= 4096
    assert config.judge.temperature == 0
    assert config.repeats == 3
    assert config.rpm == 18
    # Written out in the file, so the owner sees both levers.
    text = REPO_CONFIG.read_text(encoding="utf-8")
    assert "\njudge_repeats: first" in text and "\nstability_cases: null" in text
    assert config.judge_repeats == "first"
    assert config.stability_cases is None


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


@pytest.mark.parametrize("bad_key", ["k-123\u0444abc", "k-123\nabc", "k-123 abc"])
def test_settings_built_directly_refuse_a_bad_key(bad_key):
    with pytest.raises(ValidationError) as caught:
        Settings(api_key=SecretStr(bad_key))

    assert "OPENROUTER_API_KEY" in str(caught.value)
    assert "abc" not in str(caught.value)


@pytest.mark.parametrize("bad_key", ["k-123\u0444abc", "k-123\nabc", "k-123 abc"])
def test_a_bad_plain_string_key_is_not_echoed(bad_key):
    with pytest.raises(ValidationError) as caught:
        Settings(api_key=bad_key)

    assert "abc" not in str(caught.value)
    assert "abc" not in repr(caught.value)


def test_load_settings_error_carries_no_trace_of_the_key():
    secret = "k-SECRET-123\u0444tail"

    with pytest.raises(ConfigError) as caught:
        load_settings({"OPENROUTER_API_KEY": secret})

    printed = "".join(traceback.format_exception(caught.value)) + repr(caught.value)
    assert "SECRET" not in printed
    # A chained pydantic error would still hold the key as its input value.
    assert caught.value.__context__ is None
    assert caught.value.__cause__ is None


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


def test_a_role_is_found_by_its_name(tmp_path):
    config = load_models_config(write(tmp_path, VALID_YAML))

    assert config.role("system") is config.system
    assert config.role("judge") is config.judge
