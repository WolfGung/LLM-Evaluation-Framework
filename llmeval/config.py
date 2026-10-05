"""Load the model configuration (config/models.yaml) and the environment settings.

Model ids live only in the YAML file, so changing a model never touches code.
The API key comes only from the environment and is held as a `SecretStr`, so
it never shows up in a repr, a log line or a traceback.
"""

from __future__ import annotations

import os
import re
from collections.abc import Collection, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)

DEFAULT_MODELS_PATH = Path("config/models.yaml")
DEFAULT_MAX_RUN_COST_USD = 1.00

# Printable ASCII without whitespace. Anything else cannot go into an HTTP
# header, and httpx would fail with an error whose repr holds the whole key.
_KEY_CHARS = re.compile(r"^[\x21-\x7e]+$")


class ConfigError(ValueError):
    """The configuration file or the environment is invalid."""


class Mode(StrEnum):
    """How the model client gets answers.

    live:   call the API, keep nothing.
    record: call the API and append every call to the cassettes.
    replay: answer from the cassettes only; never touch the network.
    """

    LIVE = "live"
    RECORD = "record"
    REPLAY = "replay"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# OpenRouter's reasoning effort levels.
ReasoningEffort = Literal["max", "xhigh", "high", "medium", "low", "minimal", "none"]


class ReasoningConfig(_Strict):
    """OpenRouter's `reasoning` request option for models that think before answering.

    Reasoning tokens are billed and count towards the answer length, so a role
    sets the effort (or a token budget) on purpose instead of taking the model
    default. OpenRouter accepts an effort or a token budget, not both.
    """

    effort: ReasoningEffort | None = None
    max_tokens: int | None = Field(default=None, gt=0)
    exclude: bool | None = None

    @model_validator(mode="after")
    def _effort_or_budget(self) -> ReasoningConfig:
        if self.effort is not None and self.max_tokens is not None:
            raise ValueError("reasoning takes effort or max_tokens, not both")
        return self

    def to_request(self) -> dict[str, object]:
        return self.model_dump(exclude_none=True)


class RoleConfig(_Strict):
    """Call parameters for one model role (the system under test or the judge).

    `structured_output: true` sends the JSON schema as `response_format` and
    routes only to endpoints that enforce it. Set it only for a model that
    lists `response_format` and `structured_outputs` on OpenRouter; for any
    other model the caller puts the schema in the prompt and validates the reply.
    """

    model: str = Field(min_length=1)
    temperature: float = Field(ge=0, le=2)
    seed: int | None = None
    max_tokens: int = Field(gt=0)
    reasoning: ReasoningConfig | None = None
    structured_output: bool


# Which runs of a case the judge grades: repeat 0 only, or every repeat.
JudgeRepeats = Literal["first", "all"]


class ModelsConfig(_Strict):
    """The content of config/models.yaml.

    - `repeats`: runs per case for the stability layer.
    - `stability_cases`: the case ids that run `repeats` times; every other
      case runs once (repeat 0). None means every case runs `repeats` times.
      `check_stability_cases` checks the ids against the datasets.
    - `judge_repeats`: `first` grades repeat 0 only, `all` grades every
      repeat. The stability layer uses the rule-based layers only, so judging
      repeats 1..N buys nothing the reports use; `first` is the default, and
      extra judge calls can be recorded later without touching the others.
    - `rpm`: requests per minute the recording keeps to.
    """

    system: RoleConfig
    judge: RoleConfig
    repeats: int = Field(ge=1)
    rpm: int = Field(ge=1)
    judge_repeats: JudgeRepeats = "first"
    stability_cases: Annotated[tuple[str, ...], Field(min_length=1)] | None = None

    def role(self, name: Literal["system", "judge"]) -> RoleConfig:
        """The role config by name: `system` or `judge`."""
        return self.system if name == "system" else self.judge

    @field_validator("stability_cases")
    @classmethod
    def _each_case_once(cls, value: tuple[str, ...] | None) -> tuple[str, ...] | None:
        if value is not None and len(set(value)) != len(value):
            raise ValueError("stability_cases names a case twice")
        return value

    @model_validator(mode="after")
    def _judge_is_another_model(self) -> ModelsConfig:
        # A model tends to rate its own answers higher (self-preference bias),
        # so the judge must never be the model it grades.
        if self.judge.model == self.system.model:
            raise ValueError(
                f"judge.model must differ from system.model ({self.system.model!r}): "
                "a model grading its own answers shows self-preference bias"
            )
        return self


class Settings(_Strict):
    """Values that come from the environment, not from the repository."""

    # pydantic quotes the rejected input in its errors; for an invalid key
    # that input is the key itself.
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    mode: Mode = Mode.REPLAY
    max_run_cost_usd: float = Field(default=DEFAULT_MAX_RUN_COST_USD, ge=0)
    api_key: SecretStr | None = None

    @field_validator("api_key")
    @classmethod
    def _key_fits_in_a_header(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and not _KEY_CHARS.match(value.get_secret_value()):
            raise ValueError(
                "OPENROUTER_API_KEY must be printable ASCII without spaces "
                "(check the keyboard layout and copy the key again)"
            )
        return value


class Config(_Strict):
    """Everything a run needs: the models file plus the environment."""

    models: ModelsConfig
    settings: Settings


def check_stability_cases(models: ModelsConfig, known: Collection[str]) -> None:
    """Raise `ConfigError` when `stability_cases` names a case the datasets do not have."""
    if models.stability_cases is None:
        return
    unknown = [case for case in models.stability_cases if case not in known]
    if unknown:
        raise ConfigError(
            f"stability_cases names cases that are not in the datasets: {', '.join(unknown)}"
        )


def load_models_config(path: Path | str = DEFAULT_MODELS_PATH) -> ModelsConfig:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read model config {path}: {exc.strerror}") from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"model config {path} is not valid YAML: {exc}") from None
    try:
        return ModelsConfig.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(f"model config {path} is invalid:\n{exc}") from None


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if env is None else env
    key = env.get("OPENROUTER_API_KEY", "").strip()
    values: dict[str, object] = {"api_key": key or None}
    if mode := env.get("LLMEVAL_MODE", "").strip():
        values["mode"] = mode
    if limit := env.get("MAX_RUN_COST_USD", "").strip():
        values["max_run_cost_usd"] = limit
    try:
        return Settings.model_validate(values)
    except ValidationError as exc:
        # Report only the field names and messages: the input values could
        # include the API key.
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
    # Raised outside the except block: a chained ValidationError would keep the
    # key as its input value in __context__, even with `from None`.
    raise ConfigError(f"invalid environment settings: {problems}")


def load_config(
    path: Path | str = DEFAULT_MODELS_PATH, env: Mapping[str, str] | None = None
) -> Config:
    return Config(models=load_models_config(path), settings=load_settings(env))
