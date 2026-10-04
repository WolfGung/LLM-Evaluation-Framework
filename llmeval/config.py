"""Load the model configuration (config/models.yaml) and the environment settings.

Model ids live only in the YAML file, so changing a model never touches code.
The API key comes only from the environment and is held as a `SecretStr`, so
it never shows up in a repr, a log line or a traceback.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator

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


class RoleConfig(_Strict):
    """Call parameters for one model role (the system under test or the judge)."""

    model: str = Field(min_length=1)
    temperature: float = Field(ge=0, le=2)
    seed: int | None = None
    max_tokens: int = Field(gt=0)


class ModelsConfig(_Strict):
    """The content of config/models.yaml."""

    system: RoleConfig
    judge: RoleConfig
    repeats: int = Field(ge=1)
    rpm: int = Field(ge=1)

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

    mode: Mode = Mode.REPLAY
    max_run_cost_usd: float = Field(default=DEFAULT_MAX_RUN_COST_USD, ge=0)
    api_key: SecretStr | None = None


class Config(_Strict):
    """Everything a run needs: the models file plus the environment."""

    models: ModelsConfig
    settings: Settings


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
    if key and not _KEY_CHARS.match(key):
        raise ConfigError(
            "invalid environment settings: OPENROUTER_API_KEY must be printable ASCII "
            "without spaces (check the keyboard layout and copy the key again)"
        )
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
        raise ConfigError(f"invalid environment settings: {problems}") from None


def load_config(
    path: Path | str = DEFAULT_MODELS_PATH, env: Mapping[str, str] | None = None
) -> Config:
    return Config(models=load_models_config(path), settings=load_settings(env))
