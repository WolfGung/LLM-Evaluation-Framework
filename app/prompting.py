"""Prompt files and the model interface the features call.

Prompts live in `app/prompts/<function>_<version>.md`, for example
`assistant_v2.md`. A prompt is plain text with `{{name}}` placeholders. The
versions of a function are whatever files exist, so adding `assistant_v3.md`
adds a version without code changes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from llmeval.cassettes import CallTag
from llmeval.client import CallResult
from llmeval.config import RoleConfig

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
FUNCTIONS = ("assistant", "triage")

_VERSION = re.compile(r"^v[1-9][0-9]*$")
_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


class PromptError(ValueError):
    """An unknown prompt, or a template and its values do not fit."""


class ChatModel(Protocol):
    """What the features need from a model: `llmeval.client.ModelClient` or a test fake."""

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        role: RoleConfig,
        response_format: Mapping[str, Any] | None = None,
        repeat: int = 0,
        tag: CallTag | None = None,
    ) -> CallResult: ...


def _check_function(function: str) -> None:
    if function not in FUNCTIONS:
        raise PromptError(f"unknown prompt function {function!r}; known: {', '.join(FUNCTIONS)}")


def prompt_versions(function: str) -> tuple[str, ...]:
    """The versions that have a prompt file, in numeric order (v1, v2, ..., v10)."""
    _check_function(function)
    versions = [
        path.stem.removeprefix(f"{function}_")
        for path in PROMPTS_DIR.glob(f"{function}_v*.md")
        if _VERSION.match(path.stem.removeprefix(f"{function}_"))
    ]
    return tuple(sorted(versions, key=lambda v: int(v[1:])))


def load_prompt(function: str, version: str) -> str:
    """The text of `<function>_<version>.md`."""
    _check_function(function)
    if version not in prompt_versions(function):
        known = ", ".join(prompt_versions(function)) or "none"
        raise PromptError(f"unknown {function} prompt version {version!r}; known: {known}")
    return (PROMPTS_DIR / f"{function}_{version}.md").read_text(encoding="utf-8").strip()


def render(template: str, **values: str) -> str:
    """Fill every `{{name}}` in one pass. Values are inserted as plain text.

    Every placeholder needs a value and every value needs a placeholder, so a
    typo cannot send a prompt with a hole in it.
    """
    wanted = set(_PLACEHOLDER.findall(template))
    if missing := sorted(wanted - set(values)):
        raise PromptError(f"no value for placeholder: {', '.join(missing)}")
    if unused := sorted(set(values) - wanted):
        raise PromptError(f"no placeholder for value: {', '.join(unused)}")
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], template)
