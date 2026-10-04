"""Cassettes: recorded model calls stored as JSONL, looked up by a request key.

A cassette entry holds exactly what was sent (the request body) and what came
back (answer, usage, cost, latency, timestamps). It has no field for headers,
so the API key has nowhere to go.

The key is a sha256 of the canonical JSON of the request plus the repeat
index. Repeats are part of the key on purpose: three runs of one request are
three separate recordings, which is what the stability layer measures.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# Request fields besides model and messages that change the answer.
KEY_PARAMS = ("temperature", "seed", "max_tokens", "response_format", "provider")

UNTAGGED_FILE_STEM = "untagged"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class CassetteError(RuntimeError):
    """A cassette file is broken or a write would corrupt the store."""


def utc_now() -> datetime:
    return datetime.now(UTC)


def canonical_json(value: Any) -> str:
    """JSON with sorted keys and no spaces, so equal data gives equal text."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def request_key(body: Mapping[str, Any], repeat: int) -> str:
    material = {
        "model": body["model"],
        "messages": body["messages"],
        "params": {name: body.get(name) for name in KEY_PARAMS},
        "repeat": repeat,
    }
    return hashlib.sha256(canonical_json(material).encode("utf-8")).hexdigest()


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CallTag(_Record):
    """Which evaluation call this is: function (rag, triage, judge), case id, prompt version.

    The tag names the cassette file (`<function>-<version>.jsonl`) and makes a
    missing recording easy to find.
    """

    function: str
    case: str
    version: str

    @field_validator("function", "version")
    @classmethod
    def _safe_for_a_file_name(cls, value: str) -> str:
        if not _SAFE_NAME.match(value):
            raise ValueError(f"unsafe name for a cassette file: {value!r}")
        return value

    @property
    def file_stem(self) -> str:
        return f"{self.function}-{self.version}"

    def label(self, repeat: int) -> str:
        return f"{self.case}/{self.version}/{repeat}"


class Usage(_Record):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class StoredResponse(_Record):
    """The parts of the API response the evaluation needs; the rest is dropped."""

    id: str | None = None
    model: str
    content: str
    finish_reason: str | None = None


class StoredPrices(_Record):
    """Published USD-per-token prices, kept only when the provider sent no cost."""

    model: str
    prompt: str
    completion: str
    fetched_at: datetime


class CassetteEntry(_Record):
    key: str
    repeat: int = Field(ge=0)
    tag: CallTag | None = None
    request: dict[str, Any]
    response: StoredResponse
    usage: Usage
    cost_usd: float = Field(ge=0)
    cost_source: Literal["provider", "published_prices"]
    prices: StoredPrices | None = None
    latency_ms: float = Field(ge=0)
    requested_at: datetime
    recorded_at: datetime

    @property
    def file_stem(self) -> str:
        return UNTAGGED_FILE_STEM if self.tag is None else self.tag.file_stem


class CassetteStore:
    """All cassette files in one directory, indexed by request key.

    Files are loaded once when the store is created. Appends write one line
    and flush it to disk, so an interrupted recording keeps every call that
    finished and a rerun continues from there.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._entries: dict[str, CassetteEntry] = {}
        for path in sorted(self.root.glob("*.jsonl")):
            for entry in self._read(path):
                if entry.key in self._entries:
                    raise CassetteError(f"{path.name}: key {entry.key} is recorded twice")
                self._entries[entry.key] = entry

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def __iter__(self) -> Iterator[CassetteEntry]:
        return iter(self._entries.values())

    def get(self, key: str) -> CassetteEntry | None:
        return self._entries.get(key)

    def append(self, entry: CassetteEntry) -> Path:
        if entry.key in self._entries:
            raise CassetteError(f"key {entry.key} is already recorded")
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{entry.file_stem}.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(entry.model_dump_json() + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        self._entries[entry.key] = entry
        return path

    @staticmethod
    def _read(path: Path) -> Iterator[CassetteEntry]:
        with path.open(encoding="utf-8") as fh:
            for number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    yield CassetteEntry.model_validate_json(line)
                except ValidationError as exc:
                    raise CassetteError(f"{path.name}:{number}: not a valid entry: {exc}") from None
