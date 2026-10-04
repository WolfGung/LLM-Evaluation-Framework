"""Cassettes: recorded model calls stored as JSONL, looked up by a request key.

A cassette entry holds exactly what was sent (the request body) and what came
back (answer, usage, cost, latency, timestamps). It has no field for headers,
so the API key has nowhere to go.

The key is a sha256 of the canonical JSON of the whole request body plus the
repeat index. Repeats are part of the key on purpose: three runs of one
request are three separate recordings, which is what the stability layer
measures.

A recorded run is declared by `manifest.json` in the cassette directory, which
`record` writes only when every planned call is recorded. The directory alone
proves nothing: it always exists (it holds a `.gitkeep`), and an interrupted
recording leaves cassette files without a manifest. Without the manifest the
evaluation is "pending first recorded run" and writes no results.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

# Every top-level field a request body may carry. The key hashes the whole
# body, and a field outside this list is refused, so a new request parameter
# has to be added here on purpose instead of being sent but not keyed.
REQUEST_FIELDS = frozenset(
    {
        "model",
        "messages",
        "temperature",
        "seed",
        "max_tokens",
        "response_format",
        "provider",
        "reasoning",
    }
)

UNTAGGED_FILE_STEM = "untagged"

MANIFEST_FILE = "manifest.json"
PENDING_RECORDED_RUN = "pending first recorded run"

# Where a call's cost came from: OpenRouter's usage.cost, the published
# prices, or nowhere (the price lookup failed, but the call is still kept).
CostSource = Literal["provider", "published_prices", "unknown"]
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class CassetteError(RuntimeError):
    """A cassette file is broken or a write would corrupt the store."""


class ManifestMismatch(CassetteError):
    """The recorded run does not match the current configuration."""


def utc_now() -> datetime:
    return datetime.now(UTC)


def canonical_json(value: Any) -> str:
    """JSON with sorted keys and no spaces, so equal data gives equal text."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def request_key(body: Mapping[str, Any], repeat: int) -> str:
    """sha256 of the whole request body plus the repeat index."""
    unknown = sorted(set(body) - REQUEST_FIELDS)
    if unknown:
        raise ValueError(f"request field not covered by the cassette key: {', '.join(unknown)}")
    # An unset option (None at the top level) is not part of the request. None
    # deeper down is a real value, such as "default": null in a JSON schema.
    present = {name: value for name, value in body.items() if value is not None}
    material = {"body": present, "repeat": repeat}
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
    """Token counts reported by the API.

    `reasoning_tokens` is the part of `completion_tokens` the model spent
    thinking; a large share with an empty answer means the budget ran out.
    """

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)

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
    cost_usd: float | None = Field(ge=0)
    cost_source: CostSource
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


Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
RoleName = Literal["system", "judge"]
VersionList = Annotated[tuple[str, ...], Field(min_length=1)]


class RunManifest(_Record):
    """What a complete recording covers. `record` writes it; nothing else does.

    - `models`: the model id per role (`system` and `judge`).
    - `prompt_versions`: the prompt versions per evaluated function
      (`rag`, `triage`).
    - `repeats`: runs per case.
    - `datasets`: sha256 of each dataset file's bytes, by file name. This is
      information, not a lock: editing expected facts keeps every recording
      valid, while a changed question changes its key and fails replay loudly.
    - `recorded_from` / `recorded_to`: the first and the last recorded call.
    - `planned_calls` / `recorded_calls`: the call plan and the calls in the
      cassettes for it. A manifest exists only for a complete recording, so
      `recorded_calls` is never below `planned_calls`.
    - `stability_cases`: the case ids that run `repeats` times for the
      stability layer; every other case runs once. None means every case runs
      `repeats` times. A subset lets a first recording fit a small daily quota.
    """

    schema_version: Literal[1] = 1
    models: dict[RoleName, str]
    prompt_versions: Annotated[dict[str, VersionList], Field(min_length=1)]
    repeats: int = Field(ge=1)
    datasets: Annotated[dict[str, Sha256], Field(min_length=1)]
    recorded_from: datetime
    recorded_to: datetime
    planned_calls: int = Field(ge=1)
    recorded_calls: int = Field(ge=0)
    stability_cases: Annotated[tuple[str, ...], Field(min_length=1)] | None = None

    @model_validator(mode="after")
    def _complete_and_ordered(self) -> RunManifest:
        if set(self.models) != {"system", "judge"}:
            raise ValueError("models must name the system and the judge model")
        if self.recorded_to < self.recorded_from:
            raise ValueError("recorded_to is before recorded_from")
        if self.recorded_calls < self.planned_calls:
            raise ValueError(
                f"only {self.recorded_calls} of {self.planned_calls} planned calls are recorded; "
                "a manifest is written only for a complete recording"
            )
        return self

    def repeats_for(self, case_id: str) -> int:
        """How many times case `case_id` was recorded."""
        if self.stability_cases is None or case_id in self.stability_cases:
            return self.repeats
        return 1

    def check_models(self, *, system: str, judge: str) -> None:
        """Raise `ManifestMismatch` when the config names other models than the recording."""
        for role, configured in (("system", system), ("judge", judge)):
            recorded = self.models[role]
            if recorded != configured:
                raise ManifestMismatch(
                    f"{role} model: recorded with {recorded}, config says {configured}: "
                    "re-record or restore the config"
                )

    def changed_datasets(self, current: Mapping[str, str]) -> list[str]:
        """Dataset files whose current sha256 differs from the recorded one."""
        return [
            name
            for name, recorded in self.datasets.items()
            if name in current and current[name] != recorded
        ]

    def dataset_notice(self, current: Mapping[str, str]) -> str | None:
        """One line naming changed datasets, or None when nothing changed."""
        changed = self.changed_datasets(current)
        if not changed:
            return None
        return (
            f"notice: datasets changed since the recording: {', '.join(changed)} "
            "(expectations are re-checked; a changed input fails replay)"
        )


def load_manifest(root: Path | str) -> RunManifest | None:
    """The manifest of the recorded run in `root`, or None when there is none.

    Only `manifest.json` counts. A broken manifest raises `CassetteError`
    instead of being treated as absent, so a damaged recording cannot turn
    the evaluation into a silent skip.
    """
    path = Path(root) / MANIFEST_FILE
    if not path.is_file():
        return None
    try:
        return RunManifest.model_validate_json(path.read_bytes())
    except ValidationError as exc:
        raise CassetteError(f"{path.name}: not a valid run manifest: {exc}") from None
