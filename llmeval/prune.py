"""`llmeval prune`: remove recorded entries that the current plan no longer has.

A changed config gives its calls new keys: for example a larger judge
`max_tokens` changes every judge request body. `record` then records the new
calls, and the old recordings stay in the cassettes. They leave only through
this command; nobody edits a cassette by hand.

The order of work:

1. The lock: the same `.record.lock` as `record`, so prune never runs while a
   record run appends, and the other way round. A dry run only checks the
   lock and creates no file.
2. The plan: computed exactly as `record` and `status` compute it
   (`llmeval.callplan.full_plan`): every system key, and the judge keys
   planned from the recorded answers (identical answers share a grading and
   need no pairwise question). The config, the datasets, the rubric and every
   cassette must load; any error refuses the prune. A judge call that waits
   for answers not yet recorded has no key, so a recorded entry could still
   be its recording: prune refuses then too. So it does while the manifest
   names a prompt version whose prompt file is gone: the plan then lacks
   that version, and its recordings would look unplanned. It never works on
   a partial view.
3. The list: every recorded entry whose key is not in the plan, with its
   file, call (case/version/repeat) and recording time. An unfinished last
   line is not an entry: it follows the store rule (`CassetteStore`).
4. Without `--yes` nothing changes. With `--yes` each affected file is
   rewritten atomically and keeps its other lines byte for byte and in
   order; a file left with no entries is deleted (`CassetteStore.remove`).

Prune never writes or edits the manifest. `llmeval status` says when the
manifest no longer matches the cassettes, and `record` rewrites it once the
current plan is fully recorded. The cassettes are in git, so a removal can be
undone with git.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path

from llmeval.callplan import PlanInputs, count_plan, full_plan, unplanned
from llmeval.cassettes import MANIFEST_FILE, CassetteEntry, CassetteStore, load_manifest
from llmeval.recording import record_lock


class PruneRefused(RuntimeError):
    """The plan is not complete enough to say which entries it no longer has."""


@dataclass(frozen=True)
class PruneOutcome:
    """What prune found and, with `--yes`, removed (entries per file name)."""

    unplanned: tuple[CassetteEntry, ...]
    removed: Mapping[str, int]


def _entries(count: int) -> str:
    return "1 entry" if count == 1 else f"{count} entries"


def _label(entry: CassetteEntry) -> str:
    """The call an entry records: case/version/repeat, or its key without a tag."""
    if entry.tag is None:
        return f"key {entry.key[:12]}/{entry.repeat}"
    return entry.tag.label(entry.repeat)


def _by_file(found: Sequence[CassetteEntry]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in found:
        name = f"{entry.file_stem}.jsonl"
        counts[name] = counts.get(name, 0) + 1
    return counts


def _refuse_missing_versions(inputs: PlanInputs, cassettes_dir: Path) -> None:
    """Refuse while the manifest names a prompt version without a prompt file.

    The current plan has only the versions whose prompt files exist, so the
    recordings of a deleted version would look unplanned and be removed.
    """
    manifest = load_manifest(cassettes_dir)
    if manifest is None:
        return
    for function, versions in manifest.prompt_versions.items():
        if function not in inputs.versions:
            continue
        missing = [v for v in versions if v not in inputs.versions[function]]
        if missing:
            named = ", ".join(f"{function} {version}" for version in missing)
            word, files, its = (
                ("version", "its prompt file is", "its")
                if len(missing) == 1
                else ("versions", "their prompt files are", "their")
            )
            raise PruneRefused(
                f"the manifest names prompt {word} {named}, but {files} gone, so prune would "
                f"remove {its} recordings: restore the file, or record again first"
            )


def prune(
    inputs: PlanInputs,
    cassettes_dir: Path,
    *,
    apply: bool,
    echo: Callable[[str], None] = print,
) -> PruneOutcome:
    """List the recorded entries outside the current plan; remove them when `apply`.

    Raises `RecordLocked` while a record run holds the lock, `PruneRefused`
    while the manifest names a prompt version the plan lacks or judge calls
    wait for their answers, and `CassetteError` for a broken cassette, all
    before any file changes.
    """
    with record_lock(cassettes_dir, create=apply, work="pruning"):
        _refuse_missing_versions(inputs, cassettes_dir)
        store = CassetteStore(cassettes_dir)
        for notice in store.notices:
            echo(f"notice: {notice}")
        plan = full_plan(inputs, store)
        counts = count_plan(plan, store, inputs.models)
        if counts.waiting:
            wait = "calls wait" if counts.waiting > 1 else "call waits"
            raise PruneRefused(
                f"{counts.waiting} judge {wait} for answers that are not recorded, so their keys "
                "are unknown and the plan is not complete: run make record until every answer "
                "is recorded, then prune"
            )
        echo(f"current plan: {counts.total} distinct calls, {counts.recorded} of them recorded")
        found = unplanned(plan, store)
        if not found:
            echo("every recorded entry is in the current plan: nothing to prune")
            return PruneOutcome((), {})
        per_file = ", ".join(f"{name} {n}" for name, n in _by_file(found).items())
        noun = "entry is" if len(found) == 1 else "entries are"
        echo(f"{len(found)} recorded {noun} not in the current plan: {per_file}")
        for entry in found:
            moment = entry.recorded_at.astimezone(UTC)
            echo(
                f"  {entry.file_stem}.jsonl: {_label(entry)}, "
                f"recorded {moment:%Y-%m-%d %H:%M:%S} UTC"
            )
        if not apply:
            them = "it" if len(found) == 1 else "them"
            echo(f"dry run: nothing was changed; make prune (llmeval prune --yes) removes {them}")
            return PruneOutcome(tuple(found), {})
        removed = store.remove({entry.key for entry in found})
        for name, count in removed.items():
            gone = "" if (store.root / name).exists() else " and deleted the file: no entries left"
            echo(f"removed {_entries(count)} from {name}{gone}")
        echo(f"removed {_entries(sum(removed.values()))} in all")
        if (store.root / MANIFEST_FILE).is_file():
            echo(
                f"{MANIFEST_FILE} is unchanged: prune never edits it; llmeval status says whether "
                "make eval can still replay it"
            )
        return PruneOutcome(tuple(found), removed)
