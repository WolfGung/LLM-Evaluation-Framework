"""The command line: `llmeval` (or `python -m llmeval`).

- `eval`: replay the recorded run, judge calls included, and write
  `results/`. Without `cassettes/manifest.json` it prints "pending first
  recorded run" and writes nothing. It always replays, whatever LLMEVAL_MODE
  says: recording (with the budget guard and the free-quota handling) is
  `make record`.
- `retrieval`: the retrieval layer over the RAG dataset, offline. BM25 is
  deterministic, so this needs no model and no recording.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from app.assistant import DEFAULT_K
from app.retrieval import search
from llmeval.cassettes import CassetteError, CassetteStore, load_manifest
from llmeval.checks.judge import RUBRIC_PATH, RubricError, load_rubric
from llmeval.checks.retrieval import retrieval_recall
from llmeval.client import MissingRecording, ModelClient
from llmeval.config import DEFAULT_MODELS_PATH, ConfigError, Mode, load_config
from llmeval.datasets import (
    DATASETS_DIR,
    RAG_PATH,
    DatasetError,
    file_sha256,
    load_rag,
    load_triage,
)
from llmeval.results import RESULTS_DIR, FunctionResults, PairwiseResults
from llmeval.runner import CASSETTES_DIR, EVAL_FUNCTIONS, run

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Evaluate the Toolshop LLM features from recorded calls.",
)


def _fail(message: str) -> typer.Exit:
    typer.echo(message, err=True)
    return typer.Exit(code=1)


def _line(result: FunctionResults) -> str:
    parts = [
        f"{layer} {rate.passed}/{rate.total} ({rate.rate:.1%})"
        for layer, rate in result.summary.layers.items()
        if rate.rate is not None
    ]
    total = result.summary.all_checks
    parts.append(f"all checks {total.passed}/{total.total}")
    return f"{result.function} {result.version}: " + ", ".join(parts)


def _pairwise_line(result: PairwiseResults) -> str:
    first, second = result.versions
    counts = ", ".join(f"{outcome} {n}" for outcome, n in result.summary.outcomes.items())
    return f"{result.function} {first} vs {second}: {counts}"


@app.command("eval")
def eval_command(
    function: Annotated[
        list[str] | None,
        typer.Option("--function", help=f"Function to evaluate ({', '.join(EVAL_FUNCTIONS)})."),
    ] = None,
    config: Annotated[Path, typer.Option(help="Model config file.")] = DEFAULT_MODELS_PATH,
    datasets_dir: Annotated[Path, typer.Option(help="Directory with the datasets.")] = DATASETS_DIR,
    cassettes_dir: Annotated[Path, typer.Option(help="Recorded calls.")] = CASSETTES_DIR,
    results_dir: Annotated[Path, typer.Option(help="Where results go.")] = RESULTS_DIR,
    rubric: Annotated[Path, typer.Option(help="The judge rubric.")] = RUBRIC_PATH,
) -> None:
    """Replay the recorded run, apply every check and write results/."""
    functions = function or list(EVAL_FUNCTIONS)
    if unknown := [f for f in functions if f not in EVAL_FUNCTIONS]:
        raise typer.BadParameter(
            f"unknown function: {', '.join(unknown)} (known: {', '.join(EVAL_FUNCTIONS)})",
            param_hint="--function",
        )
    try:
        manifest = load_manifest(cassettes_dir)
        if manifest is None:
            typer.echo("pending first recorded run")
            return
        replay_config = load_config(config, env={})
        manifest.check_models(
            system=replay_config.models.system.model, judge=replay_config.models.judge.model
        )
        rag_path, triage_path = datasets_dir / "rag.jsonl", datasets_dir / "triage.jsonl"
        current = {p.name: file_sha256(p) for p in (rag_path, triage_path) if p.is_file()}
        if notice := manifest.dataset_notice(current):
            typer.echo(notice)
        rag_cases = load_rag(rag_path) if "rag" in functions else ()
        triage_cases = load_triage(triage_path) if "triage" in functions else ()
        judge_rubric = load_rubric(rubric)
        client = ModelClient(Mode.REPLAY, CassetteStore(cassettes_dir), replay_config)
        outcome = run(
            client,
            replay_config.models.system,
            judge=replay_config.models.judge,
            rubric=judge_rubric,
            mode=Mode.REPLAY,
            cassettes_dir=cassettes_dir,
            results_dir=results_dir,
            rag_cases=rag_cases,
            triage_cases=triage_cases,
            dataset_paths={"rag": rag_path, "triage": triage_path},
            versions={f: v for f, v in manifest.prompt_versions.items() if f in EVAL_FUNCTIONS},
            repeats=manifest.repeats,
            stability_cases=manifest.stability_cases,
        )
    except (
        MissingRecording,
        CassetteError,
        ConfigError,
        DatasetError,
        RubricError,
        OSError,
    ) as exc:
        raise _fail(str(exc)) from None
    lines = [_line(result) for result in outcome.results]
    lines += [_pairwise_line(result) for result in outcome.pairwise]
    for line, path in zip(lines, outcome.written, strict=True):
        typer.echo(line)
        typer.echo(f"  wrote {path}")


@app.command("retrieval")
def retrieval_command(
    dataset: Annotated[Path, typer.Option(help="RAG dataset.")] = RAG_PATH,
    k: Annotated[int, typer.Option(min=1, help="Documents retrieved per question.")] = DEFAULT_K,
) -> None:
    """Show which expected documents the search returns for each RAG case."""
    try:
        cases = [case for case in load_rag(dataset) if case.expected_docs]
    except DatasetError as exc:
        raise _fail(str(exc)) from None
    complete = expected_total = found_total = 0
    for case in cases:
        retrieved = [hit.doc_id for hit in search(case.question, k)]
        check = retrieval_recall(case.expected_docs, retrieved)
        complete += check.passed
        expected_total += len(case.expected_docs)
        found_total += sum(doc in retrieved for doc in case.expected_docs)
        mark = "ok  " if check.passed else "MISS"
        typer.echo(f"{case.id}  {mark}  {check.detail}; got {', '.join(retrieved) or 'nothing'}")
    share = found_total / expected_total if expected_total else 0.0
    typer.echo(f"{found_total}/{expected_total} expected documents retrieved ({share:.1%})")
    typer.echo(f"{complete}/{len(cases)} cases with every expected document retrieved (k={k})")


def main() -> None:
    app()
