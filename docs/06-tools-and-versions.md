# Tools and versions

Every version and model id on this page was checked on 2026-10-05.

## Python and libraries

Python 3.12 (`requires-python = ">=3.12,<3.13"`); the checks ran on 3.12.14. Each dependency in [`pyproject.toml`](../pyproject.toml) has a lower bound and an upper bound below its next major version, so a new major release cannot change a run without a commit.

| Library | Installed | Declared | Used for |
|---|---|---|---|
| fastapi | 0.142.2 | `>=0.142,<1` | the Toolshop service |
| uvicorn | 0.54.0 | `>=0.54,<1` | serving it, also in the HTTP tests on a loopback port |
| httpx | 0.28.1 | `>=0.28,<1` | the OpenRouter client and the HTTP tests |
| pydantic | 2.13.5 | `>=2.13,<3` | records, the config and validation of model replies |
| pyyaml | 6.0.3 | `>=6.0,<7` | the config files |
| typer | 0.27.2 | `>=0.27,<1` | the `llmeval` command line |
| pytest | 9.1.1 | `>=9.1,<10` (dev) | the tests and the per-case evaluation |
| allure-pytest | 2.16.2 | `>=2.16,<3` (dev) | the Allure results |
| ruff | 0.16.10 | `>=0.16,<1` (dev) | lint |

- The build backend is setuptools (`>=84.0,<85`).
- CI builds the report with the Allure command line 2.30.0, pinned in [`ci.yml`](../.github/workflows/ci.yml).
- The workflows use `actions/checkout@v7`, `actions/setup-python@v7`, `actions/upload-artifact@v7`, `actions/download-artifact@v8`, `actions/upload-pages-artifact@v5` and `actions/deploy-pages@v5`.
- There is no retrieval library: BM25 is written in [`app/retrieval.py`](../app/retrieval.py).

## Models

The model ids live in [`config/models.yaml`](../config/models.yaml) and nowhere in the code. Both are OpenRouter free variants (ids ending in `:free`).

| Role | Model id | Settings | Checked on 2026-10-05 |
|---|---|---|---|
| system | `qwen/qwen3.8-27b:free` | temperature 0.2, max_tokens 600, reasoning off; no seed and no enforced output format, because the model lists neither `seed` nor `response_format` | listed by `GET https://openrouter.ai/api/v1/models` at price 0, with `structured_outputs` but without `response_format` and `seed` |
| judge | `nvidia/nemotron-3-super-120b-a12b:free` | temperature 0, seed 7, max_tokens 4096, reasoning effort low, a strict JSON schema with `require_parameters` | listed at price 0, with `structured_outputs`, `response_format` and `seed` |

The run these results come from is named in the line under the README's main table: the models, the date and the number of calls, read from [`cassettes/manifest.json`](../cassettes/manifest.json).

OpenRouter's free variants are limited to 20 requests per minute and 50 requests per day, or 1000 requests per day once an account has bought at least 10 USD of credits ([limits](https://openrouter.ai/docs/api-reference/limits), checked on 2026-10-05). Recording keeps to 18 requests per minute (`rpm` in the config) and continues over several days when the daily allowance runs out.

## How the concepts map to other evaluation tools

This repository does not use, import or integrate any of the tools below. The table maps its concepts to theirs, so a reader who knows one of them can find their way here. Each mapping was checked against the tool's current documentation on 2026-10-05 (links under the table). A dash means this page maps nothing there, not that the tool lacks it.

| This repository | DeepEval | promptfoo | Inspect (UK AISI) | Ragas |
|---|---|---|---|---|
| A case: input and expected behaviour, in `datasets/` | `LLMTestCase`: `input`, `actual_output`, `expected_output`, `retrieval_context` | a test in the config, with its assertions | a `Sample` (`input`, `target`) in a dataset, run by a task | — |
| Deterministic checks: format, citations, forbidden claims | — | deterministic assertions such as `contains`, `regex`, `is-json`, `javascript`, `python`; any of them negated with `not-` | scorers such as `includes()`, `match()`, `pattern()`, `exact()` | — |
| Retrieval layer: the expected document ids among the retrieved ones | — | `context-recall` (model-assisted) | — | `IDBasedContextRecall`: reference context ids found among the retrieved ones |
| Required facts | — | `contains-all`, `icontains-all` | `model_graded_fact()` (a model checks the facts) | — |
| The judge: a rubric with criteria and a pass rule | `GEval`: criteria or evaluation steps, a score from 0 to 1 and a threshold | `llm-rubric`, `g-eval` | `model_graded_qa()` with a custom template | — |
| The judge's groundedness | `FaithfulnessMetric`: claims in the output against the `retrieval_context` | `context-faithfulness` | — | Faithfulness: claims in the response supported by the retrieved context |
| Pairwise comparison of two versions | — | `select-best`: picks the best of the outputs in one test row | — | — |
| Another model as judge, or several | — | the grading provider set with `--grader` or `provider` | an alternate grader `model`, or a panel decided by majority | — |
| Repeats and the stability layer | — | — | `epochs`, with reducers such as `mean`, `mode`, `at_least_k` and `pass_at_k` | — |
| Safety cases | — | red teaming: plugins and strategies such as prompt injection and jailbreaks, with generated attacks | — | — |
| Recorded calls replayed in CI | — | a response cache on disk (on by default, 14 days) | — | — |
| The per-case tests and the gate fail the build | `deepeval test run` with `assert_test()`: a metric below its threshold fails the build | — | — | — |

Where this repository differs on purpose:

- **Every call is recorded, the judge's included,** keyed by a hash of the whole request, and committed. CI replays it exactly, offline and without a key, and a changed prompt or rubric fails loudly instead of reusing an old answer. A response cache saves calls; a recording makes a result reproducible.
- **Safety is decided by rules,** never by a model, and the judge never grades the safety cases.
- **Evaluation outcomes are compared with a baseline case by case,** so a fixed case is reported as clearly as a broken one, and updating the baseline is a reviewed diff.
- **The judge is measured** against the owner's labels and for position and length bias before its grades are read ([docs/03](03-judge-validation.md)).

Sources, read on 2026-10-05:

- DeepEval: [test cases](https://deepeval.com/docs/evaluation-test-cases), [G-Eval](https://deepeval.com/docs/metrics-llm-evals), [Faithfulness](https://deepeval.com/docs/metrics-faithfulness), [unit testing in CI/CD](https://deepeval.com/docs/evaluation-unit-testing-in-ci-cd)
- promptfoo: [assertions and metrics](https://www.promptfoo.dev/docs/configuration/expected-outputs/), [model-graded metrics](https://www.promptfoo.dev/docs/configuration/expected-outputs/model-graded/), [caching](https://www.promptfoo.dev/docs/configuration/caching/), [red teaming](https://www.promptfoo.dev/docs/red-team/)
- Inspect, by the UK AI Security Institute: [overview](https://inspect.aisi.org.uk/), [scorers](https://inspect.aisi.org.uk/scorers.html), [model grading](https://inspect.aisi.org.uk/model-graded.html), [scoring metrics and epochs](https://inspect.aisi.org.uk/metrics.html)
- Ragas: [available metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/), [Faithfulness](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/), [Context Recall](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/)
- OpenRouter: [models API](https://openrouter.ai/api/v1/models), [limits](https://openrouter.ai/docs/api-reference/limits)

## How this was built

Pavel Zhukov Atum built this repository with AI coding tools: Claude Code and Codex in VS Code, the Claude browser extensions, and the superpowers skills for planning, test-first work and code review. Every change went through the tests and a review. The model answers in [`cassettes/`](../cassettes/) come only from the repository's own `make record`, and the human labels only from the owner's `make label`.
