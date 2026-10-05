# LLM Evaluation Framework

Layered checks that show in CI what a prompt or model change made better or worse in two LLM features, a support assistant and ticket triage, replayed from real recorded calls so every run is free.

[![CI](https://github.com/WolfGung/LLM-Evaluation-Framework/actions/workflows/ci.yml/badge.svg)](https://github.com/WolfGung/LLM-Evaluation-Framework/actions/workflows/ci.yml)
[![live report](https://img.shields.io/badge/live%20report-GitHub%20Pages-brightgreen)](https://wolfgung.github.io/LLM-Evaluation-Framework/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue)](pyproject.toml)
[![license: MIT](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

<!-- results:start -->

| Metric | rag v1 | rag v2 | triage v1 | triage v2 |
|---|---:|---:|---:|---:|
| All checks | 77.6% | 86.5% | 65.0% | 89.2% |
| Retrieval layer | 81.8% | 81.8% | — | — |
| Deterministic layer | 92.3% | 96.2% | 97.5% | 97.5% |
| Reference layer | 80.8% | 81.8% | 65.0% | 89.2% |
| Safety layer | 95.5% | 100.0% | — | — |
| Judge layer | 90.0% | 92.5% | — | — |
| Safety cases with no safety failure | 9 of 12 | 12 of 12 | — | — |
| Stable cases | 86.5% | 100.0% | 92.5% | 97.5% |
| Cost (system + judge calls) | $0.00 (free models) | $0.00 (free models) | $0.00 (free models) | $0.00 (free models) |
| Latency p50 / p95 (system calls) | 1.8 s / 25.3 s | 0.9 s / 25.3 s | 0.9 s / 13.0 s | 1.0 s / 10.7 s |

Each column is one prompt version. Each case ran 3 times; each layer's rate is over the runs that layer checks (the judge graded the first run of each judged case). A safety case has no safety failure when every safety check passed on every run. The pairwise comparison calls are not counted in any column. Recorded on 2026-10-05 (UTC) with qwen/qwen3.8-27b:free (system) and nvidia/nemotron-3-super-120b-a12b:free (judge), 706 calls.

<!-- results:end -->

How to read the table: rag is the support assistant (retrieval-augmented generation: it searches the knowledge base, then answers), and triage turns a ticket into JSON. Retrieval: the search found the expected documents. Deterministic: rules a program can decide, such as valid JSON, citations, forbidden claims and length. Reference: the required facts, or the expected category, priority and order id. Safety: nothing leaked and no attack worked. Judge: a second model graded the answer as grounded, helpful and polite. Stable cases: every repeat got the same verdicts. p50 / p95: half the calls were faster than the first figure, and 95 percent were faster than the second.

## What this shows

- **Catch regressions when you change a prompt or a model.** Every case is compared with an accepted baseline. A case that breaks fails the build. A case that the change fixes also fails the build until the baseline is updated on purpose, so improvements are reviewed too. A gate fails the build when a key rate drops by more than its tolerance. The table above compares two prompt versions of the same two features.
- **Test a chatbot against prompt injection and data leaks.** Rules check every answer for a customer's personal data, internal notes, the system prompt and an instruction planted in a document. Dedicated cases attack the assistant directly ([docs/04](docs/04-safety-cases.md)).
- **Know when an LLM judge can be trusted.** The judge's verdicts are compared with a person's blind labels, with percent agreement and Cohen's kappa (agreement beyond chance). When it compares two versions, it is asked twice with the answers swapped. It is checked for a bias toward longer answers, and it comes from a different vendor than the model it grades ([docs/03](docs/03-judge-validation.md)).
- **Keep evaluation in CI without paying for every run.** Every model call, the judge's included, was recorded once from free models. CI replays the recording offline, with no key, and gets the same result every time. A live run, started by hand, measures the drift since the recording.

## The two prompt versions, compared by the judge

<!-- pairwise:start -->

The judge compared the first answers of rag v1 and rag v2 case by case, asked twice with the order of the two answers swapped.

| Outcome over 40 cases | Cases |
|---|---:|
| v1 preferred in both orders | 7 |
| v2 preferred in both orders | 4 |
| A tie in both orders | 9 |
| Inconsistent: the two orders disagree | 18 |
| Identical answers, not compared | 2 |

Position consistency: 20 of 38 compared pairs (52.6%) got the same verdict in both orders.

In 18 of the 38 compared pairs, the judge's preference changed when the two answers swapped places: 3 times it chose the answer shown first in both orders, and 15 times it called a tie in one order and chose a side in the other. An inconsistent pair is never settled by picking one order.

With this many flips, the comparison says more about the judge's position bias than about the two prompts, so it picks no winner. The main table rests on the rules and the per-answer grades.

<!-- pairwise:end -->

## Can the judge be trusted?

<!-- agreement:start -->

| Judge's verdict | Author: pass | Author: fail |
|---|---:|---:|
| Judge: pass | 22 | 1 |
| Judge: fail | 5 | 2 |

Percent agreement: 24 of 30 (80.0%). Cohen's kappa: 0.30. Disagreements: 6, listed in results/judge-agreement.json with the judge's reasons.

Labelled: 30 of 30 sample answers.

Pavel Zhukov Atum, the author, labelled 30 judged answers by hand, blind to the judge's verdict (make label): all 7 answers the judge failed and 23 it passed. The sample oversamples judge failures, so agreement on it is not the agreement over all answers.

<!-- agreement:end -->

How the sample is drawn, the judge's grades, and its position and length checks: [docs/03](docs/03-judge-validation.md).

## What evaluation cannot tell you

A few dozen cases, one recording day, one knowledge base and one person's labels can show a difference between two prompts. They cannot prove an assistant safe, measure a rate to the decimal, or see a model change after the recording. The detectors match the wordings their authors foresaw, and the judge is trusted only as far as its agreement with a person goes. [docs/05](docs/05-limitations.md) lists the limits a reader of these results should know, detector by detector; each check's docstring has the full list.

## How to run

Python 3.12. Replay needs no key and no network.

```bash
make install    # .venv with the package and its dev tools
make test       # unit, app and repository tests
make eval       # replay the recording into results/, then compare every case with the baseline
make gate       # the key rates of results/ against results/baseline.json
make readme     # write the generated blocks of README.md and docs/ from results/
make lint       # ruff
```

The service, in Docker, answers from the recorded calls:

```bash
docker compose up --build
curl -s localhost:8000/assist -H 'content-type: application/json' \
  -d '{"question": "How many days do I have to return something?", "version": "v2"}'
```

In replay mode it answers only the recorded dataset questions. With `LLMEVAL_MODE=live` and a key it calls the free models for any question.

Recording and live runs need an OpenRouter key in the environment, never in a file in git:

```bash
cp .env.example .env              # then put the key in .env
set -a; . ./.env; set +a
make estimate                     # the call plan, the free-quota days and the cost, before any call
make record                       # record every planned call; rerun to continue after a stop
make live                         # every case again against the API, into results-live/
.venv/bin/llmeval gate --results-dir results-live   # the drift since the recording
```

`make label` is the author's blind labelling tool for the judge-agreement sample.

## How the repository is put together

```text
datasets/ ─► runner ─► ModelClient ─► OpenRouter          (record, live)
               │            └───────► cassettes/          (replay)
               ▼
   retrieval ─► deterministic ─► reference ─► safety ─► judge    (+ stability, performance)
               ▼
           results/ ─► baseline and gate ─► CI
               └────► tools/render.py, tools/site.py ─► README.md, docs/, the Pages site
```

- [`app/`](app/): the system under test, a FastAPI service with the support assistant (BM25 search over [`app/kb/`](app/kb/), then the model) and the ticket triage, and two prompt versions of each in [`app/prompts/`](app/prompts/).
- [`llmeval/`](llmeval/): the evaluation. [`client.py`](llmeval/client.py) makes every call in one of three modes and [`cassettes.py`](llmeval/cassettes.py) stores them; [`runner.py`](llmeval/runner.py) runs the cases; [`checks/`](llmeval/checks/) holds the layers; [`stability.py`](llmeval/stability.py) and [`perf.py`](llmeval/perf.py) measure repeats, latency, tokens and cost; [`baseline.py`](llmeval/baseline.py) and [`gate.py`](llmeval/gate.py) turn results into a verdict; [`recording.py`](llmeval/recording.py), [`callplan.py`](llmeval/callplan.py), [`pricing.py`](llmeval/pricing.py) and [`quota.py`](llmeval/quota.py) keep recording inside the budget and the free limits; [`labels.py`](llmeval/labels.py) and [`agreement.py`](llmeval/agreement.py) compare the judge with the author's labels; [`cli.py`](llmeval/cli.py) is the `llmeval` command.
- [`datasets/`](datasets/): the authored cases and the triage guideline. [`rubrics/judge.md`](rubrics/judge.md): the judge's rubric.
- [`cassettes/`](cassettes/): every recorded call, with its manifest. [`results/`](results/): the replay results, the baseline and the judge agreement, all committed. [`labels/`](labels/): the label sample and the author's labels.
- [`tests/`](tests/): `unit/` on synthetic data, `app/` for the service, `repo/` for the repository's own promises (fresh results, the baseline, the generated blocks, the workflows), `eval/` for the per-case and per-layer evaluation.
- [`tools/`](tools/): [`render.py`](tools/render.py) and [`sections.py`](tools/sections.py) write the generated blocks; [`site.py`](tools/site.py) builds the published page.
- [`.github/workflows/`](.github/workflows/): [`ci.yml`](.github/workflows/ci.yml) lints, tests and replays on every push and pull request and publishes the page and the Allure report from `main`; [`live.yml`](.github/workflows/live.yml) is the live run, started by hand.

## Modes

| Mode | Started by | Calls the API | Writes | Needs |
|---|---|---|---|---|
| replay | `make eval`, CI | never; a missing recording is an error | `results/` | nothing |
| record | `make record` | once per call; recorded calls are skipped | `cassettes/` | `OPENROUTER_API_KEY` |
| live | `make live`, [`live.yml`](.github/workflows/live.yml) | every call, every time | `results-live/` (ignored by git) | `OPENROUTER_API_KEY` |

Record and live estimate the cost first and refuse to start above `MAX_RUN_COST_USD`. Recording keeps to the configured requests per minute, stops cleanly on the free daily quota and continues on the next run. In GitHub Actions the live run needs the repository secret `OPENROUTER_API_KEY`; without it the workflow says so in a notice and skips the run, and the CI badge reflects `ci.yml` only.

## Configuration

- [`config/models.yaml`](config/models.yaml): the model id of each role (the system under test and the judge) and its temperature, seed, token budget, reasoning and output format; how often each case repeats (`repeats`, `stability_cases`), which runs the judge grades (`judge_repeats`), and the requests per minute (`rpm`). Model ids live here and nowhere in the code.
- [`config/gate.yaml`](config/gate.yaml): the gate's tolerances, each with its reason.
- [`rubrics/judge.md`](rubrics/judge.md): the judge's criteria, score anchors and pass rule.
- Environment: `OPENROUTER_API_KEY` (record and live only), `MAX_RUN_COST_USD` (the spend limit of a record or live run, default `1.00`), `LLMEVAL_MODE` (the service: `replay` or `live`). [`.env.example`](.env.example) lists them without values.

## Documentation

- [What is evaluated](docs/01-what-is-evaluated.md): the system under test, its two features and two prompt versions, the cases, and the business risks each check covers.
- [Evaluation strategy](docs/02-eval-strategy.md): the layers and their order, when a judge is needed, what each layer costs, the baseline and the gate with its tolerances.
- [Can the judge be trusted?](docs/03-judge-validation.md): agreement with the author's labels, and position, length and self-preference bias.
- [Safety cases](docs/04-safety-cases.md): each attack, the safe answer and the result per prompt version.
- [What evaluation cannot tell you](docs/05-limitations.md): the limits of the dataset, the recording, the judge and every detector.
- [Tools and versions](docs/06-tools-and-versions.md): library versions and model ids with the date checked, and how the concepts map to DeepEval, promptfoo, Inspect and Ragas.

## Licence

MIT, see [`LICENSE`](LICENSE).

## Related work

Six more repositories from the same portfolio:

- **[Toolshop-Test-Automation-Framework](https://github.com/WolfGung/Toolshop-Test-Automation-Framework)** — a test automation framework built from scratch for an online shop: API, browser and end-to-end cases against a public demo shop or a local Docker stand, with test design documents.
- **[Marketplace-Test-Automation-Framework](https://github.com/WolfGung/Marketplace-Test-Automation-Framework)** — API and browser tests for a marketplace shop, run against a small stand shipped in the repository with a nightly drift check of the public demo site, a smoke set, video and traces per browser test and a published Allure report.
- **[Web-Scraping-Automation-Framework](https://github.com/WolfGung/Web-Scraping-Automation-Framework)** — a scraper that collects two practice sites and a demo store of its own, over HTTP and through a browser, detects changes between nightly runs and publishes the data, the change report and the test report.
- **[Test-Suite-Rescue](https://github.com/WolfGung/Test-Suite-Rescue)** — a deliberately sick test suite, its cured version with the same coverage on Playwright and on Selenium, and the measured difference between them against the same application, reproducible with one command.
- **[API-Test-Generator](https://github.com/WolfGung/API-Test-Generator)** — a command-line tool that turns an OpenAPI document or a Postman collection into a runnable pytest suite, with four generated suites committed and proven against a sample API in CI.
- **[Accessibility-Test-Automation-Framework](https://github.com/WolfGung/Accessibility-Test-Automation-Framework)** — an axe-core scan and keyboard-only checks against a shop served in an accessible and a deliberately broken mode, every finding mapped to a WCAG 2.1 AA criterion, with a manual checklist for what automation cannot see.

## Hire me

I take short, well-defined jobs: a test automation framework from scratch, an API test suite for an existing backend, end-to-end tests for a critical flow, fixing flaky tests and reducing run time, setting up CI for existing tests, scrapers and data pipelines. Profile on Guru: [https://www.guru.com/freelancers/pavel-zhukov-atum](https://www.guru.com/freelancers/pavel-zhukov-atum). Time zone: Central European Time (CET/CEST), so my working hours overlap with Central European business hours. I work in writing.
