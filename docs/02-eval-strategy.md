# Evaluation strategy

Every answer goes through layers of checks, cheapest and most exact first. Each layer answers one question, so a failure says what went wrong, not only that something did.

## The layers, in order

| Layer | The question it answers | How | What it runs on |
|---|---|---|---|
| Retrieval | Did the search return the documents that hold the answer? | the expected documents among the retrieved ones | RAG cases with expected documents |
| Deterministic | Does the output follow rules a program can decide? | RAG: text present, citations name retrieved documents only, no listed forbidden claim, within the length limit, an honest "I don't know" on unanswerable questions. Triage: valid JSON, required fields, the schema, allowed values | every answer |
| Reference | Does the output match the authored expectation? | RAG: the required facts, after normalising numbers, currency, ranges and wording. Triage: category, priority and order id against the guideline's labels | cases with facts or labels |
| Safety | Did anything leak, and did the attack work? | rules: trap contents, the injected offer, the system prompt, internal notes and personal data on every answer; off-topic refusals, invented terms and policy bypasses on the safety cases | every RAG answer |
| Judge | Is the answer grounded, helpful and polite? | a second model with a rubric ([`rubrics/judge.md`](../rubrics/judge.md)), scores from 1 to 5 and a pass rule | RAG answers, except the safety cases |
| Stability | Do repeated runs reach the same verdicts? | each case runs several times; a case is stable when every repeat has the same verdict on every rule-based check | every repeated case |
| Performance | How slow and how expensive? | latency p50 and p95, tokens and the cost the provider reported | every call |

The code of each layer is in [`llmeval/checks/`](../llmeval/checks/), with a docstring that says what the check measures and what it misses.

## Why the deterministic checks come first

- **They cost nothing.** A rule reads the answer that is already there. No model call, no quota, no wait.
- **They are exact and repeatable.** The same answer always gets the same verdict, so a change in a rate is a change in the answers.
- **They explain themselves.** Every check writes a detail line: which fact is missing, which forbidden claim was made, which document was cited without being retrieved.
- **They are fair both ways.** A detector that fails honest answers is as wrong as one that passes invented ones. Before the recording, each detector was tried on honest and on invented answers, and was changed until both kinds came out right. What remains is listed in [docs/05](05-limitations.md).

Retrieval comes before everything else for one more reason: when the right document never reached the model, the answer cannot be right, and the failure belongs to the search, not to the prompt.

## When a judge is needed, and when it is not

Rules check what someone listed: the required facts, the forbidden claims, the trap contents. They cannot tell whether a sentence in new words is supported by the documents, whether the answer helps, or whether it is polite. That is the judge's work: groundedness, helpfulness and tone, graded by a second model from another vendor ([docs/03](03-judge-validation.md)).

The judge does not grade safety, required facts or the format. Those have exact answers, and rules give them for free. Safety is never left to the judge on purpose: an attack that can steer the assistant could steer a model that reads the same text, and a safety verdict should not depend on a second model's mood.

The judge also compares the two prompt versions directly: it sees both answers to the same question and picks the better one, or a tie. It is asked twice, with the order swapped, because judges tend to prefer one position ([docs/03](03-judge-validation.md)).

## What each layer costs

<!-- cost:start -->

| Calls | Count | Mean tokens in / out | Latency p50 / p95 | Cost |
|---|---:|---:|---:|---:|
| rag v1 answers (system) | 156 | 561 / 87 | 1.8 s / 25.3 s | $0.00 (free model) |
| rag v2 answers (system) | 156 | 886 / 41 | 0.9 s / 25.3 s | $0.00 (free model) |
| triage v1 answers (system) | 120 | 513 / 56 | 0.9 s / 13.0 s | $0.00 (free model) |
| triage v2 answers (system) | 120 | 959 / 56 | 1.0 s / 10.7 s | $0.00 (free model) |
| Judge grades of rag v1 | 40 | 1509 / 668 (588 reasoning) | 7.2 s / 15.2 s | $0.00 (free model) |
| Judge grades of rag v2 | 40 | 1468 / 615 (530 reasoning) | 5.5 s / 12.3 s | $0.00 (free model) |
| Pairwise questions, rag v1 vs v2 | 76 | 1571 / 898 (835 reasoning) | 7.4 s / 26.3 s | $0.00 (free model) |

The rule-based layers (retrieval, deterministic, reference, safety and stability) call no model: they read the answers above, so they add no calls and no cost.

The rows add up to 708 calls; the recording holds 706. When both prompt versions wrote the same answer, they share one recorded grading, and each version's row counts it.

<!-- cost:end -->

A judge call is longer and slower than an answer: it reads the rubric, the documents and the answer, and it reasons before its verdict. That is why the judge grades the first run of each case only (`judge_repeats: first` in [`config/models.yaml`](../config/models.yaml)): the stability layer uses the rule-based checks, so grading every repeat would buy calls and nothing the reports use.

On a paid model the same table is the bill. `make estimate` prices the calls still to record from OpenRouter's published prices before anything is sent, and `make record` and `make live` refuse to start above `MAX_RUN_COST_USD`.

## Stability

The system model runs at a non-zero temperature, so one case can pass on one run and fail on the next. Every case runs several times (`repeats` in [`config/models.yaml`](../config/models.yaml)). A case is stable when every repeat has the same verdict on every deterministic, reference and safety check, and, for triage, the same predicted category and priority. Retrieval is left out because the search does not use the model. The judge is left out because its verdict would measure the judge, not the system.

## From results to a verdict

Three mechanisms turn the measurements into a pass or a fail in CI.

**Replay is exact.** `make eval` replays the cassettes and writes [`results/`](../results/). The results are a pure function of the recorded answers: a repository test replays the run again and requires the same bytes.

**The baseline and the per-case tests.** [`results/baseline.json`](../results/baseline.json) is the accepted state: each case's known failures and the key rates. `make baseline` writes it from the results, and a repository test rebuilds it and fails on any hand edit. An evaluation outcome is a measurement, not a broken build, so the per-case tests in [`tests/eval/`](../tests/eval/) compare with the baseline:

- a case that passed in the baseline must still pass;
- a case that failed is an expected failure, and the test names the failed checks;
- a known failure that now passes fails the build as a strict unexpected pass, until the baseline is updated on purpose.

So a prompt change that fixes cases shows up as clearly as one that breaks them, and updating the baseline is a reviewed diff.

**The gate.** `make gate` compares the key rates of a results directory with the baseline. A rate may fall by its tolerance in [`config/gate.yaml`](../config/gate.yaml); a rise always passes; any new safety failure fails at once.

<!-- gate:start -->

| Gated rate | Allowed drop | Largest move between single repeats |
|---|---:|---:|
| All checks | 5.0 pp | 5.8 pp (rag v1) |
| Retrieval layer | 0.0 pp | 0.0 pp (rag v1) |
| Deterministic layer | 5.0 pp | 5.8 pp (rag v1) |
| Reference layer | 5.0 pp | 3.0 pp (rag v1) |
| Safety layer | 2.0 pp | 1.9 pp (rag v1) |
| Judge layer | 10.0 pp | — |
| Category accuracy | 5.0 pp | 2.5 pp (triage v1) |
| Priority accuracy | 5.0 pp | 2.5 pp (triage v2) |
| Stable cases | 10.0 pp | — |
| Pass by the rubric rule | 10.0 pp | — |
| Valid judge verdicts | 5.0 pp | — |
| Pairwise position consistency | 15.0 pp | — |
| Pairs with two valid verdicts | 5.0 pp | — |
| New safety failures | none allowed | — |

Allowed drop: how far a rate may fall below the baseline before the gate fails (config/gate.yaml); a rise always passes. Largest move: the widest gap between the same rate on single repeats of the recorded run, over every function and version, with the version it was seen in.

A dash: the rate is measured once per run, so single repeats cannot be compared. The judge grades the first run of each case only, the pairwise question is asked once per case, and the stable share needs every repeat at once.

<!-- gate:end -->

Why these tolerances:

- **A replay always equals the baseline**, so in CI the gate passes or fails exactly. The tolerances matter for a live run, where the models answer again ([`live.yml`](../.github/workflows/live.yml)).
- **Rates over every repeat** (all checks, the deterministic and reference layers, the triage accuracies) are set near the widest gap the recording showed between its own single repeats, the right-hand column above. A live run averages all its repeats, and an average moves less than a single repeat does, so a tolerance can sit a little below that gap.
- **Retrieval may not move at all.** The search does not use the model, so any change is a change in the code or the knowledge base.
- **Rates measured once** (the judge's pass rate and valid verdicts, pairwise consistency) are over few answers, so one changed verdict moves them by several points. Their tolerances allow a few answers to change, not a trend.
- **Safety is gated twice.** The safety rate may move only as far as the one known unstable safety failure (rag-050 on v1) moves it between repeats, so that case does not fail a live run by chance. Any safety check a case fails that its baseline entry does not list fails the gate, whatever the rates say.

## The three modes

| Mode | Command | Calls the API | Writes |
|---|---|---|---|
| replay | `make eval` (and CI) | never: a missing recording is an error | `results/` |
| record | `make record` | yes, once per call; recorded calls are skipped, so a stopped run continues | `cassettes/` |
| live | `make live`, or [`live.yml`](../.github/workflows/live.yml) by hand | yes, every call again | `results-live/` (ignored by git) |

A live run followed by `llmeval gate --results-dir results-live` shows the drift since the recording: free models are updated, rerouted and retired without notice.

## Reading the Allure report

The report groups the per-case tests by function (epic), layer (feature) and case category (story). A case is listed under each layer it fails, or under every layer it was checked on when it passes, so the status under a layer is that layer's own. The counts on the Behaviors tab are therefore tests, not layer pass rates. The layer tests, under the story "pass rate", give each layer's rate, the same rates as the main table in the README. The report's categories mark a regression against the baseline, a known failure, a known failure that now passes, and pending runs.
