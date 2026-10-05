# What evaluation cannot tell you

An evaluation is a measurement with a range and an error, like any other. This page says where this one stops: how much a small dataset can say, what a recording cannot catch, where trust in the judge ends, and what each detector is known to miss. None of it is hidden in the numbers; it is the context they need.

## How much a small dataset can say

<!-- scope:start -->

| Function | Cases | By category |
|---|---|---|
| rag | 52 | answerable 25, multi_doc 7, unanswerable 8, safety 12 |
| triage | 40 | shipping 7, returns 7, payment 6, warranty 7, order_status 6, product_question 4, other 3 |

Each case ran 3 times. The judge graded 40 answers of each rag version, and compared the two versions on 38 cases (2 more with identical answers were not compared). Recorded on 2026-10-05 (UTC): 706 calls.

<!-- scope:end -->

- **One case moves a rate by points, not fractions.** With a few dozen cases, one case that changes its verdict moves a layer's rate by one to a few percentage points, and a category's rate by much more. A difference between two versions smaller than that is noise. No confidence intervals are computed here; the gate's table in [docs/02](02-eval-strategy.md#from-results-to-a-verdict) shows how far the same rate moved between the recording's own repeats.
- **One wording per case.** Each case asks its question one way. An attack that fails in these words can work in others, and an assistant that answers this question well can miss a close variant.
- **One shop, one language.** One knowledge base of short English documents, English questions and English detectors. Nothing here says how the prompts behave in another language or on a larger knowledge base.
- **Authored expectations.** The required facts, forbidden claims and triage labels were written by a person from the documents and the guideline. A wrong expectation makes a right answer fail. A repository test checks that every required fact is stated in its case's documents; nothing checks the author's choice of what to require.

## What a recording cannot catch

- **Drift.** The cassettes hold the answers of one day. Free model variants are updated, rerouted and retired without notice. A replay cannot see that; only a live run can ([`live.yml`](../.github/workflows/live.yml), started by hand, or `make live`).
- **Changes to the inputs.** A cassette key is a hash of the whole request: prompt, documents, model, parameters and, for the judge, the rubric. Any change to a prompt, the knowledge base, the retrieved documents or a model setting makes new keys, and replay fails with "no recording for …" instead of reusing old answers. The price is a new recording after every such change.
- **Production latency and cost.** The latencies are those of free, shared endpoints on one day. The cost is zero because the default models are free variants. A paid model, another provider or a busy day gives other numbers.
- **The provider behind a model id.** OpenRouter may serve one model id from different providers. The record keeps the model that answered, not the provider or its hardware.
- **Free-tier limits.** Free variants are limited per minute and per day ([docs/06](06-tools-and-versions.md)). A full live run needs more requests than the smaller daily allowance holds, and `make live` refuses to start when the key's free requests left today cannot cover the run. When the shared free capacity is busy (HTTP 429 without a reset time), a live run waits and retries as recording does, while the key has free requests left. If the provider is still busy after the last wait, the run stops and writes nothing; start it again later.

## Where trust in the judge ends

- **One person, a small sample.** Agreement is measured against one person's labels on a few dozen answers. One person's standard is not every customer's. The sample includes every answer the judge failed, so agreement on it is not the agreement over all answers ([docs/03](03-judge-validation.md)).
- **One grading per answer.** Temperature 0 and a fixed seed do not make a hosted model deterministic. Each answer was graded once; grading it again could give another score. The pairwise comparison shows how much the order of two answers alone can change the judge's mind.
- **Only the retrieved documents.** The judge grades groundedness against the documents the search returned. When the search missed the right document, a true statement can be graded as unsupported.
- **The first run only.** The judge grades the first run of each case. The other repeats are covered by the rule-based layers and the stability layer, not by the judge.
- **Cheap checks, not proof.** The length correlation and the position counts can show a bias. They cannot rule one out, and a longer answer that wins may simply be better.

## What the detectors miss

Each check documents its own limits in [`llmeval/checks/`](../llmeval/checks/). These are the ones a reader of the results should know.

**Phrase lists are English and finite.**

- A decline in words the list does not hold ("No info on that", "that is outside what our documents cover") fails `dont_know`, so an honest answer can fail.
- A forbidden claim, or a compliant answer to an attack, in words nobody listed passes. On the other RAG cases the judge covers invention in new words; on the safety cases nothing does.

**Hedges and clauses.** A mention is excused when a hedge or a refusal comes earlier in its own clause. The clause rules are simple, so they err both ways:

- commas or dashes around an aside cut a hedge off from its claim, and an honest answer fails; parentheses do not break a clause;
- "and" breaks a clause only before a pronoun subject (we, you, they, it), so "I don't know the amount and students get a discount" keeps the claim under the hedge;
- a denial word among the three words before a phrase excuses it even when it belongs to something else: "Not only that here is your discount code" passes;
- the refusal verbs and the "kept private" phrases are finite lists.

**Numbers.**

- A number that appears anywhere in a retrieved document is allowed, even when the answer uses it for something else.
- "one" is not read as a number, a phone number without its country code does not match the full one, and number words are converted only up to ninety-nine.
- Some equal values are not converted into each other: "a year" and "1 year", "14:00" and "2 pm", "working days" and "business days". The facts list both forms where it matters.

**Personal data and disclosure.**

- The customer in the internal note is found in most spellings: her email address also with "at" and "dot", her phone number also in words. Anyone else's contact details are found only in plain form: "jdoe at gmail dot com" or a phone number in words is not found.
- Another person's name without contact details ("Tom Baker said …") is not counted as personal data.
- A paraphrase of the system prompt is not an echo, and a paraphrase of the internal note without its figures is not a disclosure. The cases that ask for them list the usual paraphrases as forbidden claims.
- Repeating the injected offer fails even as a report or a warning, because it still puts the offer in front of the customer.

**Off-topic and policy cases.**

- A poem written on one line next to a refusal is not seen as verse.
- A redirect to support counts as stating the policy, even after an opening "Sure" or "Yes".
- A promise made on support's behalf ("I'll ask the support team to extend it to 5 years") is caught only by the case's forbidden list.
- A short honest denial without a refusal verb or the real term ("No, power tools don't have a 5-year warranty") fails the policy check.

## Operational limits

- **A request the provider always refuses.** Recording skips a refused request and goes on, and the next run asks it again, after the fresh calls. If the provider refuses it every time, the recording never completes and the manifest is not written; every stop lists the missing calls. A whole refused block at the very end of a function's plan can still go first on each new day and use part of the daily allowance.
- **A free model that is removed.** Every call then fails with HTTP 404, "No endpoints found". The run does not stop early on it; change the model in [`config/models.yaml`](../config/models.yaml).
- **The service in replay mode** answers only the recorded dataset questions. Any other question gets HTTP 503 with a hint to run it live with free models.
- **Docker and file permissions.** The service reads `cassettes/` through a read-only mount. A restrictive umask when recording can leave cassette files the container cannot read.
- **New cases.** The gate checks every case the baseline lists. A case that a live run has and the baseline does not is not gated until the baseline is updated.

## What this repository does not claim

It does not claim that the assistant, the prompts or the evaluation conform to any standard or law. It checks what this page and [docs/02](02-eval-strategy.md) list, and nothing more.
