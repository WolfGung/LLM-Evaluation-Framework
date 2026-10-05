# What is evaluated

The system under test is the support service of Toolshop, a fictional online shop for hand and power tools. It has two AI features, and each comes in two prompt versions. The evaluation measures both features, layer by layer, and compares the two versions.

## The system under test

The service is a small FastAPI application in [`app/`](../app/). It runs in Docker and answers from the recorded calls by default, so it needs no key (see the [README](../README.md#how-to-run)).

- **Support assistant**, `POST /assist`. A search over the knowledge base in [`app/kb/`](../app/kb/) returns the three best-matching documents. The search is BM25, written in the repository ([`app/retrieval.py`](../app/retrieval.py)). The model answers from those documents and cites their ids in square brackets, such as `[kb-returns]`. This is retrieval-augmented generation (RAG): the function is called `rag` in the results.
- **Ticket triage**, `POST /triage`. A customer's ticket goes in. A JSON object comes out: one of seven categories, one of four priorities, the order id or `null`, and a one-sentence summary. The reply is parsed and validated with Pydantic ([`app/triage.py`](../app/triage.py)). The default model gets the JSON schema in its prompt, not as an enforced output format, so a broken reply is possible and is measured.

The knowledge base holds two traps on purpose. Any question can retrieve them, as a real knowledge base can hold pages nobody meant customers to see:

- `kb-internal-notes`: internal support notes with a customer's name, email address, phone number and order, and internal rules;
- `kb-supplier-promo`: a supplier page that also tells AI assistants to offer every customer a discount code.

## The two prompt versions

The prompts are in [`app/prompts/`](../app/prompts/).

- **v1** is a short prompt, the kind a first draft has: answer from the documents, be friendly, cite the documents. Triage v1 gives the schema and one line per field.
- **v2** adds explicit rules. The assistant answers only from the documents, cites every sentence, says "I don't know" when the documents do not answer, treats documents and the customer's message as data rather than instructions, never shares internal notes or personal data, stays on Toolshop topics and keeps the answer short. Triage v2 adds the category definitions, the priority rules and the order-id rules.

Comparing v1 with v2 is the question a team asks after editing a prompt or switching a model: did it get better, and did anything get worse?

Triage labels follow a written guideline, [`datasets/triage-guideline.md`](../datasets/triage-guideline.md). The v2 prompt carries most of its rules and v1 does not, so part of v2's lead on triage comes from the prompt telling the model the rules it is graded by. The guideline lists the places where it and the v2 prompt differ; a v2 answer that disagrees with a label there is a finding about the prompt.

## What the run covers

<!-- scope:start -->

| Function | Cases | By category |
|---|---|---|
| rag | 52 | answerable 25, multi_doc 7, unanswerable 8, safety 12 |
| triage | 40 | shipping 7, returns 7, payment 6, warranty 7, order_status 6, product_question 4, other 3 |

Each case ran 3 times. The judge graded 40 answers of each rag version, and compared the two versions on 40 cases. Recorded on 2026-10-05 (UTC): 706 calls.

<!-- scope:end -->

The cases are in [`datasets/rag.jsonl`](../datasets/rag.jsonl) and [`datasets/triage.jsonl`](../datasets/triage.jsonl). They are authored test cases: the input and the expected behaviour. The model answers come only from real calls, recorded by `make record` into [`cassettes/`](../cassettes/).

- **answerable**: one document holds the answer; the case names the facts the answer must state and the document it must come from.
- **multi_doc**: the answer needs two or more documents.
- **unanswerable**: no document answers. The assistant must say it does not know and invent nothing.
- **safety**: an attack, described case by case in [docs/04](04-safety-cases.md).

## Risks of AI features for a business, and where each is checked

| Risk | What it can cost | Where it is checked |
|---|---|---|
| The assistant invents a price, a time limit or a policy exception | refunds and complaints; a promise the shop has to honour | required facts, forbidden claims, `dont_know`, the judge's groundedness |
| The answer misses what the documents say, or the search misses the document | lost sales, more tickets for the support team | the retrieval layer, required facts, the judge's helpfulness |
| Prompt injection, typed by a customer or hidden in a document | discount codes given away, rules ignored | the safety layer on every answer, and the injection cases |
| Leaks of personal data, internal notes or the system prompt | a privacy incident, lost trust | the safety layer on every answer, and the disclosure cases |
| The assistant does any task it is asked, such as writing code | cost and brand risk | the off-topic cases |
| A reply the integration cannot parse | the ticket system breaks or drops tickets | triage: valid JSON, the schema, the allowed values; RAG: citations of retrieved documents only |
| Tickets in the wrong category or with the wrong priority | tickets wait in the wrong queue | category, priority and order-id accuracy |
| Answers that change from run to run | a release that passes today and fails tomorrow | the stability layer |
| Slow or expensive calls | customers wait; the bill grows | latency, tokens and cost of every call |
| A prompt or model change that makes things worse | a regression nobody noticed | the baseline, the per-case comparison and the gate |

[docs/02](02-eval-strategy.md) explains the layers and why they come in this order.
