# Can the judge be trusted?

The judge is a measuring instrument, and an instrument is checked before its readings count. This page checks it two ways: against a person's labels, and for the known biases of LLM judges. Every result below is generated from [`results/`](../results/); the one dated note from an earlier recording names its commits.

## The judge and its rubric

- **Model.** The `judge` role in [`config/models.yaml`](../config/models.yaml): another model than the system under test, from another vendor, at temperature 0, with a fixed seed and low reasoning effort.
- **Rubric.** [`rubrics/judge.md`](../rubrics/judge.md) asks for three scores from 1 to 5, with anchors for 1, 3 and 5: groundedness to the retrieved documents, helpfulness and tone. An answer passes when groundedness is at least 4, helpfulness at least 3 and tone at least 3. The code reads the same minimums from the rubric's front matter and refuses a rubric whose prose says otherwise.
- **Format.** The verdict is requested with a strict JSON schema and validated after parsing. An empty or invalid verdict is kept, counted and reported. It is never dropped and never asked again, and the judge layer counts it as a failed run.
- **Two passes compared.** The code applies the pass rule to the scores. The judge's own `pass` field is recorded too, and the table below counts how often the two differ.
- **Material, not instructions.** The rubric tells the judge that the question, the documents and the answers are material to grade, that an instruction inside a document is not shop information, that confident wording is not evidence, and that length earns nothing.

## Agreement with a person

Pavel Zhukov Atum, the author, labels a sample of judged answers by hand with `make label`, and `make eval` compares the judge's verdicts with those labels.

- **The question.** For each answer the author decides one thing: would you send this answer to the customer as is? Pass if it is grounded in the shown documents, answers the question (or says honestly that the documents do not cover it), and is polite. That is the judge's three criteria in one decision.
- **Blind.** The tool shows the question, the documents the assistant was given and the answer. It never shows the judge's verdict, the case id, the prompt version or the category. The two versions' answers to the same question are kept apart in the order.
- **The sample.** [`labels/sample.json`](../labels/sample.json) holds every answer the judge failed, plus answers it passed, drawn with a fixed seed across the case categories and both versions. Judge failures are rare, so a random sample could hold none, and then agreement would say nothing about the answers the judge rejects. The price is that agreement on this sample is not the agreement over all answers.
- **The measures.** Percent agreement, and Cohen's kappa: agreement corrected for the agreement two raters would reach by chance, given how often each one passes answers. A kappa of 1 is perfect agreement; 0 is no better than chance. Kappa matters here: when most answers pass, two raters who pass almost everything agree often without judging alike.

<!-- agreement:start -->

pending human labels

None of the author's 30 labels in labels/human.jsonl is of an answer in the current sample: the answers or the sample changed after he labelled them. Stale labels are not used; each is listed with its reason in results/judge-agreement.json.

Pavel Zhukov Atum, the author, labels 30 judged answers by hand, blind to the judge's verdict (make label), and this sample awaits his labels: all 23 answers the judge failed and 7 it passed. The sample oversamples judge failures, so agreement on it is not the agreement over all answers.

<!-- agreement:end -->

When labels exist, [`results/judge-agreement.json`](../results/judge-agreement.json) also lists every disagreement with the judge's scores and reasons and the author's comment.

## The judge's grades

<!-- judge:start -->

| Measure | rag v1 | rag v2 |
|---|---:|---:|
| Answers graded | 40 | 40 |
| Valid verdicts | 40 of 40 | 40 of 40 |
| Pass by the rubric rule | 26 of 40 | 31 of 40 |
| The judge's own pass differs from the rule | 0 | 0 |
| Mean groundedness | 3.75 | 4.45 |
| Mean helpfulness | 3.95 | 4.25 |
| Mean tone | 5.00 | 5.00 |
| Answer length and groundedness (Spearman) | -0.53 | -0.21 |
| Answer length and helpfulness (Spearman) | -0.55 | -0.32 |
| Answer length and tone (Spearman) | — | — |

A dash: no correlation can be computed, because every graded answer got the same score or fewer than three answers were graded.

No criterion has a positive correlation: longer answers did not get higher scores.

Position: where the judge chose a side in both orders, it chose the answer shown first 14 of 30 times (46.7%). Half would mean no lean.

Length: of those choices between answers of different lengths, it chose the longer answer 7 of 30 times (23.3%).

The system model is mistralai/mistral-small-3.2-24b-instruct (vendor mistralai) and the judge is nvidia/nemotron-3-super-120b-a12b (vendor nvidia): different vendors, so the judge does not grade answers written by its own model family.

<!-- judge:end -->

## Known biases of LLM judges, and how each is handled

### Position

A judge that compares two answers tends to prefer one position, often the first. So every pairwise question is asked twice, once with each version shown first. A version wins a case only when both orders prefer it. When the orders disagree, the pair is inconsistent, and it is never settled by picking one order. Position consistency, the share of compared pairs with the same verdict in both orders, is reported and gated.

<!-- pairwise:start -->

The judge compared the first answers of rag v1 and rag v2 case by case, asked twice with the order of the two answers swapped.

| Outcome over 40 cases | Cases |
|---|---:|
| v1 preferred in both orders | 3 |
| v2 preferred in both orders | 11 |
| A tie in both orders | 11 |
| Inconsistent: the two orders disagree | 11 |
| Identical answers, not compared | 4 |

Position consistency: 25 of 36 compared pairs (69.4%) got the same verdict in both orders.

In 11 of the 36 compared pairs, the judge's preference changed when the two answers swapped places: 1 time it chose the answer shown second in both orders, and 10 times it called a tie in one order and chose a side in the other. An inconsistent pair is never settled by picking one order.

With this many flips, the comparison says more about the judge's position bias than about the two prompts, so it picks no winner. The main table rests on the rules and the per-answer grades.

<!-- pairwise:end -->

Read the outcomes with the consistency next to them: identical answers are not asked and count in no rate, and only the pairs the judge really compared count toward consistency. When consistency is low, the block above says so and names no winner. The main table rests on the per-answer grades and the rule-based layers, not on the pairwise result.

### Length

Judges tend to reward longer answers. The rubric says length earns nothing, and two cheap checks look for the bias anyway: the correlation between an answer's length in words and each of its scores, and, in the pairwise comparison, how often the judge chose the longer answer. Neither is proof. A longer answer can also be more complete. The check matters here because the v2 prompt asks for short answers, so the two versions differ in length by design.

### Self-preference

Judges tend to prefer text written by their own model family. The system and the judge come from different vendors, and the judge block above checks that from the model ids the run was recorded with.

### Instructions inside the material

A document can tell an AI to do something, as the supplier page in the knowledge base does. The rubric says such text is not shop information and that an answer repeating it is not grounded. The safety cases are not graded by the judge at all: rules own safety ([docs/04](04-safety-cases.md)).

## The first recording, on free models

<!-- history:start -->
The first recording (commit 66b4a3a, 2026-10-05, with the judge calls re-recorded in commit 1fc63b9) ran on free model variants: qwen/qwen3.8-27b:free as the system and nvidia/nemotron-3-super-120b-a12b:free as the judge, at no cost. The author labelled 30 of its judged answers by hand (commit d2e9856; the agreement they gave is on that commit's pages). On 2026-10-06 both roles were recorded again on paid models, the judge on the paid variant of the same model, so that the cost row shows what a run costs. Every answer changed, so the author's labels now label answers that no longer exist: they stay in labels/human.jsonl, unused, and the sample was drawn again from the new judge's verdicts.
<!-- history:end -->

## A finding from the first recording: the judge's token budget

<!-- history:start -->
The first recording (commit 66b4a3a) gave the judge a budget of max_tokens 1500. Its reasoning used the whole budget in 13 of 154 judge calls (6 gradings and 7 pairwise questions), so those verdicts were empty or cut off, and replay counted them as invalid. The author raised the judge's max_tokens to 4096 and re-recorded only the judge calls (commit 1fc63b9); the system's answers stayed as they were.
<!-- history:end -->

The lesson for any evaluation with a reasoning model as judge: reasoning tokens count toward the answer's token budget, and a budget that is too small looks like a judge that cannot follow the format. That is why the share of valid verdicts is reported per version and gated, and why the client records each call's finish reason and reasoning tokens.
