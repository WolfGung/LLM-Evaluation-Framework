# LLM-Evaluation-Framework

Evaluate LLM features the way a test engineer evaluates ordinary code.

Work in progress.

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

Each column is one prompt version. Each case ran 3 times; a layer counts the runs it checks (the judge graded the first run of each judged case). A safety case has no safety failure when every safety check passed on every run. The pairwise judge calls are in no column. Recorded on 2026-10-05 (UTC) with qwen/qwen3.8-27b:free (system) and nvidia/nemotron-3-super-120b-a12b:free (judge), 706 calls.

<!-- results:end -->

## Comparing the two RAG prompt versions

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

<!-- pairwise:end -->

## Can the judge be trusted?

<!-- agreement:start -->

pending human labels

The owner labels 30 judged answers by hand, blind to the judge's verdict (make label): all 7 answers the judge failed and 23 it passed. The sample oversamples judge failures, so agreement on it is not the agreement over all answers.

<!-- agreement:end -->
