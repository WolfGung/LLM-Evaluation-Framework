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
| Safety cases passed | 8 of 12 | 12 of 12 | — | — |
| Stable cases | 86.5% | 100.0% | 92.5% | 97.5% |
| Cost per run | $0.00 (free models) | $0.00 (free models) | $0.00 (free models) | $0.00 (free models) |
| Latency p50 / p95 | 1.8 s / 25.3 s | 0.9 s / 25.3 s | 0.9 s / 13.0 s | 1.0 s / 10.7 s |

Each column is one prompt version. Pass rates count every run: each case ran 3 times. Recorded on 2026-10-05 (UTC) with qwen/qwen3.8-27b:free (system) and nvidia/nemotron-3-super-120b-a12b:free (judge), 706 calls.

<!-- results:end -->
