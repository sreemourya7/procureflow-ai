# Measured results

**Synthetic documents, fictional policy, offline execution.** These results do not measure production impact or live LLM reasoning.

| Measurement | Result |
|---|---:|
| Workflow cases / scenarios | 280 / 14 |
| Exact finding-set match | 100.0% |
| Finding precision / recall / F1 | 100.0% / 100.0% / 100.0% |
| Keyword routing baseline: exact match | 92.9% |
| Document routing accuracy | 100.0% |
| Extraction field accuracy | 100.0% |
| Exact source-span validity | 100.0% across 2760 facts |
| Versioned policy citation validity | 100.0% across 300 findings |
| Offline latency p50 / p95 | 7.1 / 8.2 ms |
| Correct failure recovery | 60 / 60 |
| Duplicate drafts after failures | 0 |
| Distinct cases from 100 concurrent replays | 1 |
| Policy retrieval recall@1 / recall@3 | 100.0% / 100.0% |
| PyTorch macro-F1 mean ± std, 3 seeds | 0.826 ± 0.071 |
| TF-IDF logistic regression macro-F1 | 0.936 |
| Keyword classifier macro-F1 | 0.778 |

## Per-scenario results

| Scenario | Cases | Exact match |
|---|---:|---:|
| clean | 20 | 100.0% |
| missing_insurance | 20 | 100.0% |
| expired_insurance | 20 | 100.0% |
| low_coverage | 20 | 100.0% |
| over_budget | 20 | 100.0% |
| identity_mismatch | 20 | 100.0% |
| invalid_tax | 20 | 100.0% |
| no_encryption | 20 | 100.0% |
| long_retention | 20 | 100.0% |
| multiple_issues | 20 | 100.0% |
| missing_field | 20 | 100.0% |
| injection | 20 | 100.0% |
| duplicate_type | 20 | 100.0% |
| unknown_attachment | 20 | 100.0% |

## Methodology and limits

Checkpoint selection uses minimum validation loss, never test score. Confidence below 0.55 routes to manual review. Temperature selection uses validation data and does not establish real-world calibration. Classical baselines are retained even when they beat the neural router.

- All documents, rules, and workflow oracles are synthetic.
- Held-out layouts within one generator are development evaluation, not untouched external validation.
- Offline extraction supports labeled text; arbitrary prose, scanned PDFs, and OCR are unsupported.
- Span validity proves source alignment, not semantic correctness.
- Policy retrieval has six clauses; it is not an enterprise-scale retrieval benchmark.
- No measured human labor savings, real vendor outcomes, deployed SLOs, or live LLM quality/cost.

Latency scope: Single CPU process: validation, SQLite create/checkpoints/events, routing, extraction, retrieval, checks, draft and trace read. No HTTP/LLM/OCR. Five warmups excluded. Throughput wall time includes baseline analysis.

Recovery trials accelerate lease expiry; they do not establish wall-clock failover time.

## Reproduce

```bash
python -m procureflow.cli generate
python -m procureflow.cli train
python -m procureflow.cli evaluate
python -m pytest -q
```

Raw observations: `workflow_cases.jsonl`, `retrieval_cases.jsonl`, `benchmark.json`, and `classifier.json`. Training is deterministic on the recorded CPU configuration; timing varies by machine.

## Environment

```json
{
  "python": "3.12.14",
  "pytorch": "2.15.0+cpu",
  "platform": "Linux-6.18.57-x86_64-with-glibc2.41",
  "cpu": "x86_64",
  "torch_threads": 2,
  "model_sha256": "909ddf263ecff469d1b4d3a210755728377d7df3e2adf8d34270f78b096fa4c8",
  "policy_sha256": "8c615cdadf78bbc57ea36a9bb983266c1fda9d49bf41b35c6d701e960d6c6a47"
}
```
