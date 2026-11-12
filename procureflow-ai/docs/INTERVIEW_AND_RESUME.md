# Interview and resume notes

Run the project, inspect the code, retrain the model, and be able to explain the
tradeoffs before representing this as work you implemented and understand. The
statements below describe this repository and its measured synthetic benchmark;
they do not imply deployment to a government or customer.

## Project title

**ProcureFlow AI — Evidence-Backed Procurement Review Agents**

Python, PyTorch, FastAPI, SQLite, scikit-learn, OpenAI-compatible tool calling, Docker.

## Three defensible project bullets

- Built a procurement review prototype with five specialist agent tools, PyTorch
  document routing, policy retrieval, exact source citations, and a human approval
  workflow; matched expected findings on 280 synthetic submissions across 14 scenarios.
- Implemented durable checkpoints, bounded retries, worker leases with fencing, and
  an idempotent draft outbox; recovered correctly in 60 injected-failure trials with
  zero duplicate drafts and collapsed 100 concurrent replay attempts into one case.
- Developed a reproducible evaluation pipeline with template-separated datasets,
  three-seed neural training, classical baselines, field-level accuracy, citation
  alignment, and p50/p95 latency; separated offline workflow results from live LLM tests.

Do not claim a percentage reduction in manual labor, real procurement accuracy,
government deployment, autonomous contract approval, or production availability.
The LLM adapter is implemented and mock-tested, but real LLM metrics must come from
running `evaluate-live` against your actual model. The neural classifier did not beat
the classical baseline; do not describe it as a neural accuracy improvement.

## Questions to prepare

1. Why does mandatory compliance checking use deterministic rules even with an LLM?
2. What is the difference between retrieval context and authoritative policy coverage?
3. Why did the router need a schema compatibility check beyond confidence thresholding?
4. How does a fencing token prevent an expired worker from writing stale state?
5. Which crash boundary would create duplicates without the outbox uniqueness constraint?
6. What does exact source-span alignment prove, and what does it fail to prove?
7. Why can synthetic template holdout still overstate real-world generalization?
8. Why keep a PyTorch model when logistic regression scores better on this task?
9. What changes would PostgreSQL, queued workers, multiple tenants, and PDF/OCR require?
10. How would you measure reviewer correction rate and actual customer value in a pilot?

## Current engineering judgment

For a production version of this small structured dataset, the classical classifier
is a serious alternative. PyTorch demonstrates implemented model training and model
operations; it does not justify replacing a better baseline without further evidence.
The main strengths of this project are source-grounded workflow execution, explicit
failure boundaries, evaluation discipline, and inspectable human control.
