# ProcureFlow AI

**Evidence-backed procurement review agents with PyTorch document routing, durable workflows, and human approval.**

ProcureFlow takes a supplier submission, routes its attachments, extracts source-backed
facts, retrieves policy requirements, checks compliance, and prepares a clarification
draft for a human reviewer. Every tool execution is traced and checkpointed. A crash
can resume without creating a duplicate draft.

This is a runnable engineering prototype using a **fictional policy and synthetic
vendors**. It does not claim real government deployment, business savings, or
production procurement accuracy.

## Measured results

The bundled offline benchmark contains **280 submissions across 14 authored
scenarios**, including missing and duplicate attachments, identity conflicts,
expired insurance, excessive retention, missing fields, and instruction-like text.

| Measurement | Observed result |
|---|---:|
| Exact expected finding-set match | 280 / 280 synthetic cases |
| Finding precision / recall | 100% / 100% on this fixture set |
| Correct recovery from injected crashes and transient failures | 60 / 60 trials |
| Duplicate clarification drafts after failures | 0 |
| Distinct cases from 100 concurrent duplicate submissions | 1 |
| Exact source span alignment | 100% across 2,760 extracted facts |
| Offline end-to-end p95 case latency | See [measured results](results/RESULTS.md) |

These are **narrow synthetic development measurements**, not external validation or
LLM reasoning scores. The PyTorch model's three-seed results and the stronger
TF-IDF/logistic baseline are reported without cherry-picking. The workflow combines
neural predictions with a schema compatibility guard; workflow accuracy is not pure
neural classifier accuracy. See [all results and limits](results/RESULTS.md),
[raw benchmark](results/benchmark.json), and [data card](docs/DATA_CARD.md).

## What works

- **Bounded agent tools:** intake, source evidence, policy retrieval, compliance, and
  review specialists coordinated through an explicit dependency graph.
- **PyTorch model:** trainable EmbeddingBag + MLP router; three seeds, validation-based
  checkpoint selection, temperature selection, and low-confidence abstention.
- **Evidence:** exact document character ranges, source quotes, clause IDs, versions,
  and a policy hash. Generated facts must match source text.
- **Policy retrieval:** TF-IDF top-k retrieval with scored citations; mandatory rules
  are checked exhaustively to prevent retrieval misses from waiving requirements.
- **Recovery:** bounded retries, SQLite WAL checkpoints, worker leases, fencing tokens,
  idempotency keys, and a transactional local draft outbox.
- **Human review:** optimistic revisions, blocking findings, separate optional reviewer
  credentials, and explicit approval before draft export.
- **Workbench:** browser UI for JSON intake, findings, agent trace, source evidence,
  saved cases, human decisions, and measured results. OpenAPI at `/docs`.
- **Packaging:** CPU checkpoint, generated fixtures, tests, Docker/Compose, and CI.

## Run locally — Windows PowerShell

Use Python **3.12**. Extract the project and open PowerShell in the `procureflow-ai`
folder. Activation is unnecessary:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m procureflow.cli serve
```

Open **http://127.0.0.1:8000**. Select **Load sample case → Create & run review**.
Inspect the three findings, source evidence, and agent trace. Enter a reviewer name
and note, approve the clarification draft, then download it. The sample has an
over-budget offer, expired insurance, and excessive retention. No email is sent.

The trained CPU model and example are bundled: no GPU, retraining, or API key is
needed for the offline demo. The root folder is the CLI's default project root;
use `--root /absolute/project/path` before the subcommand when running elsewhere.

## Run locally — macOS / Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Linux/Windows CPU wheels. On macOS use: python -m pip install torch
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python -m procureflow.cli serve
```

## Reproduce the measurements

PowerShell, from the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m procureflow.cli generate
.\.venv\Scripts\python.exe -m procureflow.cli train
.\.venv\Scripts\python.exe -m procureflow.cli evaluate
.\.venv\Scripts\python.exe -m procureflow.cli demo --key my-demo-v1
```

On an activated environment, use `python` instead of the full executable path.
`generate` writes the seeded datasets; `train` retrains three neural seeds and both
classical baselines; `evaluate` writes actual raw observations and a human-readable
report. Latency depends on your machine. `requirements-tested.txt` records the
direct dependency versions used for the bundled run; transitive dependencies are
not fully locked. Runtime databases and secrets are excluded from Git.

## Offline versus live LLM mode

| Mode | Coordinator | Extraction | What the bundled metrics measure |
|---|---|---|---|
| `offline` | Deterministic tool scheduler | Supported-label parser | Workflow correctness, model routing, retrieval, persistence, and failures |
| `live` | Real LLM chooses a dependency-ready tool | Real LLM returns extractive facts; server resolves/validates source spans | Run separately with your configured model |

**No live provider requests were run during development.** The live adapter is
contract-tested using mocked HTTP responses. No silent fallback replaces a failed
LLM with offline output. Read [live setup and evaluation](docs/LIVE_MODE.md) to use a
hosted endpoint or compatible local server and produce a separate live benchmark.

```powershell
$env:LLM_BASE_URL="https://api.openai.com/v1"
$env:LLM_MODEL="YOUR_TOOL_CALLING_MODEL"
$env:LLM_API_KEY="YOUR_API_KEY"
.\.venv\Scripts\python.exe -m procureflow.cli demo --mode live --key live-demo-v1
# Makes real model calls; start small to check compatibility and cost:
.\.venv\Scripts\python.exe -m procureflow.cli evaluate-live --cases 1
```

For the browser server also set `$env:PROCUREFLOW_MODE="live"`. Environment examples
are in `.env.example`; the Python CLI does not auto-load `.env`. Hosted model calls
can incur costs. Keep credentials out of Git and shell screenshots.

## API example

```bash
curl -X POST http://127.0.0.1:8000/api/cases \
  -H 'Content-Type: application/json' -H 'Idempotency-Key: example-v1' \
  --data-binary @data/examples/submission.json
# Use the returned ID:
curl -X POST http://127.0.0.1:8000/api/cases/CASE_ID/run
curl http://127.0.0.1:8000/api/cases/CASE_ID
```

A duplicate idempotency key returns the original case. Reusing it with changed
content returns 409. Human decisions require the case's current `revision`.
Set `PROCUREFLOW_API_TOKEN` and `PROCUREFLOW_REVIEWER_TOKEN` to enable separate API
and reviewer tokens; pass `Authorization: Bearer ...` and `X-Reviewer-Token: ...`.
The dashboard connection settings accept both without browser-local persistence.

## Docker

```bash
docker compose up --build
```

Compose exposes only `127.0.0.1:8000` and persists case state in a named volume. The
Docker configuration is included; Docker was not available for a build/run test in
the development environment. Before network exposure, add verified identity,
HTTPS, tenancy, strict streaming limits, rate limiting, and operational monitoring.

## Repository map

```text
procureflow/           model, agents, tools, API, persistence, evaluation, browser UI
artifacts/            trained CPU router and model metadata
data/policies/        versioned fictional procurement policy
data/examples/        runnable sample submission
data/generated/       seeded classifier splits and workflow fixtures
results/              actual metrics, per-case observations, model runs, test output
tests/                workflow, fault, race, API, evidence, and provider contract tests
docs/                 architecture, data card, live setup, GitHub guide, interview notes
```

Start with [architecture](docs/ARCHITECTURE.md), [results](results/RESULTS.md), and
[GitHub upload commands](docs/GITHUB_UPLOAD.md). The project license is MIT.
