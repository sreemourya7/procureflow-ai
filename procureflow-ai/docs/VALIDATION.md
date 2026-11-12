# Validation record

Executed during development:

- `python -m pytest -q`: 39 tests passed. Third-party FastAPI/pytest integration
  emitted deprecation warnings; no project test failed in the final run.
- `python -m ruff check procureflow tests --select F`: passed.
- `python -m compileall -q procureflow`: passed.
- `node --check procureflow/static/app.js`: passed.
- Three-seed CPU PyTorch training and two classical baselines completed.
- Full offline synthetic workflow, retrieval, recovery, and idempotency benchmark
  completed; raw results are bundled.
- CLI sample case executed and produced an inspectable case/agent trace.
- FastAPI test client exercised authenticated intake, execution, human review,
  clarification export, duplicate requests, and static UI responses.
- Provider HTTP contract tested with mocked transports. The tests do not count as
  real hosted-model execution or establish a model's extraction accuracy.

Not executed:

- Live model requests: no API credentials were supplied.
- Docker image build/run: Docker was unavailable.
- Full browser interaction/render verification: local Chromium launch was blocked
  by the runtime; the separate browser could not reach this local application.
  Static JavaScript syntax and API integration were checked instead. No screenshot
  or browser end-to-end pass is claimed.
- Real documents, OCR, multiple tenants, external vendor delivery, and production
  traffic or reviewer pilots.

Re-run the documented commands on your machine. The supplied checkpoint enables
an immediate offline demo without first reproducing training.
