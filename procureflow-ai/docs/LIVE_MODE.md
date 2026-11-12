# Live LLM execution

The bundled benchmark is offline. To measure real agent behavior, configure a
Chat Completions endpoint supporting `tools`, `tool_choice=required`, and JSON
responses. It can be a hosted provider or a compatible local server. Select a model
that supports tool calling; compatibility differs across endpoints.

PowerShell:

```powershell
$env:PROCUREFLOW_MODE="live"
$env:LLM_BASE_URL="https://api.openai.com/v1"
$env:LLM_MODEL="YOUR_TOOL_CALLING_MODEL"
$env:LLM_API_KEY="YOUR_API_KEY"
python -m procureflow.cli demo --mode live --key live-demo-v1
python -m procureflow.cli serve
```

Bash:

```bash
export PROCUREFLOW_MODE=live
export LLM_BASE_URL=https://api.openai.com/v1
export LLM_MODEL=YOUR_TOOL_CALLING_MODEL
export LLM_API_KEY=YOUR_API_KEY
python -m procureflow.cli demo --mode live --key live-demo-v1
```

Set optional `LLM_INPUT_USD_PER_MILLION` and `LLM_OUTPUT_USD_PER_MILLION` to the
current prices for your chosen model. Request usage is persisted in case events,
including responses later rejected by the extractor. Unknown prices yield null cost,
not a made-up estimate. Token prices are supplied by the operator.

No provider credentials were available during development, so live HTTP requests were
not run. Tests cover the HTTP contract, malformed responses, rate limits, unsupported
tools, exact source grounding, and alternate valid tool order with a mocked transport.
Mocks are not counted as live LLM evaluation. Ambiguous or incorrect source quotes are rejected;
invalid responses fail closed rather than silently replacing values with regex output.

A live run may incur charges. Do not publish an API key or commit `.env`. The project
never reads an external mailbox or contacts a vendor.

## Measure your configured model

After setting the environment variables above:

```bash
python -m procureflow.cli evaluate-live --cases 14
```

This makes actual model calls and saves `results/live_benchmark.json` and the
per-request audit records in `runtime/live-eval-*.db`. It reports completion,
exact finding-set match, finding precision/recall, p50/p95 latency, token usage,
and cost if both rates were provided. Failed cases remain in the denominator.
Start with `--cases 1` to verify compatibility and inspect cost before a larger run.
The offline command never overwrites the live report.
