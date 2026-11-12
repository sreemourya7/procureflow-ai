"""Explicit opt-in paid/local live-model benchmark. Never creates a fake result."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from .engine import Engine
from .synthetic import workflow_fixture, SCENARIOS
from .evaluation import detection


def evaluate_live(root: Path, cases=14):
    if not 1 <= cases <= 280:
        raise ValueError("cases must be between 1 and 280")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    db = root / "runtime" / f"live-eval-{stamp}.db"
    observations = []
    truth = []
    pred = []
    for i in range(cases):
        engine = Engine(root, db, mode="live")
        fixture = workflow_fixture(i, SCENARIOS[i % len(SCENARIOS)])
        cid, _ = engine.create(fixture["submission"], f"live-{stamp}-{i}")
        start = time.perf_counter()
        error = None
        try:
            engine.run(cid)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:300]
        case = engine.store.get(cid)
        expected = set(fixture["expected_codes"])
        actual = {f["code"] for f in case["snapshot"].get("check_compliance", [])}
        truth.append(expected)
        pred.append(actual)
        usage = [e["detail"] for e in case["events"] if e["status"] == "usage"]
        observations.append(
            dict(
                id=fixture["id"],
                case_id=cid,
                scenario=fixture["scenario"],
                completed=case["state"] == "awaiting_review",
                exact_match=error is None and expected == actual,
                error=error,
                expected=sorted(expected),
                actual=sorted(actual),
                latency_ms=(time.perf_counter() - start) * 1000,
                prompt_tokens=sum(u["prompt_tokens"] for u in usage),
                completion_tokens=sum(u["completion_tokens"] for u in usage),
                model=engine.provider.model,
            )
        )
        print(f"Live case {i + 1}/{cases}: {case['state']}", flush=True)
    report = dict(
        mode="live",
        synthetic=True,
        live_llm_evaluated=True,
        cases=cases,
        completion_rate=sum(r["completed"] for r in observations) / cases,
        exact_match_rate=sum(r["exact_match"] for r in observations) / cases,
        finding_detection=detection(truth, pred),
        latency_ms=dict(
            p50=float(np.percentile([r["latency_ms"] for r in observations], 50)),
            p95=float(np.percentile([r["latency_ms"] for r in observations], 95)),
        ),
        prompt_tokens=sum(r["prompt_tokens"] for r in observations),
        completion_tokens=sum(r["completion_tokens"] for r in observations),
        observations=observations,
        note="Real configured model calls on synthetic fixtures. Failed cases count against completion and exact-match rates. Citation validation is enforced by the runtime; no real-world procurement quality claim.",
    )
    import os

    incoming = os.getenv("LLM_INPUT_USD_PER_MILLION")
    outgoing = os.getenv("LLM_OUTPUT_USD_PER_MILLION")
    report["cost_usd"] = (
        None
        if incoming is None or outgoing is None
        else (
            report["prompt_tokens"] * float(incoming)
            + report["completion_tokens"] * float(outgoing)
        )
        / 1e6
    )
    (root / "results/live_benchmark.json").write_text(json.dumps(report, indent=2))
    return report
