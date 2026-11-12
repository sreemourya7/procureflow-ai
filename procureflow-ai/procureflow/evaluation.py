"""Synthetic evaluation with independent scenario oracles and persistent failure trials."""

import hashlib
import json
import platform
import statistics
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
from .compliance import check
from .domain import Conflict, TransientError
from .engine import Engine
from .evidence import REQUIRED, extract, valid_evidence
from .ml import keyword_predict
from .retrieval import PolicyIndex
from .synthetic import classification_data, workflow_fixture, SCENARIOS

RETRIEVAL_QUERIES = [
    ("The business names differ between the attachments", "P1"),
    ("Supplier identity must match the legal company", "P1"),
    ("Conflicting vendor identity across files", "P1"),
    ("What format must a tax registration identifier use?", "P2"),
    ("Missing tax number needs clarification", "P2"),
    ("Is NN-NNNNNNN a required registration format?", "P2"),
    ("Certificate has expired before review", "P3"),
    ("Minimum liability coverage in dollars", "P3"),
    ("The insurance limit is too low", "P3"),
    ("Total offer exceeds our allocated budget", "P4"),
    ("Maximum commercial quoted price", "P4"),
    ("What is the spending ceiling for the commercial offer?", "P4"),
    ("One of the mandatory files is missing", "P5"),
    ("Duplicate documents and unsupported attachments", "P5"),
    ("Unable to verify a required field", "P5"),
    ("Must stored data be encrypted?", "P6"),
    ("Maximum retention period for data", "P6"),
    ("The security response stores records for 90 days", "P6"),
]


class SimulatedCrash(BaseException):
    pass


def detection(truth, pred):
    tp = sum(len(a & b) for a, b in zip(truth, pred))
    fp = sum(len(b - a) for a, b in zip(truth, pred))
    fn = sum(len(a - b) for a, b in zip(truth, pred))
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return dict(
        precision=p,
        recall=r,
        f1=2 * p * r / (p + r) if p + r else 0.0,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
    )


def robustness(root, work):
    trials = []
    tools = [
        "classify_documents",
        "extract_fields",
        "retrieve_policy",
        "check_compliance",
        "draft_review",
    ]
    for i in range(60):
        crash = i < 30
        point = ("after:" if crash else "before:") + tools[i % 5]
        fired = [False]

        def fault(p):
            if p == point and not fired[0]:
                fired[0] = True
                if crash:
                    raise SimulatedCrash()
                raise TransientError("Injected temporary tool timeout")

        db = work / f"failure-{i}.db"
        engine = Engine(root, db, fault=fault, sleep=lambda _: None)
        cid, _ = engine.create(
            workflow_fixture(i, "multiple_issues")["submission"], f"failure-{i}"
        )
        try:
            result = engine.run(cid)
        except SimulatedCrash:
            # Simulate elapsed lease time without blocking the benchmark for 300 sec.
            with engine.store.transaction() as connection:
                connection.execute("UPDATE cases SET lease_until=0 WHERE id=?", (cid,))
            result = Engine(root, db, sleep=lambda _: None).run(cid)
        codes = {f["code"] for f in result["snapshot"]["check_compliance"]}
        ok = (
            result["state"] == "awaiting_review"
            and len(result["outbox"]) == 1
            and codes == {"EXPIRED_INSURANCE", "OVER_BUDGET", "RETENTION_EXCEEDED"}
        )
        trials.append(
            dict(
                kind="process_crash" if crash else "transient_tool_error",
                point=point,
                recovered=ok,
                drafts=len(result["outbox"]),
            )
        )
    engine = Engine(root, work / "idempotency.db")
    submission = workflow_fixture(5, "clean")["submission"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(
            pool.map(lambda _: engine.create(submission, "same-key")[0], range(100))
        )
    conflicts = 0
    for i in range(10):
        try:
            engine.create(dict(submission, vendor="Changed vendor"), "same-key")
        except Conflict:
            conflicts += 1
    return dict(
        trials=60,
        successful_recoveries=sum(t["recovered"] for t in trials),
        recovery_rate=sum(t["recovered"] for t in trials) / 60,
        duplicate_drafts=sum(max(0, t["drafts"] - 1) for t in trials),
        duplicate_submission_attempts=100,
        distinct_cases=len(set(ids)),
        conflicting_replays_rejected=conflicts,
        conflicting_replays_tested=10,
        lease_expiry="accelerated in fixture; no wall-clock recovery-time claim",
        details=trials,
    )


def evaluate(root):
    root = Path(root)
    records = []
    truth = []
    pred = []
    base_pred = []
    latencies = []
    spans = 0
    valid_spans = 0
    policy_cites = 0
    valid_policy = 0
    correct_fields = 0
    total_fields = 0
    correct_routes = 0
    total_routes = 0
    index = PolicyIndex(root / "data/policies/demo_policy.json")
    with tempfile.TemporaryDirectory(prefix="procureflow-eval-") as tmp:
        work = Path(tmp)
        engine = Engine(root, work / "workflow.db")
        for i in range(5):
            cid, _ = engine.create(
                workflow_fixture(10000 + i, "clean")["submission"], f"warmup-{i}"
            )
            engine.run(cid)
        started = time.perf_counter()
        for i in range(280):
            fixture = workflow_fixture(i, SCENARIOS[i % len(SCENARIOS)])
            sub = fixture["submission"]
            start = time.perf_counter()
            cid, _ = engine.create(sub, f"eval-{i}")
            case = engine.run(cid)
            latency = (time.perf_counter() - start) * 1000
            latencies.append(latency)
            snap = case["snapshot"]
            actual = {f["code"] for f in snap["check_compliance"]}
            expected = set(fixture["expected_codes"])
            truth.append(expected)
            pred.append(actual)
            baseline = [
                dict(document_id=d["id"], kind=keyword_predict(d["text"]))
                for d in sub["documents"]
            ]
            base_extract = {
                r["document_id"]: extract(
                    next(d for d in sub["documents"] if d["id"] == r["document_id"]),
                    r["kind"],
                )
                for r in baseline
            }
            base_pred.append(
                {f["code"] for f in check(sub, baseline, base_extract, index)}
            )
            for r in snap["classify_documents"]:
                target = fixture["truth"].get(r["document_id"], {"kind": "other"})[
                    "kind"
                ]
                total_routes += 1
                correct_routes += r["kind"] == target
            for docid, target in fixture["truth"].items():
                for field in REQUIRED[target["kind"]]:
                    if (
                        fixture["scenario"] == "missing_field"
                        and target["kind"] == "insurance_certificate"
                        and field == "expiry"
                    ):
                        continue
                    total_fields += 1
                    correct_fields += snap["extract_fields"].get(docid, {}).get(
                        field, {}
                    ).get("value") == str(target["values"][field])
            for facts in snap["extract_fields"].values():
                for fact in facts.values():
                    spans += 1
                    valid_spans += valid_evidence(fact["evidence"], sub["documents"])
            for f in snap["check_compliance"]:
                policy_cites += 1
                valid_policy += (
                    f["policy_quote"] == index.clause(f["clause_id"])["text"]
                    and f["policy_version"] == index.policy["version"]
                    and f["policy_id"] == index.policy["id"]
                )
            records.append(
                dict(
                    id=fixture["id"],
                    scenario=fixture["scenario"],
                    expected=sorted(expected),
                    actual=sorted(actual),
                    exact_match=expected == actual,
                    latency_ms=latency,
                )
            )
        elapsed = time.perf_counter() - started
        failure = robustness(root, work)
    retrieval = []
    for query, target in RETRIEVAL_QUERIES:
        ids = [h["id"] for h in index.search(query, 3)]
        retrieval.append(
            dict(
                query=query,
                expected=target,
                returned=ids,
                reciprocal_rank=1 / (ids.index(target) + 1) if target in ids else 0.0,
            )
        )
    data = classification_data()
    families = {k: {r["family"] for r in rows} for k, rows in data.items()}
    report = dict(
        synthetic=True,
        mode="offline",
        live_llm_evaluated=False,
        workflow=dict(
            cases=280,
            scenarios=len(SCENARIOS),
            exact_match_rate=statistics.mean(r["exact_match"] for r in records),
            finding_detection=detection(truth, pred),
            keyword_routing_baseline=dict(
                exact_match_rate=statistics.mean(
                    a == b for a, b in zip(truth, base_pred)
                ),
                finding_detection=detection(truth, base_pred),
            ),
            document_routing_accuracy=correct_routes / total_routes,
            extraction_field_accuracy=correct_fields / total_fields,
            extracted_source_span_validity=valid_spans / spans,
            extracted_facts=spans,
            policy_citation_validity=valid_policy / policy_cites,
            policy_citations=policy_cites,
            latency_ms=dict(
                p50=float(np.percentile(latencies, 50)),
                p95=float(np.percentile(latencies, 95)),
                maximum=max(latencies),
            ),
            sequential_cases_per_second=280 / elapsed,
            latency_scope="Single CPU process: validation, SQLite create/checkpoints/events, routing, extraction, retrieval, checks, draft and trace read. No HTTP/LLM/OCR. Five warmups excluded. Throughput wall time includes baseline analysis.",
            per_scenario={
                s: dict(
                    cases=sum(r["scenario"] == s for r in records),
                    exact_match_rate=statistics.mean(
                        r["exact_match"] for r in records if r["scenario"] == s
                    ),
                )
                for s in SCENARIOS
            },
        ),
        retrieval=dict(
            queries=18,
            recall_at_1=statistics.mean(
                r["returned"][0] == r["expected"] for r in retrieval
            ),
            recall_at_3=statistics.mean(
                r["expected"] in r["returned"] for r in retrieval
            ),
            mrr_at_3=statistics.mean(r["reciprocal_rank"] for r in retrieval),
        ),
        reliability=failure,
        split_integrity=dict(
            template_families_disjoint=all(
                not families[a] & families[b]
                for a, b in [
                    ("train", "validation"),
                    ("train", "test"),
                    ("validation", "test"),
                ]
            )
        ),
        provenance=dict(
            python=platform.python_version(),
            pytorch=torch.__version__,
            platform=platform.platform(),
            cpu=platform.processor() or platform.machine(),
            torch_threads=torch.get_num_threads(),
            model_sha256=hashlib.sha256(
                (root / "artifacts/router.pt").read_bytes()
            ).hexdigest(),
            policy_sha256=index.sha256,
        ),
        limitations=[
            "All documents, rules, and workflow oracles are synthetic.",
            "Held-out layouts within one generator are development evaluation, not untouched external validation.",
            "Offline extraction supports labeled text; arbitrary prose, scanned PDFs, and OCR are unsupported.",
            "Span validity proves source alignment, not semantic correctness.",
            "Policy retrieval has six clauses; it is not an enterprise-scale retrieval benchmark.",
            "No measured human labor savings, real vendor outcomes, deployed SLOs, or live LLM quality/cost.",
        ],
    )
    (root / "results/benchmark.json").write_text(json.dumps(report, indent=2))
    (root / "results/workflow_cases.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    (root / "results/retrieval_cases.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in retrieval)
    )
    write_report(root, report)
    return report


def write_report(root, report):
    c = json.loads((root / "results/classifier.json").read_text())
    w = report["workflow"]
    r = report["reliability"]
    q = report["retrieval"]
    pct = lambda x: f"{100 * x:.1f}%"
    rows = [
        ("Workflow cases / scenarios", f"{w['cases']} / {w['scenarios']}"),
        ("Exact finding-set match", pct(w["exact_match_rate"])),
        (
            "Finding precision / recall / F1",
            " / ".join(
                pct(w["finding_detection"][k]) for k in ["precision", "recall", "f1"]
            ),
        ),
        (
            "Keyword routing baseline: exact match",
            pct(w["keyword_routing_baseline"]["exact_match_rate"]),
        ),
        ("Document routing accuracy", pct(w["document_routing_accuracy"])),
        ("Extraction field accuracy", pct(w["extraction_field_accuracy"])),
        (
            "Exact source-span validity",
            f"{pct(w['extracted_source_span_validity'])} across {w['extracted_facts']} facts",
        ),
        (
            "Versioned policy citation validity",
            f"{pct(w['policy_citation_validity'])} across {w['policy_citations']} findings",
        ),
        (
            "Offline latency p50 / p95",
            f"{w['latency_ms']['p50']:.1f} / {w['latency_ms']['p95']:.1f} ms",
        ),
        ("Correct failure recovery", f"{r['successful_recoveries']} / {r['trials']}"),
        ("Duplicate drafts after failures", str(r["duplicate_drafts"])),
        ("Distinct cases from 100 concurrent replays", str(r["distinct_cases"])),
        (
            "Policy retrieval recall@1 / recall@3",
            pct(q["recall_at_1"]) + " / " + pct(q["recall_at_3"]),
        ),
        (
            "PyTorch macro-F1 mean ± std, 3 seeds",
            f"{c['pytorch_mean_macro_f1']:.3f} ± {c['pytorch_std_macro_f1']:.3f}",
        ),
        (
            "TF-IDF logistic regression macro-F1",
            f"{c['tfidf_logistic_baseline']['macro_f1']:.3f}",
        ),
        ("Keyword classifier macro-F1", f"{c['keyword_baseline']['macro_f1']:.3f}"),
    ]
    text = (
        "# Measured results\n\n**Synthetic documents, fictional policy, offline execution.** These results do not measure production impact or live LLM reasoning.\n\n| Measurement | Result |\n|---|---:|\n"
        + "".join(f"| {a} | {b} |\n" for a, b in rows)
    )
    text += (
        "\n## Per-scenario results\n\n| Scenario | Cases | Exact match |\n|---|---:|---:|\n"
        + "".join(
            f"| {s} | {v['cases']} | {pct(v['exact_match_rate'])} |\n"
            for s, v in w["per_scenario"].items()
        )
    )
    text += (
        "\n## Methodology and limits\n\nCheckpoint selection uses minimum validation loss, never test score. Confidence below 0.55 routes to manual review. Temperature selection uses validation data and does not establish real-world calibration. Classical baselines are retained even when they beat the neural router.\n\n"
        + "\n".join("- " + s for s in report["limitations"])
        + "\n\nLatency scope: "
        + w["latency_scope"]
        + "\n\nRecovery trials accelerate lease expiry; they do not establish wall-clock failover time.\n\n## Reproduce\n\n```bash\npython -m procureflow.cli generate\npython -m procureflow.cli train\npython -m procureflow.cli evaluate\npython -m pytest -q\n```\n\nRaw observations: `workflow_cases.jsonl`, `retrieval_cases.jsonl`, `benchmark.json`, and `classifier.json`. Training is deterministic on the recorded CPU configuration; timing varies by machine.\n\n## Environment\n\n```json\n"
        + json.dumps(report["provenance"], indent=2)
        + "\n```\n"
    )
    (root / "results/RESULTS.md").write_text(text)
