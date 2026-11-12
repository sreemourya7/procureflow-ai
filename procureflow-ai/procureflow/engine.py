from __future__ import annotations
import time
from pathlib import Path
from .domain import Submission, PermanentError, TransientError, Conflict
from .evidence import extract
from .compliance import check, QUERIES
from .ml import Router
from .providers import LLMProvider
from .retrieval import PolicyIndex
from .store import Store

DEPENDENCIES = {
    "classify_documents": [],
    "extract_fields": ["classify_documents"],
    "retrieve_policy": [],
    "check_compliance": ["extract_fields", "retrieve_policy"],
    "draft_review": ["check_compliance"],
}
AGENTS = {
    "classify_documents": "intake_agent",
    "extract_fields": "evidence_agent",
    "retrieve_policy": "policy_agent",
    "check_compliance": "compliance_agent",
    "draft_review": "review_agent",
}


class Engine:
    def __init__(
        self,
        root: Path,
        db: Path | None = None,
        mode="offline",
        provider=None,
        fault=None,
        sleep=time.sleep,
    ):
        self.root = Path(root)
        self.store = Store(db or self.root / "runtime/procureflow.db")
        self.index = PolicyIndex(self.root / "data/policies/demo_policy.json")
        self.router = Router(self.root / "artifacts/router.pt")
        self.mode = mode
        if mode not in ["offline", "live"]:
            raise ValueError("mode must be offline or live")
        self.provider = (
            provider
            if provider is not None
            else (LLMProvider() if mode == "live" else None)
        )
        self.fault = fault or (lambda point: None)
        self.sleep = sleep

    def create(self, submission, key):
        clean = Submission.model_validate(submission).model_dump()
        return self.store.create(clean, key, self.mode, self.index.sha256)

    def retry(self, cid, agent, tool, fn):
        for attempt in range(1, 4):
            try:
                return fn()
            except TransientError as e:
                self.store.event(
                    cid, agent, tool, "retry", {"attempt": attempt, "reason": str(e)}
                )
                if attempt == 3:
                    raise
                self.sleep(0.05 * 2 ** (attempt - 1))

    def run(self, cid):
        initial = self.store.get(cid)
        if initial["mode"] != self.mode:
            raise Conflict("Resume using the same execution mode")
        if initial["policy_hash"] != self.index.sha256:
            raise Conflict(
                "Policy changed; create a new case with a new idempotency key"
            )
        fence = self.store.claim(cid)
        if fence is None:
            return self.store.get(cid)
        snapshot = initial["snapshot"]
        submission = initial["submission"]
        if self.provider is not None and hasattr(self.provider, "audit"):
            self.provider.audit = lambda usage: self.store.event(
                cid, "llm_provider", "llm_request", "usage", usage
            )
        try:
            while "draft_review" not in snapshot:
                ready = [
                    k
                    for k, deps in DEPENDENCIES.items()
                    if k not in snapshot and all(d in snapshot for d in deps)
                ]
                if not ready:
                    raise PermanentError("No executable tool; corrupted workflow state")
                if self.mode == "live":
                    tool = self.retry(
                        cid,
                        "coordinator",
                        "select_tool",
                        lambda: self.provider.choose(ready, list(snapshot)),
                    )
                else:
                    tool = ready[0]
                if tool not in ready:
                    raise PermanentError("Coordinator selected disallowed tool")
                self.store.event(
                    cid,
                    "coordinator",
                    tool,
                    "selected",
                    {"mode": self.mode, "ready": ready},
                )
                agent = AGENTS[tool]
                started = time.perf_counter()

                def execute():
                    self.fault("before:" + tool)
                    result = self.execute(tool, cid, fence, submission, snapshot)
                    self.fault("after:" + tool)
                    return result

                snapshot[tool] = self.retry(cid, agent, tool, execute)
                if self.mode == "live":
                    usage_events = [
                        e["detail"]
                        for e in self.store.get(cid)["events"]
                        if e["status"] == "usage"
                    ]
                    if usage_events:
                        totals = {
                            k: sum(e.get(k, 0) for e in usage_events)
                            for k in ["prompt_tokens", "completion_tokens"]
                        }
                        import os

                        in_rate = os.getenv("LLM_INPUT_USD_PER_MILLION")
                        out_rate = os.getenv("LLM_OUTPUT_USD_PER_MILLION")
                        cost = (
                            None
                            if in_rate is None or out_rate is None
                            else (
                                totals["prompt_tokens"] * float(in_rate)
                                + totals["completion_tokens"] * float(out_rate)
                            )
                            / 1e6
                        )
                        snapshot["llm_usage"] = dict(
                            **totals,
                            calls=len(usage_events),
                            model=self.provider.model,
                            cost_usd=cost,
                        )
                    else:
                        snapshot["llm_usage"] = dict(
                            **self.provider.usage, cost_usd=self.provider.cost()
                        )
                self.store.checkpoint(
                    cid,
                    fence,
                    snapshot,
                    agent,
                    tool,
                    (time.perf_counter() - started) * 1000,
                )
            return self.store.get(cid)
        except Conflict:
            raise
        except Exception as e:
            self.store.fail(cid, fence, e)
            raise

    def execute(self, tool, cid, fence, submission, snapshot):
        if tool == "classify_documents":
            return self.router.classify(submission["documents"])
        if tool == "extract_fields":
            result = {}
            docs = {d["id"]: d for d in submission["documents"]}
            for row in snapshot["classify_documents"]:
                if row["kind"] == "other":
                    continue
                doc = docs[row["document_id"]]
                if self.mode == "live":
                    result[doc["id"]] = self.provider.extract(doc, row["kind"])
                else:
                    result[doc["id"]] = extract(doc, row["kind"])
            return result
        if tool == "retrieve_policy":
            # Exhaustive policy coverage is a deterministic guard; retrieval is a cited context layer.
            return {
                clause: self.index.search(query, k=3)
                for clause, query in QUERIES.items()
            }
        if tool == "check_compliance":
            return check(
                submission,
                snapshot["classify_documents"],
                snapshot["extract_fields"],
                self.index,
            )
        if tool == "draft_review":
            findings = snapshot["check_compliance"]
            draft = None
            if findings:
                payload = {
                    "vendor": submission["vendor"],
                    "subject": "Clarification requested for procurement submission",
                    "items": [
                        dict(
                            code=f["code"],
                            request=f["message"],
                            policy_clause=f["clause_id"],
                        )
                        for f in findings
                    ],
                    "delivery": "local draft only; no email connector configured",
                }
                draft = self.store.draft(cid, fence, payload)
            return dict(
                recommendation="request_clarification"
                if findings
                else "ready_for_human_review",
                finding_count=len(findings),
                draft=draft,
                policy_id=self.index.policy["id"],
                policy_sha256=self.index.sha256,
                execution_mode=self.mode,
            )
        raise PermanentError("Unknown tool")
