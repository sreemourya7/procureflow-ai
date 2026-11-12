import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import httpx
import pytest
from fastapi.testclient import TestClient
from procureflow.api import create_app
from procureflow.domain import Conflict, Decision, Submission, TransientError
from procureflow.engine import Engine
from procureflow.evaluation import SimulatedCrash
from procureflow.evidence import valid_evidence, verify_facts, extract
from procureflow.providers import LLMProvider
from procureflow.synthetic import SCENARIOS, classification_data, workflow_fixture

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def engine(tmp_path):
    return Engine(ROOT, tmp_path / "test.db", sleep=lambda _: None)


def create_run(engine, scenario="multiple_issues", index=999):
    f = workflow_fixture(index, scenario)
    cid, _ = engine.create(f["submission"], str(index))
    return f, engine.run(cid)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_authored_scenario_oracle(engine, scenario):
    f, result = create_run(engine, scenario)
    assert result["state"] == "awaiting_review"
    assert {r["code"] for r in result["snapshot"]["check_compliance"]} == set(
        f["expected_codes"]
    )


def test_template_and_text_splits_disjoint():
    data = classification_data()
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        assert not {r["family"] for r in data[a]} & {r["family"] for r in data[b]}
        assert not {r["text"] for r in data[a]} & {r["text"] for r in data[b]}


def test_idempotency_rejects_changed_payload(engine):
    sub = workflow_fixture()["submission"]
    a, fresh = engine.create(sub, "key")
    b, fresh2 = engine.create(sub, "key")
    assert a == b and fresh and not fresh2
    with pytest.raises(Conflict):
        engine.create(dict(sub, vendor="Other vendor"), "key")


def test_concurrent_create(engine):
    sub = workflow_fixture()["submission"]
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(lambda _: engine.create(sub, "race")[0], range(32)))
    assert len(set(ids)) == 1


def test_fence_rejects_stale_worker(engine):
    cid, _ = engine.create(workflow_fixture()["submission"], "lease")
    first = engine.store.claim(cid)
    with pytest.raises(Conflict):
        engine.store.claim(cid)
    with engine.store.transaction() as db:
        db.execute("UPDATE cases SET lease_until=0 WHERE id=?", (cid,))
    second = engine.store.claim(cid)
    assert second > first
    with pytest.raises(Conflict):
        engine.store.checkpoint(cid, first, {}, "bad", "bad", 0)
    with pytest.raises(Conflict):
        engine.store.draft(cid, first, {"bad": True})


@pytest.mark.parametrize(
    "tool",
    [
        "classify_documents",
        "extract_fields",
        "retrieve_policy",
        "check_compliance",
        "draft_review",
    ],
)
def test_crash_resume_preserves_single_outbox(tmp_path, tool):
    fired = [False]

    def fault(point):
        if point == "after:" + tool and not fired[0]:
            fired[0] = True
            raise SimulatedCrash()

    db = tmp_path / "crash.db"
    engine = Engine(ROOT, db, fault=fault)
    cid, _ = engine.create(
        workflow_fixture(999, "multiple_issues")["submission"], "crash"
    )
    with pytest.raises(SimulatedCrash):
        engine.run(cid)
    with engine.store.transaction() as connection:
        connection.execute("UPDATE cases SET lease_until=0 WHERE id=?", (cid,))
    result = Engine(ROOT, db).run(cid)
    assert result["state"] == "awaiting_review" and len(result["outbox"]) == 1
    assert (
        len(
            [
                e
                for e in result["events"]
                if e["tool"] == tool and e["status"] == "completed"
            ]
        )
        == 1
    )


def test_transient_retry_and_bounded_exhaustion(tmp_path):
    calls = [0]

    def fault(point):
        if point == "before:classify_documents":
            calls[0] += 1
            raise TransientError("timeout")

    e = Engine(ROOT, tmp_path / "retry.db", fault=fault, sleep=lambda _: None)
    cid, _ = e.create(workflow_fixture()["submission"], "retry")
    with pytest.raises(TransientError):
        e.run(cid)
    assert calls[0] == 3 and e.store.get(cid)["state"] == "failed"
    assert Engine(ROOT, tmp_path / "retry.db").run(cid)["state"] == "awaiting_review"


def test_citations_are_exact_source_slices(engine):
    fixture, result = create_run(engine)
    for facts in result["snapshot"]["extract_fields"].values():
        for f in facts.values():
            assert valid_evidence(f["evidence"], fixture["submission"]["documents"])
    bad = {"document_id": "doc-1", "start": 0, "end": 4, "quote": "FAKE"}
    assert not valid_evidence(bad, fixture["submission"]["documents"])


def test_llm_value_cannot_exceed_quote():
    doc = workflow_fixture()["submission"]["documents"][0]
    facts = extract(doc, "vendor_profile")
    facts["vendor"]["value"] = "Fabricated"
    with pytest.raises(ValueError):
        verify_facts(facts, doc, "vendor_profile")


def test_duplicate_document_ids_rejected():
    f = workflow_fixture()["submission"]
    f["documents"][1]["id"] = f["documents"][0]["id"]
    with pytest.raises(ValueError):
        Submission.model_validate(f)


def test_human_gate_and_optimistic_revision(engine):
    _, case = create_run(engine)
    decision = Decision(
        action="approve_review",
        reviewer="Test reviewer",
        note="Checked source evidence",
        revision=case["revision"],
    )
    with pytest.raises(Conflict):
        engine.store.decide(case["id"], decision)
    decision.action = "request_clarification"
    decision.revision -= 1
    with pytest.raises(Conflict):
        engine.store.decide(case["id"], decision)
    decision.revision = case["revision"]
    approved = engine.store.decide(case["id"], decision)
    assert approved["state"] == "clarification_requested"
    assert approved["outbox"][0]["status"] == "approved_for_export"
    with pytest.raises(Conflict):
        engine.store.decide(case["id"], decision)


def test_clean_review_still_requires_human(engine):
    _, case = create_run(engine, "clean")
    assert case["state"] == "awaiting_review" and not case["outbox"]
    result = engine.store.decide(
        case["id"],
        Decision(
            action="approve_review",
            reviewer="Human",
            note="All evidence inspected",
            revision=case["revision"],
        ),
    )
    assert result["state"] == "approved_review"


def test_replay_terminal_case_does_not_add_events(engine):
    _, case = create_run(engine)
    again = engine.run(case["id"])
    assert (
        len(case["events"]) == len(again["events"])
        and case["revision"] == again["revision"]
    )


def test_policy_version_guard(engine):
    cid, _ = engine.create(workflow_fixture()["submission"], "policy")
    engine.index.sha256 = "changed"
    with pytest.raises(Conflict):
        engine.run(cid)


def test_api_authorization_and_export(tmp_path, monkeypatch):
    monkeypatch.setenv("PROCUREFLOW_API_TOKEN", "test-api")
    monkeypatch.setenv("PROCUREFLOW_REVIEWER_TOKEN", "test-review")
    app = create_app(ROOT, tmp_path / "api.db")
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/api/cases").status_code == 401
    headers = {"Authorization": "Bearer test-api", "Idempotency-Key": "api-key"}
    fixture = workflow_fixture(999, "multiple_issues")
    response = client.post("/api/cases", headers=headers, json=fixture["submission"])
    assert response.status_code == 201
    cid = response.json()["id"]
    assert (
        client.post(
            "/api/cases", headers=headers, json=fixture["submission"]
        ).status_code
        == 200
    )
    result = client.post("/api/cases/" + cid + "/run", headers=headers).json()
    assert result["state"] == "awaiting_review"
    assert (
        client.get("/api/cases/" + cid + "/export", headers=headers).status_code == 403
    )
    headers["X-Reviewer-Token"] = "test-review"
    assert (
        client.get("/api/cases/" + cid + "/export", headers=headers).status_code == 409
    )
    decision = dict(
        action="request_clarification",
        reviewer="Test",
        note="Reviewed all sources",
        revision=result["revision"],
    )
    assert (
        client.post(
            "/api/cases/" + cid + "/decision", headers=headers, json=decision
        ).status_code
        == 200
    )
    assert (
        client.get("/api/cases/" + cid + "/export", headers=headers).status_code == 200
    )
    assert (
        client.get("/").status_code == 200
        and client.get("/static/app.js").status_code == 200
    )
    assert "frame-ancestors" in client.get("/").headers["Content-Security-Policy"]


def test_llm_http_contract_and_grounding(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "contract-test")
    monkeypatch.setenv("LLM_API_KEY", "not-a-real-key")
    doc = workflow_fixture()["submission"]["documents"][0]
    facts = extract(doc, "vendor_profile")

    def handler(request):
        body = json.loads(request.content)
        if "tools" in body:
            message = {
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "classify_documents", "arguments": "{}"},
                    }
                ]
            }
        else:
            message = {
                "content": json.dumps(
                    {
                        "facts": {
                            k: {"value": v["value"], "quote": v["value"]}
                            for k, v in facts.items()
                        }
                    }
                )
            }
        return httpx.Response(
            200,
            json={
                "choices": [{"message": message}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            },
        )

    provider = LLMProvider(httpx.Client(transport=httpx.MockTransport(handler)))
    assert provider.choose(["classify_documents"], []) == "classify_documents"
    assert provider.extract(doc, "vendor_profile") == facts
    assert provider.usage["calls"] == 2


def test_llm_rejects_disallowed_tool(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "contract-test")
    monkeypatch.setenv("LLM_API_KEY", "not-a-real-key")

    def handler(request):
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {"function": {"name": "send_email", "arguments": "{}"}}
                            ]
                        }
                    }
                ]
            },
        )

    provider = LLMProvider(httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(TransientError):
        provider.choose(["classify_documents"], [])


def test_llm_rate_limit_is_retryable(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "contract-test")
    monkeypatch.setenv("LLM_API_KEY", "not-a-real-key")
    provider = LLMProvider(
        httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(429)))
    )
    with pytest.raises(TransientError):
        provider.choose(["classify_documents"], [])


class FakeLiveProvider:
    """Contract test only; never counted as measured real LLM execution."""

    def __init__(self):
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def choose(self, ready, completed):
        self.usage["calls"] += 1
        return ready[-1]

    def extract(self, doc, kind):
        self.usage["calls"] += 1
        return extract(doc, kind)

    def cost(self):
        return None


def test_live_orchestration_alternate_valid_order(tmp_path):
    provider = FakeLiveProvider()
    engine = Engine(ROOT, tmp_path / "live.db", mode="live", provider=provider)
    f, result = create_run(engine)
    selected = [e["tool"] for e in result["events"] if e["status"] == "selected"]
    assert selected[0] == "retrieve_policy" and result["state"] == "awaiting_review"
    assert {x["code"] for x in result["snapshot"]["check_compliance"]} == set(
        f["expected_codes"]
    )


def test_llm_rejects_invented_quotes(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "contract-test")
    monkeypatch.setenv("LLM_API_KEY", "not-a-real-key")
    answer = {"facts": {"vendor": {"value": "Imaginary LLC", "quote": "Imaginary LLC"}}}
    provider = LLMProvider(
        httpx.Client(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    json={"choices": [{"message": {"content": json.dumps(answer)}}]},
                )
            )
        )
    )
    with pytest.raises(TransientError):
        provider.extract(
            workflow_fixture()["submission"]["documents"][0], "vendor_profile"
        )


def test_schema_guard_rejects_unrelated_labeled_prose(engine):
    docs = [
        {
            "id": "note",
            "name": "notes.txt",
            "text": "Team workshop notes. A supplier discussed pricing and insurance; no actual submission is provided.",
        }
    ]
    result = engine.router.classify(docs)
    assert result[0]["kind"] == "other"


def test_llm_request_budget_fails_permanently(monkeypatch):
    from procureflow.domain import PermanentError

    monkeypatch.setenv("LLM_MODEL", "contract-test")
    monkeypatch.setenv("LLM_API_KEY", "not-a-real-key")
    provider = LLMProvider(
        httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(429)))
    )
    provider.attempts = 24
    with pytest.raises(PermanentError):
        provider.choose(["classify_documents"], [])
