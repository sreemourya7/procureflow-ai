from __future__ import annotations
import json
import os
import secrets
from pathlib import Path
from fastapi import FastAPI, Depends, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from .domain import Submission, Decision, Conflict
from .engine import Engine


def create_app(root: Path, db: Path | None = None):
    root = Path(root)
    mode = os.getenv("PROCUREFLOW_MODE", "offline")
    engine = Engine(root, db, mode=mode)
    app = FastAPI(
        title="ProcureFlow AI",
        version="0.1.0",
        description="Synthetic procurement review workbench. Local demo; not a procurement authority.",
    )
    app.state.engine = engine
    api_token = os.getenv("PROCUREFLOW_API_TOKEN", "")
    review_token = os.getenv("PROCUREFLOW_REVIEWER_TOKEN", "")

    def auth(authorization: str | None = Header(default=None)):
        if api_token and not secrets.compare_digest(
            authorization or "", "Bearer " + api_token
        ):
            raise HTTPException(401, "Valid API bearer token required")

    def reviewer(
        authorization: str | None = Header(default=None),
        x_reviewer_token: str | None = Header(default=None),
    ):
        auth(authorization)
        if review_token and not secrets.compare_digest(
            x_reviewer_token or "", review_token
        ):
            raise HTTPException(403, "Valid reviewer token required")

    @app.exception_handler(Conflict)
    async def conflict_handler(request, exc):
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(KeyError)
    async def not_found(request, exc):
        return JSONResponse(status_code=404, content={"detail": "Case not found"})

    @app.middleware("http")
    async def headers(request: Request, call_next):
        # JSON submissions are intentionally bounded; document text limits are also validated.
        try:
            declared_size = int(request.headers.get("content-length", "0") or 0)
        except ValueError:
            return JSONResponse(
                status_code=400, content={"detail": "Invalid Content-Length"}
            )
        if declared_size > 500000:
            return JSONResponse(
                status_code=413, content={"detail": "Submission exceeds 500 KB"}
            )
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'"
        )
        if request.url.path == "/docs":
            # Swagger uses a pinned CDN bundle and an inline initialization script.
            # The application workbench keeps its stricter self-only policy.
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
                "img-src 'self' data: https://fastapi.tiangolo.com; frame-ancestors 'none'"
            )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "mode": mode,
            "synthetic_policy": True,
            "authentication_enabled": bool(api_token),
            "reviewer_token_enabled": bool(review_token),
        }

    @app.get("/api/config", dependencies=[Depends(auth)])
    def config():
        return {
            "mode": mode,
            "policy": engine.index.policy,
            "policy_sha256": engine.index.sha256,
            "local_development": not bool(api_token),
            "reviewer_auth_enabled": bool(review_token),
        }

    @app.get("/api/example", dependencies=[Depends(auth)])
    def example():
        return json.loads((root / "data/examples/submission.json").read_text())

    @app.get("/api/results", dependencies=[Depends(auth)])
    def results():
        return {
            p.stem: json.loads(p.read_text())
            for p in [root / "results/classifier.json", root / "results/benchmark.json"]
            if p.exists()
        }

    @app.get("/api/cases", dependencies=[Depends(auth)])
    def cases():
        return engine.store.list()

    @app.post("/api/cases", dependencies=[Depends(auth)])
    def create(
        submission: Submission,
        idempotency_key: str = Header(min_length=1, max_length=120),
    ):
        cid, created = engine.create(submission.model_dump(), idempotency_key)
        return JSONResponse(
            status_code=201 if created else 200, content={"id": cid, "created": created}
        )

    @app.get("/api/cases/{cid}", dependencies=[Depends(auth)])
    def get(cid: str):
        return engine.store.get(cid)

    @app.post("/api/cases/{cid}/run", dependencies=[Depends(auth)])
    def run(cid: str):
        try:
            return Engine(root, db, mode=mode).run(cid)
        except (Conflict, KeyError):
            raise
        except Exception:
            raise HTTPException(
                502,
                "Workflow failed; inspect case trace and resume after resolving the cause",
            )

    @app.post("/api/cases/{cid}/decision", dependencies=[Depends(reviewer)])
    def decide(cid: str, decision: Decision):
        return engine.store.decide(cid, decision)

    @app.get("/api/cases/{cid}/export", dependencies=[Depends(reviewer)])
    def export(cid: str):
        case = engine.store.get(cid)
        approved = [d for d in case["outbox"] if d["status"] == "approved_for_export"]
        if not approved:
            raise Conflict("A human must approve a clarification request before export")
        return {
            "case_id": cid,
            "drafts": approved,
            "delivery": "No external message has been sent",
        }

    static = Path(__file__).parent / "static"
    app.mount("/static", StaticFiles(directory=static), name="static")

    @app.get("/")
    def home():
        return FileResponse(static / "index.html")

    return app
