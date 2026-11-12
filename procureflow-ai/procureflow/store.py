"""SQLite WAL, transactional checkpoints, leases with fencing, and an idempotent outbox."""

from __future__ import annotations
import hashlib
import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from .domain import Conflict


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS cases (
              id TEXT PRIMARY KEY, request_key TEXT UNIQUE NOT NULL, payload_hash TEXT NOT NULL,
              submission TEXT NOT NULL, state TEXT NOT NULL, snapshot TEXT NOT NULL,
              mode TEXT NOT NULL, policy_hash TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0,
              lease_until REAL NOT NULL DEFAULT 0, fence INTEGER NOT NULL DEFAULT 0,
              created REAL NOT NULL, updated REAL NOT NULL, error TEXT);
            CREATE TABLE IF NOT EXISTS events (
              seq INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL REFERENCES cases(id),
              at REAL NOT NULL, agent TEXT NOT NULL, tool TEXT NOT NULL, status TEXT NOT NULL,
              detail TEXT NOT NULL, duration_ms REAL NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS outbox (
              id TEXT PRIMARY KEY, case_id TEXT NOT NULL REFERENCES cases(id), kind TEXT NOT NULL,
              payload TEXT NOT NULL, status TEXT NOT NULL, UNIQUE(case_id,kind));
            CREATE INDEX IF NOT EXISTS event_case ON events(case_id,seq);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=10000")
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def transaction(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    def create(self, submission, key, mode, policy_hash):
        raw = json.dumps(submission, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(raw.encode()).hexdigest()
        now = time.time()
        with self.transaction() as db:
            existing = db.execute(
                "SELECT * FROM cases WHERE request_key=?", (key,)
            ).fetchone()
            if existing:
                if (
                    existing["payload_hash"] != digest
                    or existing["mode"] != mode
                    or existing["policy_hash"] != policy_hash
                ):
                    raise Conflict(
                        "Idempotency key was already used with different content, mode, or policy"
                    )
                return existing["id"], False
            cid = str(uuid.uuid4())
            db.execute(
                "INSERT INTO cases(id,request_key,payload_hash,submission,state,snapshot,mode,policy_hash,created,updated) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (cid, key, digest, raw, "queued", "{}", mode, policy_hash, now, now),
            )
            self._event(
                db, cid, "intake", "create_case", "completed", {"request_key": key}
            )
            return cid, True

    def _event(self, db, cid, agent, tool, status, detail, duration=0):
        db.execute(
            "INSERT INTO events(case_id,at,agent,tool,status,detail,duration_ms) VALUES(?,?,?,?,?,?,?)",
            (cid, time.time(), agent, tool, status, json.dumps(detail), duration),
        )

    def event(self, cid, agent, tool, status, detail, duration=0):
        with self.transaction() as db:
            self._event(db, cid, agent, tool, status, detail, duration)

    def get(self, cid):
        with self.connect() as db:
            row = db.execute("SELECT * FROM cases WHERE id=?", (cid,)).fetchone()
            if row is None:
                raise KeyError(cid)
            out = dict(row)
            for k in ["submission", "snapshot"]:
                out[k] = json.loads(out[k])
            out["events"] = [
                dict(r, detail=json.loads(r["detail"]))
                for r in db.execute(
                    "SELECT * FROM events WHERE case_id=? ORDER BY seq", (cid,)
                )
            ]
            out["outbox"] = [
                dict(r, payload=json.loads(r["payload"]))
                for r in db.execute("SELECT * FROM outbox WHERE case_id=?", (cid,))
            ]
            return out

    def list(self):
        with self.connect() as db:
            return [
                dict(r)
                for r in db.execute(
                    "SELECT id,state,mode,revision,created,updated,error FROM cases ORDER BY created DESC LIMIT 100"
                )
            ]

    def claim(self, cid):
        with self.transaction() as db:
            row = db.execute(
                "SELECT state,lease_until,fence FROM cases WHERE id=?", (cid,)
            ).fetchone()
            if row is None:
                raise KeyError(cid)
            if row["state"] in [
                "awaiting_review",
                "approved_review",
                "clarification_requested",
            ]:
                return None
            if row["lease_until"] > time.time():
                raise Conflict("Case already has an active worker lease")
            fence = row["fence"] + 1
            db.execute(
                "UPDATE cases SET state=?,lease_until=?,fence=?,error=NULL,updated=? WHERE id=?",
                ("running", time.time() + 300, fence, time.time(), cid),
            )
            return fence

    def _assert_fence(self, db, cid, fence):
        row = db.execute(
            "SELECT fence,lease_until FROM cases WHERE id=?", (cid,)
        ).fetchone()
        if row is None or row["fence"] != fence or row["lease_until"] < time.time():
            raise Conflict("Worker lease expired or was superseded")

    def checkpoint(self, cid, fence, snapshot, agent, tool, duration):
        with self.transaction() as db:
            self._assert_fence(db, cid, fence)
            final = "draft_review" in snapshot
            db.execute(
                "UPDATE cases SET snapshot=?,state=?,revision=revision+1,lease_until=?,updated=? WHERE id=?",
                (
                    json.dumps(snapshot),
                    "awaiting_review" if final else "running",
                    0 if final else time.time() + 300,
                    time.time(),
                    cid,
                ),
            )
            self._event(
                db, cid, agent, tool, "completed", {"checkpoint": tool}, duration
            )

    def fail(self, cid, fence, error):
        with self.transaction() as db:
            self._assert_fence(db, cid, fence)
            db.execute(
                "UPDATE cases SET state=?,lease_until=0,error=?,updated=? WHERE id=?",
                ("failed", str(error)[:400], time.time(), cid),
            )
            self._event(
                db, cid, "coordinator", "run", "failed", {"error": str(error)[:400]}
            )

    def draft(self, cid, fence, payload):
        with self.transaction() as db:
            self._assert_fence(db, cid, fence)
            identifier = hashlib.sha256((cid + ":clarification").encode()).hexdigest()[
                :24
            ]
            db.execute(
                "INSERT OR IGNORE INTO outbox(id,case_id,kind,payload,status) VALUES(?,?,?,?,?)",
                (identifier, cid, "clarification", json.dumps(payload), "draft"),
            )
            return dict(id=identifier, kind="clarification", status="draft")

    def decide(self, cid, decision):
        with self.transaction() as db:
            row = db.execute(
                "SELECT state,revision,snapshot FROM cases WHERE id=?", (cid,)
            ).fetchone()
            if row is None:
                raise KeyError(cid)
            if row["revision"] != decision.revision:
                raise Conflict("Stale revision: reload case before deciding")
            if row["state"] != "awaiting_review":
                raise Conflict("Case is not awaiting human review")
            snapshot = json.loads(row["snapshot"])
            findings = snapshot["check_compliance"]
            if decision.action == "approve_review" and findings:
                raise Conflict(
                    "Blocking findings must be clarified; approval is disabled"
                )
            if decision.action == "request_clarification" and not findings:
                raise Conflict("No findings to clarify")
            state = (
                "approved_review"
                if decision.action == "approve_review"
                else "clarification_requested"
            )
            db.execute(
                "UPDATE cases SET state=?,revision=revision+1,updated=? WHERE id=?",
                (state, time.time(), cid),
            )
            if decision.action == "request_clarification":
                db.execute(
                    "UPDATE outbox SET status='approved_for_export' WHERE case_id=?",
                    (cid,),
                )
            self._event(
                db, cid, "human", "review_decision", "completed", decision.model_dump()
            )
        return self.get(cid)
