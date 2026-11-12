from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator

LABELS = [
    "vendor_profile",
    "commercial_offer",
    "insurance_certificate",
    "security_attestation",
    "other",
]


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=80, pattern=r"^[a-zA-Z0-9_.-]+$")
    name: str = Field(min_length=1, max_length=160)
    text: str = Field(min_length=1, max_length=30000)


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    vendor: str = Field(min_length=1, max_length=160)
    documents: list[Document] = Field(min_length=1, max_length=12)

    @field_validator("documents")
    @classmethod
    def unique_ids(cls, docs):
        if len({d.id for d in docs}) != len(docs):
            raise ValueError("Document IDs must be unique")
        return docs


class Evidence(BaseModel):
    document_id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str = Field(min_length=1)


class Fact(BaseModel):
    value: str
    evidence: Evidence


class Finding(BaseModel):
    code: str
    severity: Literal["blocker", "warning"] = "blocker"
    message: str
    policy_id: str
    policy_version: str
    clause_id: str
    policy_quote: str
    evidence: list[Evidence] = Field(default_factory=list)


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["approve_review", "request_clarification"]
    reviewer: str = Field(min_length=1, max_length=120)
    note: str = Field(min_length=5, max_length=2000)
    revision: int = Field(ge=0)


class ExtractionAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    facts: dict[str, Fact]


class PlannedCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tool: Literal[
        "classify_documents",
        "extract_fields",
        "retrieve_policy",
        "check_compliance",
        "draft_review",
    ]
    arguments: dict = Field(default_factory=dict)


class PermanentError(Exception):
    pass


class TransientError(Exception):
    pass


class Conflict(Exception):
    pass


class CandidateFact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class ExtractionCandidates(BaseModel):
    model_config = ConfigDict(extra="forbid")
    facts: dict[str, CandidateFact]
