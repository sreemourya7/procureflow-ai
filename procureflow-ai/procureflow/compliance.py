"""Policy predicates are code. The LLM cannot waive a rule or approve a supplier."""

from datetime import date
import re
from .domain import Finding
from .evidence import REQUIRED

QUERIES = {
    "P1": "supplier legal identity conflicting company names",
    "P2": "tax registration identifier format",
    "P3": "liability insurance coverage limit expiry review date",
    "P4": "total quoted price commercial budget",
    "P5": "required submission documents missing unsupported unverifiable fields",
    "P6": "security encryption at rest data retention days",
}


def check(submission, classified, extractions, index):
    p = index.policy
    findings = []
    docs = {d["id"]: d for d in submission["documents"]}
    by_kind = {}

    def add(code, message, clause, evidence=None):
        c = index.clause(clause)
        findings.append(
            Finding(
                code=code,
                message=message,
                policy_id=p["id"],
                policy_version=p["version"],
                clause_id=clause,
                policy_quote=c["text"],
                evidence=evidence or [],
            ).model_dump()
        )

    for row in classified:
        by_kind.setdefault(row["kind"], []).append(row["document_id"])
        doc = docs[row["document_id"]]
        suspicious = re.search(
            r"ignore\s+(?:all\s+)?previous\s+instructions|do\s+not\s+report\s+any\s+findings|system\s*prompt",
            doc["text"],
            re.I,
        )
        if suspicious:
            add(
                "SUSPICIOUS_INSTRUCTION",
                "Document contains an instruction-like passage; human inspection required.",
                "P5",
                [
                    dict(
                        document_id=doc["id"],
                        start=suspicious.start(),
                        end=suspicious.end(),
                        quote=suspicious.group(),
                    )
                ],
            )
        if row["kind"] == "other":
            add(
                "UNSUPPORTED_DOCUMENT",
                "Unrecognized or low-confidence document; route to manual review.",
                "P5",
            )
    for kind, keys in REQUIRED.items():
        ids = by_kind.get(kind, [])
        if not ids:
            add(
                "MISSING_" + kind.upper(),
                f"Required {kind.replace('_', ' ')} absent.",
                "P5",
            )
        if len(ids) > 1:
            add(
                "DUPLICATE_DOCUMENT_TYPE",
                f"Multiple {kind} documents; do not silently select one.",
                "P5",
            )
        for docid in ids:
            facts = extractions.get(docid, {})
            for key in keys:
                if key not in facts:
                    add(
                        "UNVERIFIABLE_FIELD",
                        f"{docid}: {key} could not be verified from source.",
                        "P5",
                    )

            def value(key):
                return facts.get(key, {}).get("value")

            def evidence(*keys):
                return [facts[k]["evidence"] for k in keys if k in facts]

            if value("vendor") and " ".join(
                value("vendor").casefold().split()
            ) != " ".join(submission["vendor"].casefold().split()):
                add(
                    "IDENTITY_MISMATCH",
                    f"{docid}: supplier identity differs from the submission.",
                    "P1",
                    evidence("vendor"),
                )
            if (
                kind == "vendor_profile"
                and value("tax_id")
                and not re.fullmatch(r"\d{2}-\d{7}", value("tax_id"))
            ):
                add(
                    "INVALID_TAX_ID",
                    "Tax registration format is invalid.",
                    "P2",
                    evidence("tax_id"),
                )
            try:
                if kind == "insurance_certificate":
                    if (
                        value("coverage")
                        and float(value("coverage").replace(",", ""))
                        < p["minimum_insurance_usd"]
                    ):
                        add(
                            "INSUFFICIENT_COVERAGE",
                            "Liability limit is below policy minimum.",
                            "P3",
                            evidence("coverage"),
                        )
                    if value("expiry") and date.fromisoformat(
                        value("expiry")
                    ) < date.fromisoformat(p["review_date"]):
                        add(
                            "EXPIRED_INSURANCE",
                            "Insurance expired before the review date.",
                            "P3",
                            evidence("expiry"),
                        )
                if (
                    kind == "commercial_offer"
                    and value("price")
                    and float(value("price").replace(",", "")) > p["budget_usd"]
                ):
                    add(
                        "OVER_BUDGET",
                        "Quoted total exceeds the procurement budget.",
                        "P4",
                        evidence("price"),
                    )
                if kind == "security_attestation":
                    if value("encryption") and value("encryption").lower() not in [
                        "yes",
                        "true",
                        "enabled",
                    ]:
                        add(
                            "ENCRYPTION_REQUIRED",
                            "Encryption at rest is not confirmed.",
                            "P6",
                            evidence("encryption"),
                        )
                    if (
                        value("retention")
                        and int(value("retention")) > p["maximum_retention_days"]
                    ):
                        add(
                            "RETENTION_EXCEEDED",
                            "Retention exceeds the allowed period.",
                            "P6",
                            evidence("retention"),
                        )
            except (ValueError, TypeError, OverflowError):
                add(
                    "UNVERIFIABLE_FIELD",
                    f"{docid}: field value has an unsupported format.",
                    "P5",
                )
    return findings
