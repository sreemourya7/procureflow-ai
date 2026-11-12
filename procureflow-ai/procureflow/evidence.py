"""Extracted facts must point to exact source offsets; no uncited generated fields."""

import re
from .domain import Evidence, Fact

PATTERNS = {
    "vendor": r"^(?:Vendor|Legal name|Company)\s*:\s*([^\r\n]+)",
    "tax_id": r"^(?:Tax ID|TIN|Tax registration)\s*:\s*([^\r\n]+)",
    "price": r"^(?:Total price|Amount USD|Contract value)\s*:\s*(?:USD\s*|\$\s*)?([\d,]+(?:\.\d{1,2})?)\s*$",
    "coverage": r"^(?:Coverage limit|Liability limit|Insured amount)\s*:\s*(?:USD\s*|\$\s*)?([\d,]+(?:\.\d{1,2})?)\s*$",
    "expiry": r"^(?:Expiry date|Valid until|Expiration)\s*:\s*([^\r\n]+)",
    "encryption": r"^(?:Encryption at rest|Storage encryption|Encryption)\s*:\s*([^\r\n]+)",
    "retention": r"^(?:Retention days|Data retention|Retention period)\s*:\s*(\d+)(?:\s*days)?\s*$",
}
REQUIRED = {
    "vendor_profile": ["vendor", "tax_id"],
    "commercial_offer": ["vendor", "price"],
    "insurance_certificate": ["vendor", "coverage", "expiry"],
    "security_attestation": ["vendor", "encryption", "retention"],
}


def valid_evidence(e, documents):
    try:
        e = Evidence.model_validate(e)
        text = next(d["text"] for d in documents if d["id"] == e.document_id)
        return e.start < e.end <= len(text) and text[e.start : e.end] == e.quote
    except (ValueError, StopIteration, KeyError, TypeError):
        return False


def extract(document, kind):
    facts = {}
    for key in REQUIRED.get(kind, []):
        match = re.search(PATTERNS[key], document["text"], flags=re.I | re.M)
        if match:
            raw = match.group(1)
            value = raw.strip()
            start = match.start(1) + len(raw) - len(raw.lstrip())
            facts[key] = Fact(
                value=value,
                evidence=Evidence(
                    document_id=document["id"],
                    start=start,
                    end=start + len(value),
                    quote=value,
                ),
            ).model_dump()
    return facts


def verify_facts(facts, document, kind):
    """LLM facts are extractive: the value must be the exact cited substring."""
    clean = {}
    for key, fact in facts.items():
        if key not in REQUIRED.get(kind, []):
            continue
        f = Fact.model_validate(fact).model_dump()
        if (
            not valid_evidence(f["evidence"], [document])
            or f["value"] != f["evidence"]["quote"]
        ):
            raise ValueError("Unsupported extracted value or invalid source citation")
        clean[key] = f
    return clean
