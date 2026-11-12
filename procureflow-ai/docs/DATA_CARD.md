# Data card

All bundled documents and vendor identities are generated, and the procurement
policy is invented. No jurisdiction's legal policy is represented.

- Classifier: 1,200 training / 300 validation / 500 development-test documents.
- Classes: company profile, commercial offer, insurance certificate, security
  attestation, and other. Counts are balanced.
- Families: train 0–3, validation 4, test 5–6 for each class; no family overlap.
- Variation: supplier names, amounts, addresses, dates, whitespace and benign
  cross-topic text. File names are excluded from classifier features.
- Workflow: 280 cases, 20 each across 14 scenarios. Expected finding codes are
  authored in the fixture generator, independently of compliance implementation.
- Retrieval: 18 authored paraphrases against six fictional policy clauses.
- Reliability: 30 simulated crashes, 30 transient failures, and 100 concurrent
  duplicate-create attempts (8 threads), plus 10 conflicting replays.

The synthetic generator and model were developed together. Held-out template scores
are development measurements, not a blind external benchmark. The initial exploratory
classifier results are retained in `results/development_initial_classifier.json`.
Schema compatibility was added after an unsupported attachment was misrouted.

No actual procurement reviewers, documents, vendors, customer approvals, business
savings, patient data, or government deployments were used. Structured labels make
the offline task easier than general document understanding. Common and unknown
failure modes on real documents remain unmeasured.
