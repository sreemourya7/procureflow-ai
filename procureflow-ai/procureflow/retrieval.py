import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class PolicyIndex:
    def __init__(self, path: Path):
        raw = path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.policy = json.loads(raw)
        self.clauses = self.policy["clauses"]
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True, stop_words="english"
        )
        self.matrix = self.vectorizer.fit_transform(
            [c["title"] + " " + c["text"] for c in self.clauses]
        )

    def search(self, query, k=3):
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        return [
            dict(
                **self.clauses[int(i)],
                score=float(scores[i]),
                policy_id=self.policy["id"],
                policy_version=self.policy["version"],
                sha256=self.sha256,
            )
            for i in np.argsort(-scores)[: min(k, len(scores))]
        ]

    def clause(self, identifier):
        return next(c for c in self.clauses if c["id"] == identifier)
