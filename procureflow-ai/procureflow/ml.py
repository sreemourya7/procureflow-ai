"""Small trainable neural text router; classical baselines are evaluated honestly."""

from __future__ import annotations
import hashlib
import json
import random
import re
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from .domain import LABELS

BUCKETS = 8192


def token_ids(text):
    words = re.findall(r"[a-z]+", text.lower())
    tokens = words + [a + "_" + b for a, b in zip(words, words[1:])]
    return [
        1
        + int.from_bytes(hashlib.blake2b(w.encode(), digest_size=4).digest(), "little")
        % (BUCKETS - 1)
        for w in tokens
    ] or [0]


def batch(texts):
    flat = []
    offsets = []
    for text in texts:
        offsets.append(len(flat))
        flat.extend(token_ids(text))
    return torch.tensor(flat, dtype=torch.long), torch.tensor(offsets, dtype=torch.long)


class TextRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.EmbeddingBag(BUCKETS, 64, mode="mean")
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.head = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.15), nn.Linear(64, len(LABELS))
        )

    def forward(self, tokens, offsets):
        return self.head(self.embedding(tokens, offsets))


def keyword_predict(text):
    scores = []
    t = text.lower()
    keywords = [
        ["vendor", "supplier", "registration", "company", "tax", "tin"],
        ["price", "pricing", "commercial", "quote", "quotation", "contract value"],
        ["insurance", "liability", "coverage", "insured", "indemnity", "expiration"],
        ["security", "encryption", "retention", "safeguards", "cybersecurity"],
        ["notes", "agenda", "newsletter", "brochure", "receipt", "success story"],
    ]
    for words in keywords:
        scores.append(sum(t.count(w) for w in words))
    return LABELS[int(np.argmax(scores))]


def metrics(y, p):
    return dict(
        accuracy=float(accuracy_score(y, p)),
        macro_f1=float(f1_score(y, p, labels=LABELS, average="macro", zero_division=0)),
        confusion_matrix=confusion_matrix(y, p, labels=LABELS).tolist(),
    )


def train(root: Path, seeds=(7, 19, 41), epochs=28):
    from .synthetic import classification_data

    data = classification_data()
    train_rows = data["train"]
    val = data["validation"]
    test = data["test"]
    torch.set_num_threads(2)
    results = []
    best = None
    texts = [r["text"] for r in train_rows]
    y = torch.tensor([LABELS.index(r["label"]) for r in train_rows])
    vx = batch([r["text"] for r in val])
    vy = torch.tensor([LABELS.index(r["label"]) for r in val])
    tx = batch([r["text"] for r in test])
    truth = [r["label"] for r in test]
    for seed in seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = TextRouter()
        opt = torch.optim.AdamW(model.parameters(), lr=0.008, weight_decay=0.01)
        started = time.perf_counter()
        best_state = None
        best_loss = float("inf")
        for epoch in range(epochs):
            model.train()
            order = torch.randperm(len(texts))
            for idx in order.split(96):
                x = batch([texts[i] for i in idx.tolist()])
                opt.zero_grad()
                loss = nn.functional.cross_entropy(model(*x), y[idx])
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = float(nn.functional.cross_entropy(model(*vx), vy))
            if vl < best_loss:
                best_loss = vl
                best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            vlogits = model(*vx)
            temps = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
            temperature = min(
                temps, key=lambda t: float(nn.functional.cross_entropy(vlogits / t, vy))
            )
            probs = (model(*tx) / temperature).softmax(-1).numpy()
        pred = [LABELS[i] for i in probs.argmax(-1)]
        row = dict(
            seed=seed,
            epochs=epochs,
            selected_validation_loss=best_loss,
            temperature=temperature,
            train_seconds=time.perf_counter() - started,
            **metrics(truth, pred),
        )
        results.append(row)
        # Checkpoint choice is based ONLY on validation loss, never test scores.
        if best is None or best_loss < best[0]:
            best = (best_loss, best_state, temperature, seed)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    features = vectorizer.fit_transform(texts)
    baseline = LogisticRegression(max_iter=500, random_state=7).fit(
        features, [r["label"] for r in train_rows]
    )
    classic = metrics(
        truth, baseline.predict(vectorizer.transform([r["text"] for r in test]))
    )
    keyword = metrics(truth, [keyword_predict(r["text"]) for r in test])
    dest = root / "artifacts"
    dest.mkdir(exist_ok=True)
    torch.save(
        dict(
            state_dict=best[1],
            temperature=best[2],
            seed=best[3],
            labels=LABELS,
            buckets=BUCKETS,
            threshold=0.55,
        ),
        dest / "router.pt",
    )
    (dest / "router_metadata.json").write_text(
        json.dumps(
            dict(
                labels=LABELS,
                architecture="EmbeddingBag(8192,64) → Linear(64,64) → ReLU → Dropout(.15) → Linear(64,5)",
                parameters=sum(p.numel() for p in model.parameters()),
                selected_seed=best[3],
                selection="minimum validation loss",
                temperature=best[2],
                abstention_threshold=0.55,
            ),
            indent=2,
        )
    )
    report = dict(
        synthetic=True,
        split="disjoint template families; not real-world validation",
        samples={k: len(v) for k, v in data.items()},
        labels=LABELS,
        pytorch_runs=results,
        pytorch_mean_macro_f1=float(np.mean([r["macro_f1"] for r in results])),
        pytorch_std_macro_f1=float(np.std([r["macro_f1"] for r in results])),
        keyword_baseline=keyword,
        tfidf_logistic_baseline=classic,
        checkpoint_selection="minimum validation loss; test set not used for selection",
    )
    (root / "results/classifier.json").write_text(json.dumps(report, indent=2))
    return report


class Router:
    def __init__(self, path: Path):
        torch.set_num_threads(2)
        state = torch.load(path, map_location="cpu", weights_only=True)
        self.model = TextRouter()
        self.model.load_state_dict(state["state_dict"])
        self.model.eval()
        self.temperature = state["temperature"]
        self.threshold = state["threshold"]

    def classify(self, docs):
        with torch.inference_mode():
            probs = (
                (self.model(*batch([d["text"] for d in docs])) / self.temperature)
                .softmax(-1)
                .numpy()
            )
        result = []
        for doc, p in zip(docs, probs):
            idx = int(p.argmax())
            confidence = float(p[idx])
            label = LABELS[idx] if confidence >= self.threshold else "other"
            # Structured-intake contract: a type needs at least two supported field
            # labels. This guard rejects unrelated prose despite confident logits.
            from .evidence import REQUIRED, PATTERNS

            support = sum(
                bool(re.search(PATTERNS[key], doc["text"], flags=re.I | re.M))
                for key in REQUIRED.get(label, [])
            )
            schema_rejected = label != "other" and support < 2
            if schema_rejected:
                label = "other"
            result.append(
                dict(
                    document_id=doc["id"],
                    kind=label,
                    confidence=confidence,
                    abstained=confidence < self.threshold or schema_rejected,
                    schema_rejected=schema_rejected,
                )
            )
        return result
