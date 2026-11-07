import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path

from app.paths import models_dir
from app.textnorm import clean

ORDERS = (1, 2, 3, 4)


def ngrams(text: str) -> Counter:
    padded = f" {clean(text).casefold()} "
    grams: Counter = Counter()
    for n in ORDERS:
        for i in range(len(padded) - n + 1):
            grams[padded[i : i + n]] += 1
    return grams


class CharNgramModel:
    """Multinomial Naive Bayes over character 1-4 grams."""

    def __init__(self, data: dict) -> None:
        self.labels: list[str] = data["labels"]
        self.priors: dict[str, float] = data["priors"]
        self.unseen: dict[str, float] = data["unseen"]
        self.logprob: dict[str, list[float]] = data["logprob"]
        self.floors: dict[str, float] = data.get("floors", {})

    @classmethod
    def train(cls, samples: dict[str, list[str]], min_count: int = 2, max_features: int = 60000) -> "CharNgramModel":
        counts = {label: Counter() for label in samples}
        for label, texts in samples.items():
            for text in texts:
                counts[label].update(ngrams(text))
        total = Counter()
        for counter in counts.values():
            total.update(counter)
        vocab = [gram for gram, count in total.most_common(max_features) if count >= min_count]
        size = len(vocab)
        logprob: dict[str, list[float]] = {}
        unseen = {}
        for label, counter in counts.items():
            mass = sum(counter[gram] for gram in vocab) + size
            unseen[label] = math.log(1 / mass)
            for gram in vocab:
                logprob.setdefault(gram, []).append(round(math.log((counter[gram] + 1) / mass), 4))
        documents = sum(len(texts) for texts in samples.values())
        priors = {label: math.log(len(texts) / documents) for label, texts in samples.items()}
        return cls({"labels": list(samples), "priors": priors, "unseen": unseen, "logprob": logprob})

    def to_json(self) -> dict:
        return {
            "labels": self.labels,
            "priors": self.priors,
            "unseen": self.unseen,
            "logprob": self.logprob,
            "floors": self.floors,
        }

    def fit(self, text: str, label: str) -> float:
        """Mean log-probability per n-gram under one label, unseen n-grams included."""
        index = self.labels.index(label)
        grams = ngrams(text)
        total = sum(grams.values())
        score = 0.0
        for gram, count in grams.items():
            row = self.logprob.get(gram)
            score += count * (row[index] if row else self.unseen[label])
        return score / max(total, 1)

    def calibrate(self, samples: dict[str, list[str]], quantile: float = 0.02) -> None:
        """Store, per label, the fit that all but `quantile` of its held-out texts reach."""
        for label, texts in samples.items():
            if label not in self.labels or not texts:
                continue
            fits = sorted(self.fit(text, label) for text in texts)
            self.floors[label] = round(fits[int(len(fits) * quantile)], 4)

    def predict(self, text: str) -> tuple[str, float]:
        """Return the best label and its posterior.

        Unseen n-grams are skipped for the choice: their smoothing mass differs
        per label and would push unknown text to the label with least data.
        Text that fits the chosen label worse than its calibrated floor is
        another language, and the confidence is 0.
        """
        scores = [self.priors[label] for label in self.labels]
        for gram, count in ngrams(text).items():
            row = self.logprob.get(gram)
            if row:
                for i in range(len(self.labels)):
                    scores[i] += count * row[i]
        best = max(scores)
        weights = [math.exp(score - best) for score in scores]
        index = scores.index(best)
        label = self.labels[index]
        floor = self.floors.get(label)
        if floor is not None and self.fit(text, label) < floor:
            return label, 0.0
        return label, weights[index] / sum(weights)


def langid_path() -> Path:
    return models_dir() / "langid.json"


@lru_cache
def load_langid() -> CharNgramModel | None:
    path = langid_path()
    if not path.is_file():
        return None
    return CharNgramModel(json.loads(path.read_text(encoding="utf-8")))
