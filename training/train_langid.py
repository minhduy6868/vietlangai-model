"""Train the character n-gram language identifier.

    python training/train_langid.py

Classes are the ethnic languages present in the reviewed CSVs plus
Vietnamese (taken from the Vietnamese side of the same pairs). English stays
on the lexicon rule in app.detect. Writes models/langid.json.
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from app.langid import CharNgramModel, langid_path  # noqa: E402
from finetune_nllb import read_pairs  # noqa: E402


def collect(paths: list[Path], split: str, cap: int, seed: int, unknown_ok: bool = False) -> dict[str, list[str]]:
    samples: dict[str, set[str]] = {}
    for src_lang, tgt_lang, src, tgt in read_pairs(paths, split, unknown_ok):
        samples.setdefault(src_lang, set()).add(src)
        samples.setdefault(tgt_lang, set()).add(tgt)
    rng = random.Random(seed)
    out = {}
    for label, texts in sorted(samples.items()):
        items = sorted(texts)
        rng.shuffle(items)
        out[label] = items[:cap]
    return out


def answers(model: CharNgramModel, text: str, threshold: float = 0.9) -> str:
    label, confidence = model.predict(text)
    return label if confidence >= threshold else "und"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="*", type=Path, default=[ROOT / "data" / "public" / "pairs.csv"])
    parser.add_argument("--cap", type=int, default=20000)
    parser.add_argument("--output", type=Path, default=langid_path())
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--allow-unknown-license", action="store_true")
    args = parser.parse_args()

    unknown_ok = args.allow_unknown_license
    train = collect(args.data, "train", args.cap, args.seed, unknown_ok)
    test = collect(args.data, "test", 2000, args.seed, unknown_ok)
    print({label: len(texts) for label, texts in train.items()})
    model = CharNgramModel.train(train)
    model.calibrate(collect(args.data, "dev", 2000, args.seed, unknown_ok))
    print({"floors": model.floors})

    report = {}
    for label, texts in test.items():
        if label not in model.labels:
            continue
        long_texts = [t for t in texts if len(t.split()) >= 3]
        hits = sum(answers(model, t) == label for t in texts)
        long_hits = sum(answers(model, t) == label for t in long_texts)
        report[label] = {
            "accuracy": round(hits / max(len(texts), 1), 3),
            "accuracy_3_words_plus": round(long_hits / max(len(long_texts), 1), 3),
            "sentences": len(texts),
        }
    print(json.dumps(report, ensure_ascii=False))
    foreign = {
        "en": ["Where is the nearest hospital please", "I would like to buy some rice", "Thank you very much"],
        "hmn": ["Kuv mus tsev", "Koj puas noj mov lawm", "Nyob zoo os"],
    }
    print({lang: [model.predict(t) for t in texts] for lang, texts in foreign.items()})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model.to_json(), ensure_ascii=False), encoding="utf-8")
    print(f"Đã ghi {args.output}")


if __name__ == "__main__":
    main()
