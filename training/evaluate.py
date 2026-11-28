"""Score a checkpoint on the held-out test split.

    python training/evaluate.py --adapter models/nllb-ethnic

Both directions of every ethnic <-> vi pair are scored with chrF++ and BLEU
(sacrebleu). The `copy` baseline returns the source unchanged; Tày shares
many letters with Vietnamese, so a model has to beat it to count.
Results go to reports/eval/<checkpoint>.json and a markdown table on stdout.
A full test run also writes the scores into the checkpoint metadata; the
server serves only directions that beat the copy baseline.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from finetune_nllb import both_directions, read_pairs

from app.checkpoints import read_metadata, write_metadata
from app.paths import default_nllb_base

NAMES = {
    "tyz": "Tày",
    "rad": "Ê Đê",
    "jra": "Jrai",
    "cjm": "Chăm",
    "hmn": "H'Mông",
    "bdq": "Ba Na",
    "mtq": "Mường",
    "khm": "Khmer",
    "vi": "Việt",
}
BUCKETS = ("1-7w", "8-15w", "16+w")


def length_bucket(text: str) -> str:
    words = len(text.split())
    if words <= 7:
        return "1-7w"
    if words <= 15:
        return "8-15w"
    return "16+w"


def repetition_rate(hypotheses: list[str], n: int = 3) -> float:
    """Fraction of outputs containing at least one repeated word n-gram."""
    if not hypotheses:
        return 0.0
    repeated = 0
    for hyp in hypotheses:
        words = hyp.casefold().split()
        if len(words) < n:
            continue
        grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        if len(grams) != len(set(grams)):
            repeated += 1
    return round(repeated / len(hypotheses), 4)


def length_ratio(hypotheses: list[str], references: list[str]) -> float:
    hyp_words = sum(len(h.split()) for h in hypotheses)
    ref_words = max(1, sum(len(r.split()) for r in references))
    return round(hyp_words / ref_words, 3)


def score(hypotheses: list[str], references: list[str]) -> dict:
    import sacrebleu

    return {
        "chrf": round(sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score, 2),
        "bleu": round(sacrebleu.corpus_bleu(hypotheses, [references]).score, 2),
    }


def score_by_length(sources: list[str], references: list[str], hypotheses: list[str]) -> dict[str, dict]:
    grouped: dict[str, tuple[list[str], list[str], list[str]]] = {
        b: ([], [], []) for b in BUCKETS
    }
    for src, ref, hyp in zip(sources, references, hypotheses):
        bucket = length_bucket(src)
        grouped[bucket][0].append(src)
        grouped[bucket][1].append(ref)
        grouped[bucket][2].append(hyp)

    out: dict[str, dict] = {}
    for bucket in BUCKETS:
        b_src, b_ref, b_hyp = grouped[bucket]
        if not b_src:
            continue
        m = score(b_hyp, b_ref)
        c = score(b_src, b_ref)
        out[bucket] = {
            "sentences": len(b_src),
            "chrf": m["chrf"],
            "bleu": m["bleu"],
            "copy_chrf": c["chrf"],
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", nargs="*", type=Path, default=[ROOT / "data" / "public" / "pairs.csv"])
    parser.add_argument("--adapter", type=Path, default=ROOT / "models" / "nllb-ethnic")
    parser.add_argument("--base", default=default_nllb_base())
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None, help="Số câu tối đa mỗi hướng")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--beams", type=int, default=4)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-unknown-license", action="store_true")
    args = parser.parse_args()

    from app.nllb import NllbEngine

    engine = NllbEngine(args.base, str(args.adapter), beams=args.beams, gated=False)
    by_direction: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for src_lang, tgt_lang, src, tgt in both_directions(read_pairs(args.data, args.split, args.allow_unknown_license)):
        if (src_lang, tgt_lang) in engine.pairs():
            by_direction.setdefault((src_lang, tgt_lang), []).append((src, tgt))

    results = []
    for (src_lang, tgt_lang), items in sorted(by_direction.items()):
        items = items[: args.limit] if args.limit else items
        sources = [src for src, _ in items]
        references = [tgt for _, tgt in items]
        started = time.time()
        hypotheses: list[str] = []
        for i in range(0, len(sources), args.batch):
            hypotheses.extend(engine.translate_batch(sources[i : i + args.batch], src_lang, tgt_lang))
        by_length = score_by_length(sources, references, hypotheses)
        row = {
            "direction": f"{src_lang}->{tgt_lang}",
            "label": f"{NAMES[src_lang]} → {NAMES[tgt_lang]}",
            "sentences": len(items),
            "model": score(hypotheses, references),
            "copy_baseline": score(sources, references),
            "by_length": by_length,
            "length_ratio": length_ratio(hypotheses, references),
            "repetition_rate": repetition_rate(hypotheses),
            "seconds": round(time.time() - started, 1),
            "samples": [
                {"source": s, "reference": r, "output": h}
                for s, r, h in list(zip(sources, references, hypotheses))[: args.samples]
            ],
        }
        print(json.dumps({k: v for k, v in row.items() if k != "samples"}, ensure_ascii=False))
        results.append(row)

    metadata = read_metadata(args.adapter)
    if args.split == "test" and not args.limit:
        metadata["evaluation"] = {
            row["direction"]: {
                "chrf": row["model"]["chrf"],
                "copy_chrf": row["copy_baseline"]["chrf"],
                "long_chrf": row["by_length"].get("16+w", {}).get("chrf"),
                "length_ratio": row["length_ratio"],
                "repetition_rate": row["repetition_rate"],
                "sentences": row["sentences"],
            }
            for row in results
        }
        write_metadata(args.adapter, metadata)
    report = {
        "checkpoint": args.adapter.name,
        "metadata": metadata,
        "split": args.split,
        "beams": args.beams,
        "results": results,
    }
    output = args.output or ROOT.parent / "reports" / "eval" / f"{args.adapter.name}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n| Cặp | Số câu test | chrF++ | chrF++ (16+w) | BLEU | chép nguồn | Len ratio | Rep3 |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in results:
        long_chrf = row["by_length"].get("16+w", {}).get("chrf", "-")
        print(
            f"| {row['label']} | {row['sentences']} | {row['model']['chrf']} | {long_chrf} | {row['model']['bleu']} "
            f"| {row['copy_baseline']['chrf']} | {row['length_ratio']} | {row['repetition_rate']} |"
        )
    print(f"\nĐã ghi {output}")


if __name__ == "__main__":
    main()
