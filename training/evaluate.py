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

from app.checkpoints import read_metadata, write_metadata  # noqa: E402
from app.paths import default_nllb_base  # noqa: E402
from finetune_nllb import both_directions, read_pairs  # noqa: E402

NAMES = {"tyz": "Tày", "rad": "Ê Đê", "jra": "Jrai", "cjm": "Chăm", "hmn": "H'Mông", "vi": "Việt"}


def score(hypotheses: list[str], references: list[str]) -> dict:
    import sacrebleu

    return {
        "chrf": round(sacrebleu.corpus_chrf(hypotheses, [references], word_order=2).score, 2),
        "bleu": round(sacrebleu.corpus_bleu(hypotheses, [references]).score, 2),
    }


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
        row = {
            "direction": f"{src_lang}->{tgt_lang}",
            "label": f"{NAMES[src_lang]} → {NAMES[tgt_lang]}",
            "sentences": len(items),
            "model": score(hypotheses, references),
            "copy_baseline": score(sources, references),
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

    print("\n| Cặp | Số câu test | chrF++ | BLEU | chrF++ chép nguồn |")
    print("| --- | --- | --- | --- | --- |")
    for row in results:
        print(
            f"| {row['label']} | {row['sentences']} | {row['model']['chrf']} | {row['model']['bleu']} "
            f"| {row['copy_baseline']['chrf']} |"
        )
    print(f"\nĐã ghi {output}")


if __name__ == "__main__":
    main()
