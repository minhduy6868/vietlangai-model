"""Compare evaluated NLLB checkpoints and eval reports side by side."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DIRECTIONS = (
    "bdq->vi",
    "vi->bdq",
    "jra->vi",
    "vi->jra",
    "mtq->vi",
    "vi->mtq",
    "rad->vi",
    "vi->rad",
    "tyz->vi",
    "vi->tyz",
)
SCORE_FORMULA = "mean_chrf + 0.35 * mean_long_chrf + 2.0 * languages - 20.0 * mean_repetition"


def _parse_arg(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, raw_path = spec.split("=", 1)
        if label.strip() and raw_path.strip():
            return label.strip(), Path(raw_path.strip())
    path = Path(spec.strip())
    label = path.parent.name if path.name == "ngonsacviet.json" else path.stem
    return label, path


def _resolve_json(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "ngonsacviet.json"
        if candidate.is_file():
            return candidate
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp đánh giá: {path}")
    return path


def load_candidate(label: str, path: Path) -> dict:
    """Load metrics from ngonsacviet.json or eval_report*.json."""
    resolved = _resolve_json(path)
    data = json.loads(resolved.read_text(encoding="utf-8"))

    directions: dict[str, dict] = {}
    meta = data.get("metadata") if "results" in data else data
    if not isinstance(meta, dict):
        meta = {}

    if isinstance(data.get("evaluation"), dict):
        for direction, row in data["evaluation"].items():
            if not isinstance(row, dict):
                continue
            directions[direction] = {
                "chrf": float(row.get("chrf") or 0.0),
                "copy_chrf": float(row.get("copy_chrf") or 0.0),
                "long_chrf": float(row["long_chrf"]) if row.get("long_chrf") is not None else None,
                "length_ratio": float(row["length_ratio"]) if row.get("length_ratio") is not None else None,
                "repetition_rate": float(row.get("repetition_rate") or 0.0),
                "sentences": int(row.get("sentences") or 0),
                "bleu": float(row["bleu"]) if row.get("bleu") is not None else None,
            }
    elif isinstance(data.get("results"), list):
        for row in data["results"]:
            if not isinstance(row, dict) or not row.get("direction"):
                continue
            model_scores = row.get("model") or {}
            copy_scores = row.get("copy_baseline") or {}
            by_len = row.get("by_length") or {}
            long_bucket = by_len.get("16+w") or {}
            directions[row["direction"]] = {
                "chrf": float(model_scores.get("chrf") or 0.0),
                "copy_chrf": float(copy_scores.get("chrf") or 0.0),
                "long_chrf": float(long_bucket["chrf"]) if long_bucket.get("chrf") is not None else None,
                "length_ratio": float(row["length_ratio"]) if row.get("length_ratio") is not None else None,
                "repetition_rate": float(row.get("repetition_rate") or 0.0),
                "sentences": int(row.get("sentences") or 0),
                "bleu": float(model_scores["bleu"]) if model_scores.get("bleu") is not None else None,
            }

    passed = {d: m for d, m in directions.items() if m["chrf"] > m["copy_chrf"]}
    ethnic_langs = sorted(
        {
            part
            for d in passed
            for part in d.split("->", 1)
            if part and part != "vi"
        }
    )

    n = len(directions)
    if n > 0:
        mean_chrf = round(sum(m["chrf"] for m in directions.values()) / n, 2)
        long_vals = [m["long_chrf"] for m in directions.values() if m["long_chrf"] is not None]
        mean_long_chrf = round(sum(long_vals) / len(long_vals), 2) if long_vals else 0.0
        mean_margin = round(sum(m["chrf"] - m["copy_chrf"] for m in directions.values()) / n, 2)
        mean_rep = round(sum(m["repetition_rate"] for m in directions.values()) / n, 4)
        composite = round(
            mean_chrf + 0.35 * mean_long_chrf + 2.0 * len(ethnic_langs) - 20.0 * mean_rep,
            2,
        )
    else:
        mean_chrf = 0.0
        mean_long_chrf = 0.0
        mean_margin = 0.0
        mean_rep = 0.0
        composite = None

    return {
        "label": label,
        "path": str(resolved),
        "epochs": meta.get("epochs"),
        "step": meta.get("step"),
        "dev_loss": meta.get("dev_loss"),
        "directions": directions,
        "passed_directions": len(passed),
        "total_directions": n,
        "languages": ethnic_langs,
        "mean_chrf": mean_chrf,
        "mean_long_chrf": mean_long_chrf,
        "mean_margin": mean_margin,
        "mean_repetition": mean_rep,
        "composite_score": composite,
    }


def compare_candidates(specs: list[tuple[str, Path]]) -> dict:
    """Load, rank, and pick per-direction winners across evaluated candidates."""
    candidates = [load_candidate(label, path) for label, path in specs]
    ranked = sorted(
        (c for c in candidates if c["composite_score"] is not None),
        key=lambda c: (-c["composite_score"], -c["mean_chrf"], -c["passed_directions"], c["label"]),
    )
    unranked = [c for c in candidates if c["composite_score"] is None]

    all_dirs = list(DIRECTIONS)
    for c in candidates:
        for d in sorted(c["directions"]):
            if d not in all_dirs:
                all_dirs.append(d)

    per_direction: dict[str, dict] = {}
    for d in all_dirs:
        contenders = [c for c in candidates if d in c["directions"]]
        if not contenders:
            continue
        best = max(
            contenders,
            key=lambda c: (
                c["directions"][d]["chrf"],
                c["directions"][d]["chrf"] - c["directions"][d]["copy_chrf"],
                -c["directions"][d]["repetition_rate"],
            ),
        )
        m = best["directions"][d]
        per_direction[d] = {
            "winner": best["label"],
            "chrf": m["chrf"],
            "copy_chrf": m["copy_chrf"],
            "long_chrf": m["long_chrf"],
            "passed": m["chrf"] > m["copy_chrf"],
        }

    return {
        "formula": SCORE_FORMULA,
        "ranked": ranked,
        "unranked": unranked,
        "per_direction": per_direction,
        "winner": ranked[0]["label"] if ranked else None,
    }


def format_markdown(summary: dict) -> str:
    """Format leaderboard and per-direction winner tables in GitHub Markdown."""
    lines = [
        f"Score formula: `{summary['formula']}`",
        "",
        "### Overall Leaderboard",
        "| Rank | Candidate | Pass dirs | Langs | Mean chrF++ | Mean 16+w | Mean margin | Mean Rep3 | Composite |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for idx, c in enumerate(summary["ranked"], start=1):
        langs = ", ".join(c["languages"]) if c["languages"] else "-"
        lines.append(
            f"| {idx} | {c['label']} | {c['passed_directions']}/{c['total_directions']} | {langs} "
            f"| {c['mean_chrf']:.2f} | {c['mean_long_chrf']:.2f} | {c['mean_margin']:+.2f} "
            f"| {c['mean_repetition']:.4f} | {c['composite_score']:.2f} |"
        )
    for c in summary["unranked"]:
        lines.append(
            f"| - | {c['label']} | 0/0 | - | - | - | - | - | unranked |"
        )

    lines.extend([
        "",
        "### Per-Direction Winners",
        "| Direction | Winner | chrF++ | 16+w | Copy baseline | Passed |",
        "| --- | --- | --- | --- | --- | --- |",
    ])
    for direction, info in summary["per_direction"].items():
        long_val = f"{info['long_chrf']:.2f}" if info["long_chrf"] is not None else "-"
        passed_str = "yes" if info["passed"] else "no"
        lines.append(
            f"| {direction} | {info['winner']} | {info['chrf']:.2f} | {long_val} "
            f"| {info['copy_chrf']:.2f} | {passed_str} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare evaluated NLLB models and pick optimal winner.")
    parser.add_argument("candidates", nargs="+", help="Paths or label=path to ngonsacviet.json, directory, or eval_report.json")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path to save structured comparison JSON")
    args = parser.parse_args()

    specs = [_parse_arg(spec) for spec in args.candidates]
    summary = compare_candidates(specs)
    print(format_markdown(summary))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
