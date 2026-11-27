"""Export, evaluate, smoke-test, and zip the best/latest checkpoint from an interrupted run."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.nllb import METADATA, NEW_TAGS
from scripts.run_cloud_pipeline import package_bundle, run_smoke, run_streamed
from training.finetune_nllb import latest_checkpoint

TOKENIZER_FILES = (
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "added_tokens.json",
)
ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "adapter_model.bin",
)


def pick_best_checkpoint(checkpoints_dir: Path) -> tuple[Path, dict]:
    """Select best_model_checkpoint recorded in trainer_state.json, or the latest checkpoint."""
    latest = latest_checkpoint(checkpoints_dir)
    if latest is None:
        raise FileNotFoundError(f"Không tìm thấy checkpoint-* nào trong {checkpoints_dir}")

    info: dict = {"latest": latest.name}
    state_file = latest / "trainer_state.json"
    chosen = latest
    if state_file.is_file():
        state = json.loads(state_file.read_text(encoding="utf-8"))
        info["best_metric"] = state.get("best_metric")
        info["global_step"] = state.get("global_step")
        info["epoch"] = state.get("epoch")
        best_raw = state.get("best_model_checkpoint")
        if best_raw:
            best_path = Path(best_raw)
            if not best_path.is_dir():
                candidate = checkpoints_dir / best_path.name
                if candidate.is_dir():
                    best_path = candidate
            if best_path.is_dir():
                chosen = best_path
                best_step = chosen.name.removeprefix("checkpoint-")
                if best_step.isdigit():
                    info["global_step"] = int(best_step)
    info["chosen"] = chosen.name
    return chosen, info


def _ensure_tokenizer(output_dir: Path, base_model: str) -> None:
    if (output_dir / "tokenizer_config.json").is_file():
        return
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    new_tags = [tag for tag in NEW_TAGS.values() if tag not in tokenizer.get_vocab()]
    if new_tags:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(tokenizer.additional_special_tokens) + new_tags},
            replace_additional_special_tokens=False,
        )
    tokenizer.save_pretrained(output_dir)


def export_adapter(output_dir: Path, base_model: str, languages: list[str], method: str = "lora-r64") -> Path:
    """Copy the best/latest checkpoint adapter + tokenizer and write ngonsacviet.json."""
    chosen, info = pick_best_checkpoint(output_dir / "checkpoints")
    output_dir.mkdir(parents=True, exist_ok=True)

    copied = False
    for fname in (*ADAPTER_FILES, *TOKENIZER_FILES):
        src = chosen / fname
        if src.is_file():
            shutil.copy2(src, output_dir / fname)
            if fname in ("adapter_model.safetensors", "adapter_model.bin"):
                copied = True
    if not copied:
        raise FileNotFoundError(f"Checkpoint {chosen} thiếu trọng số LoRA (adapter_model.safetensors/.bin)")

    _ensure_tokenizer(output_dir, base_model)

    metadata = {
        "base_model": Path(base_model).name if Path(base_model).exists() else base_model,
        "method": method,
        "languages": sorted(languages),
        "epochs": info.get("epoch"),
        "step": info.get("global_step"),
        "dev_loss": info.get("best_metric"),
        "exported_from": chosen.name,
    }
    (output_dir / METADATA).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Exported adapter metadata:", metadata, flush=True)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and evaluate best/latest partial checkpoint.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--languages", nargs="+", default=["rad", "tyz"])
    parser.add_argument("--method", default="lora-r64")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--zip-out", type=Path, default=Path("/content/nllb-ethnic-ready.zip"))
    parser.add_argument("--eval-out", type=Path, default=Path("/content/eval_report.json"))
    parser.add_argument("--eval-beams", type=int, default=4)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--drive-dir", type=Path, default=None)
    parser.add_argument("--allow-unknown-license", action="store_true")
    args = parser.parse_args()

    export_adapter(args.output, args.base, args.languages, method=args.method)

    if args.evaluate:
        eval_cmd = [
            sys.executable,
            "-u",
            str(ROOT / "training" / "evaluate.py"),
            "--base",
            args.base,
            "--adapter",
            str(args.output),
            "--split",
            "test",
            "--batch",
            str(args.eval_batch),
            "--beams",
            str(args.eval_beams),
            "--output",
            str(args.eval_out),
        ]
        if args.allow_unknown_license:
            eval_cmd.append("--allow-unknown-license")
        run_streamed(eval_cmd)

    if args.smoke:
        run_smoke(args.base, args.output)

    if args.zip_out:
        package_bundle(args.output, args.zip_out, args.eval_out, args.drive_dir)

    print("[EXPORT_COMPLETE]", flush=True)


if __name__ == "__main__":
    main()
