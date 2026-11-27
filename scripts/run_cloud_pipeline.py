"""Unbuffered cloud training, evaluation, smoke check, and bundle packaging runner."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.nllb import METADATA, NllbEngine

SMOKE_SAMPLES = [
    (
        "vi",
        "tyz",
        "Sáng nay trời mát mẻ, gia đình tôi cùng nhau lên rẫy hái măng. Buổi chiều chúng tôi quay về bản nấu cơm và trò chuyện bên bếp lửa.",
    ),
    (
        "vi",
        "rad",
        "Hôm nay buôn làng tổ chức lễ hội đầu mùa rất vui vẻ. Mọi người mặc trang phục truyền thống và cùng nhau đánh chiêng đón khách quý.",
    ),
]


def run_streamed(cmd: list[str]) -> None:
    """Execute a subprocess and stream merged stdout/stderr line-by-line immediately."""
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    with subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
        code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def run_smoke(base_model: str, output_dir: Path) -> None:
    """Run bilingual paragraph + back-translation smoke checks with the trained adapter."""
    engine = NllbEngine(base_model, str(output_dir), beams=4, gated=False)
    for src, tgt, text in SMOKE_SAMPLES:
        if (src, tgt) in engine.pairs():
            out = engine.translate(text, src, tgt)
            back = engine.translate(out, tgt, src)
            print(f"[{src}->{tgt}] {out}", flush=True)
            print(f"[{tgt}->{src}] {back}\n", flush=True)


def package_bundle(output_dir: Path, zip_out: Path, eval_out: Path, drive_dir: Path | None) -> Path:
    """Clean temporary checkpoints once metadata exists, build ready zip, and mirror to Drive."""
    ckpt_dir = output_dir / "checkpoints"
    if (output_dir / METADATA).is_file() and ckpt_dir.exists():
        shutil.rmtree(ckpt_dir)

    zip_out.parent.mkdir(parents=True, exist_ok=True)
    archive_path = Path(
        shutil.make_archive(
            str(zip_out.with_suffix("")),
            "zip",
            root_dir=str(output_dir.parent),
            base_dir=output_dir.name,
        )
    )
    print(f"Đã đóng gói: {archive_path} ({archive_path.stat().st_size / 1e6:.2f} MB)", flush=True)

    if drive_dir and (drive_dir.exists() or drive_dir.parent.exists()):
        drive_dir.mkdir(parents=True, exist_ok=True)
        drive_zip = drive_dir / archive_path.name
        shutil.copy2(archive_path, drive_zip)
        print(f"Drive zip: {drive_zip} ({drive_zip.stat().st_size / 1e6:.2f} MB)", flush=True)
        if eval_out.is_file():
            drive_eval = drive_dir / eval_out.name
            shutil.copy2(eval_out, drive_eval)
            print(f"Drive eval: {drive_eval} ({drive_eval.stat().st_size / 1e3:.1f} KB)", flush=True)
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unbuffered cloud training + eval + packaging pipeline.")
    parser.add_argument("--base", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--drive-dir", type=Path, default=None)
    parser.add_argument("--zip-out", type=Path, default=Path("/content/nllb-ethnic-ready.zip"))
    parser.add_argument("--eval-out", type=Path, default=Path("/content/eval_report.json"))
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--concat-ratio", type=float, default=0.25)
    parser.add_argument("--long-upsample", type=int, default=1)
    parser.add_argument("--temp", type=float, default=2.0)
    parser.add_argument("--save-steps", type=int, default=150)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--eval-beams", type=int, default=4)
    parser.add_argument("--eval-batch", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--allow-unknown-license", action="store_true")
    args = parser.parse_args()

    train_cmd = [
        sys.executable, "-u", str(ROOT / "training" / "finetune_nllb.py"),
        "--base", args.base,
        "--output", str(args.output),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--batch", str(args.batch),
        "--grad-accum", str(args.grad_accum),
        "--max-length", str(args.max_length),
        "--label-smoothing", str(args.label_smoothing),
        "--rank", str(args.rank),
        "--concat-ratio", str(args.concat_ratio),
        "--long-upsample", str(args.long_upsample),
        "--temp", str(args.temp),
        "--save-steps", str(args.save_steps),
        "--logging-steps", str(args.logging_steps),
        "--workers", str(args.workers),
    ]
    if not args.grad_checkpoint:
        train_cmd.append("--no-grad-checkpoint")
    if args.resume:
        train_cmd.append("--resume")
    if args.full:
        train_cmd.append("--full")
    if args.allow_unknown_license:
        train_cmd.append("--allow-unknown-license")

    print("=== [1/4] TRAIN ===", flush=True)
    run_streamed(train_cmd)

    eval_cmd = [
        sys.executable, "-u", str(ROOT / "training" / "evaluate.py"),
        "--base", args.base,
        "--adapter", str(args.output),
        "--split", "test",
        "--batch", str(args.eval_batch),
        "--beams", str(args.eval_beams),
        "--output", str(args.eval_out),
    ]
    if args.allow_unknown_license:
        eval_cmd.append("--allow-unknown-license")

    print("=== [2/4] EVALUATE ===", flush=True)
    run_streamed(eval_cmd)

    print("=== [3/4] SMOKE TEST ===", flush=True)
    run_smoke(args.base, args.output)

    print("=== [4/4] PACKAGE ===", flush=True)
    package_bundle(args.output, args.zip_out, args.eval_out, args.drive_dir)
    print("[PIPELINE_COMPLETE]", flush=True)


if __name__ == "__main__":
    main()
