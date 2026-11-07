"""Fine-tune Whisper (PhoWhisper by default) on speech in one ethnic language.

    python training/prepare_base.py --name vinai/PhoWhisper-small --kind speech
    python training/import_speech.py --allow-unknown-license
    python training/finetune_whisper.py --task speech_translation --eval-only   # điểm zero-shot
    python training/finetune_whisper.py --task speech_translation --epochs 10

Two tasks, chosen by what the manifest text really is (`text_lang`):
- asr: text in the spoken language (H'Mông transcripts in the domestic script).
- speech_translation: H'Mông audio -> Vietnamese text, which is what the
  public H'Mông sets contain.

Decoding is forced to Whisper's Vietnamese token (`--whisper-language vi`):
Whisper has no H'Mông token, and both the domestic script and the
translation target use Vietnamese letters. The checkpoint gets
ngonsacviet.json with the test WER/CER. The server only loads checkpoints
that have a test WER and were trained on verified rows.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from app.audio import SAMPLE_RATE, decode_audio  # noqa: E402
from app.checkpoints import write_metadata  # noqa: E402
from app.paths import models_dir  # noqa: E402
from speech_metrics import error_rates  # noqa: E402

TRUE = {"1", "true", "yes"}
DOMESTIC_HMONG = "vn"
MAX_SECONDS = 30
MAX_CHARS_PER_SECOND = 25


def usable(row: dict) -> bool:
    """Drop clips longer than Whisper's window or whose text cannot fit the audio."""
    try:
        duration = float(row.get("duration") or 0)
    except ValueError:
        return False
    return 0.3 <= duration <= MAX_SECONDS and len(row["text"]) / duration <= MAX_CHARS_PER_SECOND


def read_manifest(path: Path, split: str, language: str, task: str, allow_unverified: bool) -> list[dict]:
    text_lang = "vi" if task == "speech_translation" else language
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["split"] != split or row["language"] != language:
                continue
            if (row.get("text_lang") or row["language"]) != text_lang:
                continue
            if task == "asr" and language == "hmn" and row.get("orthography") != DOMESTIC_HMONG:
                continue
            if not allow_unverified and (row.get("verified") or "").strip().lower() not in TRUE:
                continue
            if not usable(row):
                continue
            rows.append({"path": str(path.parent / row["path"]), "text": row["text"]})
    return rows


def load_samples(path: str):
    return decode_audio(Path(path).read_bytes())


def transcribe_all(model, processor, rows: list[dict], whisper_language: str | None, batch: int) -> list[str]:
    import torch

    outputs = []
    kwargs = {"task": "transcribe", "max_new_tokens": 225}
    if whisper_language:
        kwargs["language"] = whisper_language
    for i in range(0, len(rows), batch):
        audio = [load_samples(row["path"]) for row in rows[i : i + batch]]
        features = processor.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        with torch.inference_mode():
            ids = model.generate(features.to(model.device, model.dtype), **kwargs)
        outputs.extend(processor.batch_decode(ids, skip_special_tokens=True))
    return outputs


def default_base() -> str:
    local = models_dir() / "PhoWhisper-small"
    return str(local) if (local / "model.safetensors").is_file() else "openai/whisper-small"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "speech" / "manifest.csv")
    parser.add_argument("--language", default="hmn", help="Ngôn ngữ được nói trong audio")
    parser.add_argument("--task", choices=("asr", "speech_translation"), default="asr")
    parser.add_argument("--whisper-language", default="vi")
    parser.add_argument("--base", default=default_base())
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--epochs", type=float, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=16, help="Batch hiệu dụng sau khi tích luỹ gradient")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--allow-unverified", action="store_true", help="Chỉ để thử nghiệm; server sẽ không nạp checkpoint này")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output = args.output or ROOT / "models" / f"whisper-{args.language}-{args.task.replace('_', '-')}"

    splits = {
        s: read_manifest(args.manifest, s, args.language, args.task, args.allow_unverified)
        for s in ("train", "dev", "test")
    }
    print({s: len(rows) for s, rows in splits.items()})
    if args.dry_run:
        return
    if not splits["test"]:
        raise SystemExit("Không có đoạn test đã duyệt. Chạy import_speech.py và duyệt transcript trước.")

    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, Seq2SeqTrainer, Seq2SeqTrainingArguments

    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    processor = AutoProcessor.from_pretrained(args.base)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.base)
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    if cuda:
        model.to("cuda")

    if args.eval_only:
        model.eval()
        hypotheses = transcribe_all(model, processor, splits["test"], args.whisper_language, 8)
        print({"model": args.base, "test_clips": len(hypotheses), **error_rates(hypotheses, [r["text"] for r in splits["test"]])})
        return
    if not splits["train"]:
        raise SystemExit("Không có đoạn train đã duyệt.")

    processor.tokenizer.set_prefix_tokens(language=args.whisper_language, task="transcribe")
    start_id = model.config.decoder_start_token_id

    def collate(rows: list[dict]) -> dict:
        audio = [load_samples(row["path"]) for row in rows]
        features = processor.feature_extractor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        labels = processor.tokenizer([row["text"] for row in rows], padding=True, return_tensors="pt")
        ids = labels["input_ids"].masked_fill(labels["attention_mask"].ne(1), -100)
        if (ids[:, 0] == start_id).all():
            ids = ids[:, 1:]
        return {"input_features": features, "labels": ids}

    per_device = max(1, args.batch // args.grad_accum)
    trainer = Seq2SeqTrainer(
        model=model,
        args=Seq2SeqTrainingArguments(
            output_dir=str(output / "checkpoints"),
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            warmup_ratio=0.1,
            per_device_train_batch_size=per_device,
            per_device_eval_batch_size=per_device,
            gradient_accumulation_steps=args.grad_accum,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim="adafactor",
            bf16=bf16,
            fp16=cuda and not bf16,
            eval_strategy="epoch" if splits["dev"] else "no",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=bool(splits["dev"]),
            metric_for_best_model="eval_loss",
            prediction_loss_only=True,
            remove_unused_columns=False,
            logging_steps=25,
            report_to=[],
            dataloader_num_workers=0,
        ),
        train_dataset=splits["train"],
        eval_dataset=splits["dev"] or None,
        data_collator=collate,
    )
    model.config.use_cache = False
    started = time.time()
    trainer.train()
    model.config.use_cache = True
    model.eval()

    hypotheses = transcribe_all(model, processor, splits["test"], args.whisper_language, 8)
    rates = error_rates(hypotheses, [row["text"] for row in splits["test"]])
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    processor.save_pretrained(output)
    metadata = {
        "task": args.task,
        "language": args.language,
        "text_language": "vi" if args.task == "speech_translation" else args.language,
        "whisper_language": args.whisper_language,
        "base_model": Path(args.base).name,
        "train_clips": len(splits["train"]),
        "test_clips": len(splits["test"]),
        "speaker_disjoint": False,
        "test_wer": rates["wer"],
        "test_cer": rates["cer"],
        "trained_on_unverified": args.allow_unverified,
        "minutes": round((time.time() - started) / 60, 1),
    }
    write_metadata(output, metadata)
    print(metadata)


if __name__ == "__main__":
    main()
