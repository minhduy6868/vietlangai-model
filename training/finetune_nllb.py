"""Fine-tune NLLB-200 for ethnic <-> Vietnamese with one LoRA adapter.

Each ethnic language gets its own tag token, started from related NLLB
languages (see app.nllb.INIT_FROM). Every reviewed pair is trained in both
directions. Vietnamese <-> English is not trained: the server turns the
adapter off for that pair.

    python training/import_public.py
    python training/prepare_base.py
    python training/finetune_nllb.py --epochs 3

Inputs are reviewed CSVs (verified=true). `data/reviewed_pairs.csv`, the
reviewer export from the API, is added automatically so approved community
sentences flow into the next run.
"""

import argparse
import csv
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "training"))

from validate_pairs import validate_pairs

from app.nllb import BASE_TAGS, METADATA, NEW_TAGS, init_new_tag_embeddings, tag_for
from app.paths import default_nllb_base, reviewed_pairs_path
from app.textnorm import clean, to_model

TRUE = {"1", "true", "yes"}


def read_pairs(paths: list[Path], split: str, allow_unknown_license: bool = False) -> list[tuple[str, str, str, str]]:
    """Return (src_lang, tgt_lang, src, tgt) for ethnic <-> vi rows in a split.

    Rows without a split column count as train.
    """
    rows = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if (row.get("verified") or "").strip().lower() not in TRUE:
                    continue
                if not allow_unknown_license and (row.get("license") or "").strip().lower() == "unknown":
                    continue
                if (row.get("split") or "train").strip() != split:
                    continue
                src_lang, tgt_lang = row["source_lang"].strip(), row["target_lang"].strip()
                if not ({src_lang, tgt_lang} & NEW_TAGS.keys() and "vi" in {src_lang, tgt_lang}):
                    continue
                rows.append((src_lang, tgt_lang, row["source"].strip(), row["target"].strip()))
    return rows


def _key(lang: str, text: str) -> tuple[str, str]:
    return lang, clean(text).casefold()


def held_out_sentences(rows) -> set[tuple[str, str]]:
    """Both sides of every dev/test pair, since training runs both directions."""
    return {key for src_lang, tgt_lang, src, tgt in rows for key in (_key(src_lang, src), _key(tgt_lang, tgt))}


def leaks(row, held_out: set[tuple[str, str]]) -> bool:
    src_lang, tgt_lang, src, tgt = row
    return _key(src_lang, src) in held_out or _key(tgt_lang, tgt) in held_out


def both_directions(rows: list[tuple[str, str, str, str]], dedup: bool = True) -> list[tuple[str, str, str, str]]:
    seen = set()
    out = []
    for src_lang, tgt_lang, src, tgt in rows:
        for item in ((src_lang, tgt_lang, src, tgt), (tgt_lang, src_lang, tgt, src)):
            if dedup:
                if item in seen:
                    continue
                seen.add(item)
            out.append(item)
    return out


def cap_per_language(rows, limit: int | None, seed: int):
    if not limit:
        return rows
    rng = random.Random(seed)
    grouped: dict[str, list] = {}
    for row in rows:
        lang = row[0] if row[0] != "vi" else row[1]
        grouped.setdefault(lang, []).append(row)
    out = []
    for items in grouped.values():
        rng.shuffle(items)
        out.extend(items[:limit])
    return out


def upsample_long_sentences(
    rows: list[tuple[str, str, str, str]], min_words: int = 8, factor: int = 2
) -> list[tuple[str, str, str, str]]:
    """Repeat longer sentence pairs so short dictionary entries do not dominate training."""
    if factor <= 1:
        return list(rows)
    out: list[tuple[str, str, str, str]] = []
    for row in rows:
        _, _, src, tgt = row
        if max(len(src.split()), len(tgt.split())) >= min_words:
            out.extend([row] * factor)
        else:
            out.append(row)
    return out


def _ensure_terminal(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return stripped
    return stripped if stripped[-1] in ".!?…" else f"{stripped}."


def synthesize_long_pairs(
    rows: list[tuple[str, str, str, str]],
    ratio: float = 0.35,
    min_words: int = 4,
    max_sents: int = 4,
    max_chars: int = 380,
    seed: int = 13,
) -> list[tuple[str, str, str, str]]:
    """Concatenate verified sentences from the same language pair into longer training examples."""
    if ratio <= 0 or max_sents < 2:
        return []
    rng = random.Random(seed)
    grouped: dict[tuple[str, str], list[tuple[str, str, str, str]]] = {}
    for row in rows:
        grouped.setdefault((row[0], row[1]), []).append(row)

    out: list[tuple[str, str, str, str]] = []
    for (src_lang, tgt_lang), items in sorted(grouped.items()):
        candidates = [
            (src, tgt)
            for _, _, src, tgt in items
            if len(src.split()) >= min_words and len(tgt.split()) >= min_words
        ]
        if len(candidates) < 2:
            continue
        target_count = int(len(items) * ratio)
        upper = min(max_sents, len(candidates))
        for _ in range(target_count):
            k = rng.randint(2, upper)
            picked = rng.sample(candidates, k)
            src_parts: list[str] = []
            tgt_parts: list[str] = []
            for src_seg, tgt_seg in picked:
                s_seg = _ensure_terminal(src_seg)
                t_seg = _ensure_terminal(tgt_seg)
                next_src = " ".join([*src_parts, s_seg])
                next_tgt = " ".join([*tgt_parts, t_seg])
                if src_parts and (len(next_src) > max_chars or len(next_tgt) > max_chars):
                    break
                src_parts.append(s_seg)
                tgt_parts.append(t_seg)
            if len(src_parts) >= 2:
                out.append((src_lang, tgt_lang, " ".join(src_parts), " ".join(tgt_parts)))
    return out


def balance_by_temperature(
    rows: list[tuple[str, str, str, str]], temperature: float = 2.0, seed: int = 13
) -> list[tuple[str, str, str, str]]:
    """Rebalance multi-language training rows using temperature sampling (N^(1/T))."""
    if not rows or temperature <= 1.0:
        return list(rows)
    grouped: dict[str, list[tuple[str, str, str, str]]] = {}
    for row in rows:
        lang = row[0] if row[0] != "vi" else row[1]
        grouped.setdefault(lang, []).append(row)
    if len(grouped) <= 1:
        return list(rows)

    rng = random.Random(seed)
    total = len(rows)
    weights = {lang: len(items) ** (1.0 / temperature) for lang, items in grouped.items()}
    weight_sum = sum(weights.values())
    out: list[tuple[str, str, str, str]] = []
    for lang, items in sorted(grouped.items()):
        target = max(1, round(total * weights[lang] / weight_sum))
        if target <= len(items):
            out.extend(rng.sample(items, target))
        else:
            out.extend(items)
            out.extend(rng.choices(items, k=target - len(items)))
    return out


def encode(tokenizer, rows, max_length: int) -> list[dict]:
    by_direction: dict[tuple[str, str], list] = {}
    for src_lang, tgt_lang, src, tgt in rows:
        by_direction.setdefault((src_lang, tgt_lang), []).append((src, tgt))
    features = []
    for (src_lang, tgt_lang), items in by_direction.items():
        tokenizer.src_lang = tag_for(src_lang)
        tokenizer.tgt_lang = tag_for(tgt_lang)
        batch = tokenizer(
            [to_model(src) for src, _ in items],
            text_target=[to_model(tgt) for _, tgt in items],
            truncation=True,
            max_length=max_length,
        )
        for i in range(len(items)):
            features.append(
                {
                    "input_ids": batch["input_ids"][i],
                    "attention_mask": batch["attention_mask"][i],
                    "labels": batch["labels"][i],
                }
            )
    return features


def add_language_tags(tokenizer, model) -> list[int]:
    new = [tag for tag in NEW_TAGS.values() if tag not in tokenizer.get_vocab()]
    if new:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(tokenizer.additional_special_tokens) + new},
            replace_additional_special_tokens=False,
        )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    return init_new_tag_embeddings(model, tokenizer)


def check_embedding_scale(model, tokenizer) -> None:
    """Fail early if the LoRA wrapper drops NLLB's sqrt(d_model) embedding scale."""
    import torch

    token = tokenizer.convert_tokens_to_ids(BASE_TAGS["vi"])
    embed = model.get_encoder().embed_tokens
    base = embed.get_base_layer() if hasattr(embed, "get_base_layer") else embed
    with torch.no_grad():
        wrapped = embed(torch.tensor([[token]], device=base.weight.device)).float().norm().item()
        raw = base.weight[token].float().norm().item()
    scale = model.config.d_model**0.5 if model.config.scale_embedding else 1.0
    if not math.isclose(wrapped, raw * scale, rel_tol=0.05):
        raise SystemExit(f"Embedding scale lệch: {wrapped:.2f} so với {raw * scale:.2f}")


def latest_checkpoint(checkpoints_dir: Path) -> Path | None:
    """Return the checkpoint-<step> subdirectory with the highest numeric step."""
    if not checkpoints_dir.is_dir():
        return None
    best_step = -1
    best_path: Path | None = None
    for entry in checkpoints_dir.iterdir():
        if not entry.is_dir() or not entry.name.startswith("checkpoint-"):
            continue
        suffix = entry.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            step = int(suffix)
            if step > best_step:
                best_step = step
                best_path = entry
    return best_path


def has_data_rows(path: Path) -> bool:
    """Return True if a CSV file has at least one non-empty data row after its header."""
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        return any(any(cell.strip() for cell in row) for row in reader)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="*", type=Path, default=[ROOT / "data" / "public" / "pairs.csv"])
    parser.add_argument("--no-reviewed", action="store_true", help="Bỏ qua data/reviewed_pairs.csv")
    parser.add_argument("--base", default=default_nllb_base())
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "nllb-ethnic")
    parser.add_argument("--epochs", type=float, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=32, help="Batch hiệu dụng sau khi tích luỹ gradient")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--concat-ratio", type=float, default=0.35, help="Tỷ lệ mẫu ghép nhiều câu")
    parser.add_argument("--concat-min-words", type=int, default=4)
    parser.add_argument("--concat-max-sents", type=int, default=4)
    parser.add_argument("--long-min-words", type=int, default=8)
    parser.add_argument("--long-upsample", type=int, default=2)
    parser.add_argument("--temp", type=float, default=2.0, help="Nhiệt độ cân bằng giữa các ngôn ngữ")
    parser.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-per-lang", type=int, default=None)
    parser.add_argument("--full", action="store_true", help="Fine-tune toàn bộ, cần GPU từ 16 GB")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--allow-unknown-license", action="store_true")
    parser.add_argument("--save-steps", type=int, default=0, help="Lưu & eval mỗi N bước (0 = theo từng epoch)")
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--workers", type=int, default=0, help="Số worker DataLoader (0 tránh kẹt fork khi dữ liệu đã nạp RAM)")
    parser.add_argument("--resume", action="store_true", help="Tiếp tục từ checkpoint gần nhất trong output/checkpoints nếu có")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = list(args.train)
    reviewed = reviewed_pairs_path()
    if not args.no_reviewed and reviewed.is_file() and has_data_rows(reviewed):
        paths.append(reviewed)
    for path in paths:
        print(path.name, validate_pairs(path), flush=True)

    unknown_ok = args.allow_unknown_license
    dev_pairs = read_pairs(paths, "dev", unknown_ok)
    held_out = held_out_sentences(dev_pairs + read_pairs(paths, "test", unknown_ok))
    train_pairs = list(dict.fromkeys(row for row in read_pairs(paths, "train", unknown_ok) if not leaks(row, held_out)))
    base_train = upsample_long_sentences(train_pairs, args.long_min_words, args.long_upsample)
    synth_train = synthesize_long_pairs(
        train_pairs,
        ratio=args.concat_ratio,
        min_words=args.concat_min_words,
        max_sents=args.concat_max_sents,
        seed=args.seed,
    )
    combined = balance_by_temperature(base_train + synth_train, args.temp, args.seed)
    train_rows = cap_per_language(both_directions(combined, dedup=False), args.max_per_lang, args.seed)
    dev_rows = both_directions(dev_pairs)
    languages = sorted({lang for row in train_rows for lang in row[:2]} & NEW_TAGS.keys())
    counts: dict[str, int] = {}
    for src_lang, tgt_lang, *_ in train_rows:
        key = f"{src_lang}->{tgt_lang}"
        counts[key] = counts.get(key, 0) + 1
    print(
        {
            "raw_train": len(train_pairs),
            "long_upsampled": len(base_train),
            "synthesized_long": len(synth_train),
            "final_train": len(train_rows),
            "dev": len(dev_rows),
            "languages": languages,
            "directions": counts,
        },
        flush=True,
    )
    if args.dry_run:
        return
    if not languages:
        raise SystemExit("Không có câu dân tộc ↔ Việt đã duyệt để huấn luyện.")

    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )
    from transformers.models.m2m_100.modeling_m2m_100 import shift_tokens_right

    torch.manual_seed(args.seed)
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base, dtype=torch.bfloat16 if bf16 and not args.full else torch.float32)
    tag_ids = add_language_tags(tokenizer, model)

    if not args.full:
        from peft import LoraConfig, get_peft_model

        config = LoraConfig(
            r=args.rank,
            lora_alpha=2 * args.rank,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
            task_type="SEQ_2_SEQ_LM",
            trainable_token_indices={"embed_tokens": tag_ids},
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    if cuda and args.grad_checkpoint:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if not args.full and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    if cuda:
        model.to("cuda")
    check_embedding_scale(model, tokenizer)

    train_features = encode(tokenizer, train_rows, args.max_length)
    dev_features = encode(tokenizer, dev_rows, args.max_length)
    random.Random(args.seed).shuffle(train_features)

    per_device = max(1, args.batch // args.grad_accum)
    by_steps = args.save_steps > 0
    eval_strategy = ("steps" if by_steps else "epoch") if dev_features else "no"
    save_strategy = "steps" if by_steps else ("epoch" if dev_features else "no")
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr if not args.full else min(args.lr, 5e-5),
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=per_device,
        per_device_eval_batch_size=max(per_device, 8),
        gradient_accumulation_steps=args.grad_accum,
        max_grad_norm=1.0,
        weight_decay=0.01,
        label_smoothing_factor=args.label_smoothing,
        prediction_loss_only=True,
        bf16=bf16,
        fp16=cuda and not bf16,
        group_by_length=True,
        eval_strategy=eval_strategy,
        eval_steps=args.save_steps if by_steps and dev_features else None,
        save_strategy=save_strategy,
        save_steps=args.save_steps if by_steps else 500,
        save_total_limit=2 if by_steps else 1,
        load_best_model_at_end=bool(dev_features),
        metric_for_best_model="eval_loss",
        logging_steps=args.logging_steps,
        disable_tqdm=not sys.stderr.isatty(),
        report_to=[],
        dataloader_num_workers=args.workers,
        seed=args.seed,
    )
    pad_collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8)
    start_id = model.config.decoder_start_token_id

    # M2M100 has no prepare_decoder_input_ids_from_labels, and the label smoother
    # pops labels before the forward pass, so decoder inputs are built here.
    def collate(features: list[dict]) -> dict:
        batch = pad_collator(features)
        batch["decoder_input_ids"] = shift_tokens_right(batch["labels"], tokenizer.pad_token_id, start_id)
        return batch

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_features,
        eval_dataset=dev_features or None,
        data_collator=collate,
    )
    resume_ckpt = latest_checkpoint(args.output / "checkpoints") if args.resume else None
    if resume_ckpt:
        print(f"Tiếp tục huấn luyện từ checkpoint: {resume_ckpt}", flush=True)
    started = time.time()
    trainer.train(resume_from_checkpoint=str(resume_ckpt) if resume_ckpt else None)
    metrics = trainer.evaluate() if dev_features else {}

    args.output.mkdir(parents=True, exist_ok=True)
    if args.full:
        model.save_pretrained(args.output)
    else:
        model.save_pretrained(args.output, save_embedding_layers=False)
    tokenizer.save_pretrained(args.output)
    metadata = {
        "base_model": Path(args.base).name if Path(args.base).exists() else args.base,
        "method": "full" if args.full else f"lora-r{args.rank}",
        "languages": languages,
        "directions": counts,
        "train_sources": [p.name for p in paths],
        "epochs": args.epochs,
        "max_length": args.max_length,
        "label_smoothing": args.label_smoothing,
        "concat_ratio": args.concat_ratio,
        "long_upsample": args.long_upsample,
        "dev_loss": metrics.get("eval_loss"),
        "minutes": round((time.time() - started) / 60, 1),
    }
    (args.output / METADATA).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metadata, flush=True)


if __name__ == "__main__":
    main()
