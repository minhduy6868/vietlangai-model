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

from app.nllb import BASE_TAGS, INIT_FROM, METADATA, NEW_TAGS, tag_for  # noqa: E402
from app.paths import default_nllb_base, reviewed_pairs_path  # noqa: E402
from app.textnorm import clean, to_model  # noqa: E402
from validate_pairs import validate_pairs  # noqa: E402

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


def both_directions(rows: list[tuple[str, str, str, str]]) -> list[tuple[str, str, str, str]]:
    seen = set()
    out = []
    for src_lang, tgt_lang, src, tgt in rows:
        for item in ((src_lang, tgt_lang, src, tgt), (tgt_lang, src_lang, tgt, src)):
            if item not in seen:
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
    import torch

    new = [tag for tag in NEW_TAGS.values() if tag not in tokenizer.get_vocab()]
    if new:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(tokenizer.additional_special_tokens) + new},
            replace_additional_special_tokens=False,
        )
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    weight = model.get_input_embeddings().weight
    ids = []
    with torch.no_grad():
        for tag, sources in INIT_FROM.items():
            tag_id = tokenizer.convert_tokens_to_ids(tag)
            source_ids = [tokenizer.convert_tokens_to_ids(s) for s in sources]
            weight[tag_id] = weight[source_ids].float().mean(dim=0).to(weight.dtype)
            ids.append(tag_id)
    return ids


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="*", type=Path, default=[ROOT / "data" / "public" / "pairs.csv"])
    parser.add_argument("--no-reviewed", action="store_true", help="Bỏ qua data/reviewed_pairs.csv")
    parser.add_argument("--base", default=default_nllb_base())
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "nllb-ethnic")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch", type=int, default=16, help="Batch hiệu dụng sau khi tích luỹ gradient")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--max-per-lang", type=int, default=None)
    parser.add_argument("--full", action="store_true", help="Fine-tune toàn bộ, cần GPU từ 16 GB")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--allow-unknown-license", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    paths = list(args.train)
    reviewed = reviewed_pairs_path()
    if not args.no_reviewed and reviewed.is_file():
        paths.append(reviewed)
    for path in paths:
        print(path.name, validate_pairs(path))

    unknown_ok = args.allow_unknown_license
    dev_pairs = read_pairs(paths, "dev", unknown_ok)
    held_out = held_out_sentences(dev_pairs + read_pairs(paths, "test", unknown_ok))
    train_pairs = [row for row in read_pairs(paths, "train", unknown_ok) if not leaks(row, held_out)]
    train_rows = cap_per_language(both_directions(train_pairs), args.max_per_lang, args.seed)
    dev_rows = both_directions(dev_pairs)
    languages = sorted({lang for row in train_rows for lang in row[:2]} & NEW_TAGS.keys())
    counts: dict[str, int] = {}
    for src_lang, tgt_lang, *_ in train_rows:
        key = f"{src_lang}->{tgt_lang}"
        counts[key] = counts.get(key, 0) + 1
    print({"train": len(train_rows), "dev": len(dev_rows), "languages": languages, "directions": counts})
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
    if cuda:
        model.to("cuda")
    check_embedding_scale(model, tokenizer)

    train_features = encode(tokenizer, train_rows, args.max_length)
    dev_features = encode(tokenizer, dev_rows, args.max_length)
    random.Random(args.seed).shuffle(train_features)

    per_device = max(1, args.batch // args.grad_accum)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.lr if not args.full else min(args.lr, 1e-4),
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=per_device,
        per_device_eval_batch_size=per_device,
        gradient_accumulation_steps=args.grad_accum,
        weight_decay=0.01,
        label_smoothing_factor=args.label_smoothing,
        prediction_loss_only=True,
        bf16=bf16,
        fp16=cuda and not bf16,
        group_by_length=True,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=100,
        report_to=[],
        dataloader_num_workers=0,
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
    started = time.time()
    trainer.train()
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
        "dev_loss": metrics.get("eval_loss"),
        "minutes": round((time.time() - started) / 60, 1),
    }
    (args.output / METADATA).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(metadata)


if __name__ == "__main__":
    main()
