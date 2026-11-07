"""Import published human-translated corpora into one reviewed CSV.

    python training/import_public.py
    python training/import_public.py --include jarai-vie-bible --allow-unknown-license

Output columns: source_lang, target_lang, source, target, verified, split,
source_dataset, license. Rows always go ethnic -> vi; training adds the
reverse direction. Run this in its own process: on Windows pyarrow and
sentencepiece crash when loaded together.
"""

import argparse
import csv
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.textnorm import clean  # noqa: E402

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{file}"

SOURCES = {
    "tay-vietnamese-nmt": {
        "repo": "HeyDunaX/tay-vietnamese-nmt",
        "lang": "tyz",
        "license": "cc-by-nc-4.0",
        "files": {
            "train": "data/train-00000-of-00001.parquet",
            "test": "data/val-00000-of-00001.parquet",
        },
        "fields": ("translation", "tay", "viet"),
    },
    "rhade-vietnamese-mt": {
        "repo": "NIRVLab/rhade-vietnamese-mt",
        "lang": "rad",
        "license": "cc-by-nc-4.0",
        "files": {
            "train": "data/train-00000-of-00001.parquet",
            "dev": "data/validation-00000-of-00001.parquet",
            "test": "data/test-00000-of-00001.parquet",
        },
        "fields": ("translation", "ede", "vi"),
    },
    "jarai-vie-bible": {
        "repo": "21uyennt/jarai_vie_bible",
        "lang": "jra",
        "license": "unknown",
        "files": {"all": "bible_vietnamese_jarai/vie151_jrai1973/vie_jrai.parquet"},
        "fields": (None, "Jarai", "Vietnamese"),
    },
}
DEFAULT = ("tay-vietnamese-nmt", "rhade-vietnamese-mt")
COLUMNS = ("source_lang", "target_lang", "source", "target", "verified", "split", "source_dataset", "license")


def bucket(text: str) -> float:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF


def fetch(repo: str, file: str, cache: Path) -> Path:
    target = cache / repo.replace("/", "__") / file
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(HF.format(repo=repo, file=file), target)
    return target


def read_rows(path: Path, fields: tuple) -> list[tuple[str, str]]:
    import pyarrow.parquet as pq

    struct, src_key, tgt_key = fields
    rows = []
    for record in pq.read_table(path).to_pylist():
        item = record[struct] if struct else record
        rows.append((item.get(src_key) or "", item.get(tgt_key) or ""))
    return rows


def keep(src: str, tgt: str) -> bool:
    if not src or not tgt:
        return False
    if len(src) > 400 or len(tgt) > 400:
        return False
    a, b = len(src.split()), len(tgt.split())
    if max(a, b) >= 4 and max(a, b) > 3.5 * min(a, b):
        return False
    return True


def assign_split(declared: str, src: str, dev_share: float, has_dev: bool) -> str:
    if declared in {"dev", "test"} or (declared == "train" and has_dev):
        return declared
    roll = bucket(src)
    if declared == "all":
        if roll < dev_share:
            return "dev"
        if roll < 2 * dev_share:
            return "test"
        return "train"
    return "dev" if roll < dev_share else "train"


def import_source(name: str, cache: Path, dev_share: float) -> tuple[list[dict], dict]:
    spec = SOURCES[name]
    rows: list[dict] = []
    stats = {"read": 0, "filtered": 0, "duplicate": 0, "leak": 0}
    seen: set[tuple[str, str]] = set()
    for declared, file in spec["files"].items():
        for raw_src, raw_tgt in read_rows(fetch(spec["repo"], file, cache), spec["fields"]):
            stats["read"] += 1
            src, tgt = clean(raw_src), clean(raw_tgt)
            if not keep(src, tgt):
                stats["filtered"] += 1
                continue
            key = (src.casefold(), tgt.casefold())
            if key in seen:
                stats["duplicate"] += 1
                continue
            seen.add(key)
            rows.append(
                {
                    "source_lang": spec["lang"],
                    "target_lang": "vi",
                    "source": src,
                    "target": tgt,
                    "verified": "true",
                    "split": assign_split(declared, src, dev_share, "dev" in spec["files"]),
                    "source_dataset": spec["repo"],
                    "license": spec["license"],
                }
            )
    held_out = {row["source"].casefold() for row in rows if row["split"] != "train"}
    clean_rows = []
    for row in rows:
        if row["split"] == "train" and row["source"].casefold() in held_out:
            stats["leak"] += 1
            continue
        clean_rows.append(row)
    return clean_rows, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include", nargs="*", default=list(DEFAULT), choices=sorted(SOURCES))
    parser.add_argument("--allow-unknown-license", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "public" / "pairs.csv")
    parser.add_argument("--cache", type=Path, default=ROOT / "data" / "public" / "raw")
    parser.add_argument("--dev-share", type=float, default=0.03)
    args = parser.parse_args()

    all_rows: list[dict] = []
    for name in args.include:
        if SOURCES[name]["license"] == "unknown" and not args.allow_unknown_license:
            raise SystemExit(f"{name} không ghi license. Thêm --allow-unknown-license nếu nhóm đã kiểm tra quyền dùng.")
        rows, stats = import_source(name, args.cache, args.dev_share)
        splits = {s: sum(r["split"] == s for r in rows) for s in ("train", "dev", "test")}
        print(f"{name}: {stats} -> {splits}")
        all_rows.extend(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Đã ghi {len(all_rows)} cặp vào {args.output}")


if __name__ == "__main__":
    main()
