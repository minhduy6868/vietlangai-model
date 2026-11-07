"""Import H'Mông speech datasets into data/speech/manifest.csv.

    python training/import_speech.py --allow-unknown-license
    python training/import_speech.py --allow-unknown-license --mark-verified   # sau khi nhóm đã nghe lại

Rows are written verified=false unless --mark-verified is given: transcripts
must be checked against the domestic Vietnamese H'Mông orthography before
training. Re-running keeps the text, text_lang, orthography, verified and
split a reviewer edited in manifest.csv. Run in its own process (pyarrow,
see import_public.py).

`text_lang` records what the text column really is. Both public H'Mông sets
pair H'Mông audio with a Vietnamese sentence, so they feed speech
translation (hmn audio -> vi text), not H'Mông ASR. `duration` lets training
drop clips whose text cannot fit the audio. `orthography` stays empty until a
reviewer writes `vn` for H'Mông text in the domestic script; H'Mông ASR
training skips every other row so overseas RPA never mixes in.

The split hashes the text, not the speaker, because the public sets carry no
speaker id. Test WER therefore includes voices heard in training.
"""

import argparse
import csv
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.textnorm import clean  # noqa: E402
from import_public import bucket  # noqa: E402

HF = "https://huggingface.co/datasets/{repo}/resolve/main/{file}"
SOURCES = {
    "hmong-vn-audio": {
        "repo": "ThaiKami/hmong-vn-audio",
        "file": "data/train-00000-of-00001.parquet",
        "language": "hmn",
        "license": "unknown",
        "id_field": "audio_name",
    },
    "hmong-low-bleu": {
        "repo": "nhatminh/hmong_low_bleu",
        "file": "data/train-00000-of-00001.parquet",
        "language": "hmn",
        "license": "unknown",
        "id_field": None,
    },
}
REVIEWED_COLUMNS = ("text", "text_lang", "orthography", "verified", "split")
COLUMNS = (
    "path",
    "text",
    "language",
    "text_lang",
    "orthography",
    "duration",
    "split",
    "verified",
    "source_dataset",
    "license",
    "clip_id",
)
# Vietnamese tone marks. The domestic H'Mông script writes tones with final
# consonant letters, so these do not appear in H'Mông text.
_TONED = set("àáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ")


_TONELESS_VI = {
    "không", "tôi", "đi", "đâu", "nhé", "hôm", "nay", "ơi", "mình", "chơi", "thêm", "phim",
    "sao", "cho", "xem", "mang", "em", "anh", "chị", "con", "cô", "alo", "hi", "vui", "suy", "nghĩ",
    "đang", "phải", "đây", "gian", "lên", "công", "an", "hiểu", "chưa", "ăn", "uống", "về", "nhà",
}


def looks_vietnamese(text: str) -> bool:
    words = [w.strip(".,!?;:\"'()") for w in text.casefold().split()]
    words = [w for w in words if w]
    if not words:
        return False
    hits = sum(any(ch in _TONED for ch in word) or word in _TONELESS_VI for word in words)
    return hits / len(words) >= 0.3


def duration_seconds(data: bytes) -> float:
    from app.audio import SAMPLE_RATE, AudioDecodeError, decode_audio

    try:
        return round(len(decode_audio(data)) / SAMPLE_RATE, 2)
    except AudioDecodeError:
        return 0.0


def assign_split(key: str, share: float) -> str:
    roll = bucket(key)
    if roll < share:
        return "dev"
    if roll < 2 * share:
        return "test"
    return "train"


def read_previous(manifest: Path) -> dict[tuple[str, str], dict]:
    """Rows a reviewer may have edited, keyed by dataset and clip id."""
    if not manifest.is_file():
        return {}
    with manifest.open(encoding="utf-8", newline="") as handle:
        return {(row["source_dataset"], row["clip_id"]): row for row in csv.DictReader(handle)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include", nargs="*", default=["hmong-vn-audio"], choices=sorted(SOURCES))
    parser.add_argument("--allow-unknown-license", action="store_true")
    parser.add_argument("--mark-verified", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "speech")
    parser.add_argument("--share", type=float, default=0.1)
    args = parser.parse_args()

    import pyarrow.parquet as pq

    manifest = args.output / "manifest.csv"
    previous = read_previous(manifest)
    rows = []
    for name in args.include:
        spec = SOURCES[name]
        if spec["license"] == "unknown" and not args.allow_unknown_license:
            raise SystemExit(f"{name} không ghi license. Thêm --allow-unknown-license nếu nhóm đã kiểm tra quyền dùng.")
        raw = args.output / "raw" / spec["repo"].replace("/", "__") / spec["file"]
        if not raw.is_file():
            raw.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(HF.format(repo=spec["repo"], file=spec["file"]), raw)
        clips = args.output / name
        clips.mkdir(parents=True, exist_ok=True)
        kept = skipped = 0
        for i, record in enumerate(pq.read_table(raw).to_pylist()):
            audio = record.get("audio") or {}
            text = clean(record.get("text") or "")
            if not audio.get("bytes") or not text:
                skipped += 1
                continue
            suffix = Path(audio.get("path") or "clip.wav").suffix or ".wav"
            clip = clips / f"{i:06d}{suffix}"
            clip.write_bytes(audio["bytes"])
            clip_id = str(record.get(spec["id_field"]) or i) if spec["id_field"] else str(i)
            row = {
                    "path": clip.relative_to(args.output).as_posix(),
                    "text": text,
                    "language": spec["language"],
                    "text_lang": "vi" if looks_vietnamese(text) else spec["language"],
                    "orthography": "",
                    "duration": duration_seconds(audio["bytes"]),
                    "split": assign_split(text.casefold(), args.share),
                    "verified": "false",
                    "source_dataset": spec["repo"],
                    "license": spec["license"],
                    "clip_id": clip_id,
                }
            reviewed = previous.get((spec["repo"], clip_id))
            if reviewed:
                row.update({column: reviewed[column] for column in REVIEWED_COLUMNS if column in reviewed})
            if args.mark_verified:
                row["verified"] = "true"
            rows.append(row)
            kept += 1
        vietnamese = sum(r["text_lang"] == "vi" for r in rows if r["source_dataset"] == spec["repo"])
        print(f"{name}: {kept} đoạn ({vietnamese} đoạn có chữ tiếng Việt), bỏ {skipped}")

    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    splits = {s: sum(r["split"] == s for r in rows) for s in ("train", "dev", "test")}
    print(f"Đã ghi {len(rows)} đoạn vào {manifest} {splits}")


if __name__ == "__main__":
    main()
