import ast
import json
import zipfile
from pathlib import Path

from app.nllb import CHUNK_MAX_CHARS, NllbEngine, group_sentences
from app.ocr import order_ocr_lines
from app.translate import Glossary, Translator, _adapter_ready
from scripts.export_checkpoint import pick_best_checkpoint
from scripts.kaggle_auto_train import build_staging, install_bundle
from training.evaluate import length_bucket, length_ratio, repetition_rate
from training.finetune_nllb import (
    balance_by_temperature,
    both_directions,
    has_data_rows,
    latest_checkpoint,
    synthesize_long_pairs,
    upsample_long_sentences,
)

ROOT = Path(__file__).resolve().parents[1]


class _EchoEngine:
    name = "echo"

    def pairs(self):
        return {("vi", "en"), ("en", "vi")}

    def step_name(self, source, target):
        return "echo"

    def translate(self, text, source, target):
        return f"[{text}]"


def test_translator_preserves_paragraph_breaks():
    result = Translator(Glossary(), _EchoEngine()).translate("  Câu một.   Câu hai.\n\n  Đoạn   hai. ", "vi", "en")
    assert result["text"] == "[Câu một. Câu hai.\n\nĐoạn hai.]"


def test_group_sentences_merges_context_within_limit():
    chunks = group_sentences(["Câu một.", "Câu hai.", "Câu ba rất dài " * 20], max_chars=40)
    assert chunks[0] == "Câu một. Câu hai."
    assert len(chunks) == 2
    assert CHUNK_MAX_CHARS >= 400


def test_adapter_ready_supports_peft_and_full_checkpoints(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _adapter_ready(str(empty)) is None

    peft_dir = tmp_path / "peft"
    peft_dir.mkdir()
    (peft_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert _adapter_ready(str(peft_dir)) == str(peft_dir)

    meta_only = tmp_path / "meta_only"
    meta_only.mkdir()
    (meta_only / "config.json").write_text("{}", encoding="utf-8")
    assert _adapter_ready(str(meta_only)) is None

    full_dir = tmp_path / "full"
    full_dir.mkdir()
    (full_dir / "config.json").write_text("{}", encoding="utf-8")
    (full_dir / "model.safetensors").write_bytes(b"weights")
    assert _adapter_ready(str(full_dir)) == str(full_dir)


def test_long_sentence_augmentation_helpers():
    rows = [
        ("tyz", "vi", "một từ", "one"),
        ("tyz", "vi", "câu thứ nhất có đủ tám từ trở lên nhé", "câu tiếng Việt thứ nhất dài hơn tám từ"),
        ("tyz", "vi", "câu thứ hai cũng đủ dài để ghép câu", "câu tiếng Việt thứ hai đủ dài để nối"),
        ("rad", "vi", "klei dŭm sa dua tlâo pă", "câu Ê Đê mẫu số một khá dài"),
    ]
    upsampled = upsample_long_sentences(rows, min_words=8, factor=2)
    assert len(upsampled) == 7

    synth = synthesize_long_pairs(rows, ratio=1.0, min_words=4, max_sents=2, seed=7)
    assert len(synth) == 3
    for src_lang, tgt_lang, src, tgt in synth:
        assert src_lang == "tyz" and tgt_lang == "vi"
        assert src.count(".") >= 2 and tgt.count(".") >= 2

    balanced = balance_by_temperature(rows * 3, temperature=2.0, seed=7)
    assert len(balanced) == len(rows) * 3
    rad_count = sum(1 for r in balanced if r[0] == "rad")
    assert rad_count > 3


def test_evaluate_long_metrics():
    assert length_bucket("một hai ba") == "1-7w"
    assert length_bucket(" ".join(["w"] * 10)) == "8-15w"
    assert length_bucket(" ".join(["w"] * 16)) == "16+w"
    assert repetition_rate(["a b c a b c", "a b c d e"]) == 0.5
    assert length_ratio(["a b c d"], ["x y"]) == 2.0


def test_kaggle_notebook_and_auto_installer(tmp_path: Path):
    for rel in (
        Path("notebooks/kaggle/train_nllb_ethnic.ipynb"),
        Path("notebooks/kaggle/train_nllb_cpu.ipynb"),
        Path("notebooks/kaggle/train_whisper_hmn.ipynb"),
        Path("notebooks/kaggle/train_whisper_bahnar.ipynb"),
        Path("notebooks/colab/train_nllb_ethnic_colab.ipynb"),
    ):
        nb = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        assert nb["nbformat"] == 4
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))

    staging = tmp_path / "staging"
    meta_path = build_staging(staging, "demo/ngonsacviet-train-nllb-ethnic", "NvidiaTeslaP100")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["id"] == "demo/ngonsacviet-train-nllb-ethnic"
    assert meta["machine_shape"] == "NvidiaTeslaP100"

    bundle_zip = tmp_path / "nllb-ethnic-ready.zip"
    with zipfile.ZipFile(bundle_zip, "w") as zf:
        zf.writestr("nllb-ethnic/adapter_config.json", '{"peft_type": "LORA"}')
        zf.writestr("nllb-ethnic/ngonsacviet.json", '{"languages": ["tyz"], "evaluation": {"tyz->vi": {"chrf": 52.1, "copy_chrf": 30.0}}}')

    target = tmp_path / "installed"
    installed_meta = install_bundle(bundle_zip, target_dir=target)
    assert installed_meta["languages"] == ["tyz"]
    assert (target / "adapter_config.json").is_file()


def test_both_directions_preserves_upsampled_rows_when_dedup_false():
    rows = [("tyz", "vi", "câu dài đủ tám từ trở lên nhé bạn", "câu tiếng Việt tương ứng cũng dài")]
    upsampled = upsample_long_sentences(rows, min_words=8, factor=2)
    assert len(both_directions(upsampled, dedup=True)) == 2
    assert len(both_directions(upsampled, dedup=False)) == 4


def test_order_ocr_lines_handles_large_height_and_tilt():
    def box(x, y, w=40, h=80):
        return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

    results = [
        (box(200, 120), "sau", 0.9),
        (box(20, 10), "Dòng", 0.9),
        (box(20, 130), "Dòng", 0.9),
        (box(200, 35), "đầu", 0.9),
    ]
    assert order_ocr_lines(results) == ["Dòng đầu", "Dòng sau"]


def test_nllb_passed_checks_length_and_repetition(tmp_path: Path):
    from app.checkpoints import write_metadata

    write_metadata(
        tmp_path,
        {
            "languages": ["tyz", "rad"],
            "evaluation": {
                "tyz->vi": {"chrf": 45.0, "copy_chrf": 25.0, "length_ratio": 0.95, "repetition_rate": 0.05},
                "vi->tyz": {"chrf": 45.0, "copy_chrf": 25.0, "length_ratio": 0.40, "repetition_rate": 0.05},
                "rad->vi": {"chrf": 45.0, "copy_chrf": 25.0, "length_ratio": 1.0, "repetition_rate": 0.30},
            },
        },
    )
    engine = NllbEngine("unused-base", str(tmp_path))
    assert engine.passed("tyz", "vi") is True
    assert engine.passed("vi", "tyz") is False
    assert engine.passed("rad", "vi") is False


def test_latest_checkpoint_picks_highest_numeric_step(tmp_path: Path):
    ckpts = tmp_path / "checkpoints"
    assert latest_checkpoint(ckpts) is None

    ckpts.mkdir()
    assert latest_checkpoint(ckpts) is None

    (ckpts / "checkpoint-abc").mkdir()
    (ckpts / "other").mkdir()
    assert latest_checkpoint(ckpts) is None

    (ckpts / "checkpoint-500").mkdir()
    best = ckpts / "checkpoint-1500"
    best.mkdir()
    assert latest_checkpoint(ckpts) == best


def test_has_data_rows_skips_header_only_csv(tmp_path: Path):
    header_only = tmp_path / "header_only.csv"
    header_only.write_text("source_lang,target_lang,source,target,verified\n\n", encoding="utf-8")
    assert has_data_rows(header_only) is False

    with_row = tmp_path / "with_row.csv"
    with_row.write_text(
        "source_lang,target_lang,source,target,verified\ntyz,vi,câu nguồn,câu đích,true\n",
        encoding="utf-8",
    )
    assert has_data_rows(with_row) is True


def test_pick_best_checkpoint_prefers_trainer_state_best(tmp_path: Path):
    ckpts = tmp_path / "checkpoints"
    c300 = ckpts / "checkpoint-300"
    c450 = ckpts / "checkpoint-450"
    c300.mkdir(parents=True)
    c450.mkdir(parents=True)
    (c450 / "trainer_state.json").write_text(
        json.dumps(
            {
                "best_model_checkpoint": "/content/nllb-ethnic/checkpoints/checkpoint-300",
                "best_metric": 1.85,
                "global_step": 450,
                "epoch": 0.36,
            }
        ),
        encoding="utf-8",
    )
    chosen, info = pick_best_checkpoint(ckpts)
    assert chosen == c300
    assert info["chosen"] == "checkpoint-300"
    assert info["latest"] == "checkpoint-450"
    assert info["global_step"] == 300
    assert info["best_metric"] == 1.85


def test_import_public_reads_csv_with_bom_and_parallel_text(tmp_path: Path):
    from training.import_public import read_csv_rows, read_parallel_text

    csv_path = tmp_path / "muong.csv"
    csv_path.write_bytes("\ufeffSentenceM,SentenceV\nho pắt ản,tớ bắt được\n".encode("utf-8"))
    assert read_csv_rows(csv_path, "SentenceM", "SentenceV") == [("ho pắt ản", "tớ bắt được")]

    ba_path = tmp_path / "train.ba"
    vi_path = tmp_path / "train.vi"
    ba_path.write_text("Pơgơl tơdrong\n", encoding="utf-8")
    vi_path.write_text("Chỉ đạo công việc\n", encoding="utf-8")
    assert read_parallel_text(ba_path, vi_path) == [("Pơgơl tơdrong", "Chỉ đạo công việc")]


def test_glossary_matches_trailing_punctuation_and_asr_pretrained(monkeypatch, tmp_path: Path):
    import pytest

    from app.asr import CheckpointNotVerified, PretrainedWhisperEngine
    from app.io_engines import EngineUnavailable, asr_engine
    from app.translate import _add_pair

    g = Glossary()
    _add_pair(g, "rad", "vi", "Lač jăk lu sonăk.", "Cám ơn nhiều lắm.")
    assert g.translate("Cám ơn nhiều lắm", "vi", "rad") == "Lač jăk lu sonăk."
    assert g.translate("Lač jăk lu sonăk!", "rad", "vi") == "Cám ơn nhiều lắm."

    asr_engine.cache_clear()
    monkeypatch.setenv("ML_ASR_ENGINE", "whisper")
    monkeypatch.setenv("ML_ASR_MODEL", "vinai/PhoWhisper-small")
    engine = asr_engine()
    assert isinstance(engine, PretrainedWhisperEngine)
    assert engine.whisper_language == "vi"

    bad_ckpt = tmp_path / "bad-whisper"
    bad_ckpt.mkdir()
    (bad_ckpt / "ngonsacviet.json").write_text(
        json.dumps({"task": "speech_translation", "language": "hmn", "test_wer": 0.9, "trained_on_unverified": True}),
        encoding="utf-8",
    )
    asr_engine.cache_clear()
    monkeypatch.setenv("ML_ASR_MODEL", str(bad_ckpt))
    try:
        with pytest.raises(EngineUnavailable) as exc_info:
            asr_engine()
        assert isinstance(exc_info.value.__cause__, CheckpointNotVerified)
    finally:
        asr_engine.cache_clear()
