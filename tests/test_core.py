from pathlib import Path

import pytest

from app.detect import detect_language
from app.translate import Glossary, RouteNotReady, Translator, builtin_glossary, plan_route
from training.validate_pairs import PairFileError, validate_pairs

ROOT = Path(__file__).resolve().parents[1]


def test_catalog_contains_the_five_languages():
    from app.catalog import ethnic_codes

    assert ethnic_codes() == ["tyz", "rad", "cjm", "hmn", "jra"]


def test_direct_glossary_translation():
    result = Translator(builtin_glossary()).translate("Xin chào", "vi", "en")
    assert result["text"] == "Hello"
    assert result["route"] == "direct"
    assert result["engine"] == "glossary"


def test_pivot_uses_vietnamese_when_both_hops_exist():
    glossary = Glossary(
        {
            ("tyz", "vi"): {"pair_a": "xin chào"},
            ("vi", "en"): {"xin chào": "Hello"},
        }
    )
    result = Translator(glossary).translate("pair_a", "tyz", "en")
    assert result["text"] == "Hello"
    assert result["route"] == "pivot"
    assert [step["target"] for step in result["steps"]] == ["vi", "en"]


def test_ethnic_direction_is_not_invented():
    with pytest.raises(RouteNotReady):
        Translator(builtin_glossary()).translate("xin chào", "vi", "tyz")


def test_plan_route_identity_and_direct():
    assert plan_route("vi", "vi", set()) == []
    assert plan_route("vi", "en", {("vi", "en")}) == [("vi", "en")]


@pytest.fixture
def no_langid(tmp_path: Path, monkeypatch):
    from app.langid import load_langid

    monkeypatch.setenv("ML_MODELS_DIR", str(tmp_path))
    load_langid.cache_clear()
    yield
    load_langid.cache_clear()


def test_detect_cham_script_vietnamese_and_unknown_latin(no_langid):
    assert detect_language("ꨀꨁ")["language"] == "cjm"
    assert detect_language("Xin chào các bạn")["language"] == "vi"
    assert detect_language("Hello")["language"] == "en"
    assert detect_language("kơdih")["language"] == "und"


def test_pair_file_rejects_unverified_rows(tmp_path: Path):
    path = tmp_path / "pairs.csv"
    path.write_text(
        "source_lang,target_lang,source,target,verified\ntyz,vi,pair_a,xin chào,false\n",
        encoding="utf-8",
    )
    with pytest.raises(PairFileError):
        validate_pairs(path)


def test_sample_pair_file_dry_run():
    summary = validate_pairs(ROOT / "training" / "sample_vi_en.csv")
    assert summary["rows"] == 2
    assert summary["pairs"]["vi->en"] == 1


def test_text_round_trip_for_letters_missing_in_nllb():
    from app.textnorm import clean, from_model, to_model

    assert clean("“Brơi  hmâo”\u00a0—") == '"Brơi hmâo" -'
    encoded = to_model("Ƀuôn ƀu Ŏ")
    assert "ƀ" not in encoded and "Ƀ" not in encoded and "Ŏ" not in encoded
    assert from_model(encoded) == "Ƀuôn ƀu Ŏ"


class FakeNeural:
    name = "fake"

    def pairs(self):
        return {("tyz", "vi"), ("vi", "tyz")}

    def step_name(self, source, target):
        return "fake-ethnic"

    def translate(self, text, source, target):
        return f"<{text}>"


def test_neural_ethnic_step_is_marked_unreviewed():
    from app.translate import UNREVIEWED_WARNING

    glossary = Glossary({("vi", "en"): {"<cau a>": "Sentence"}})
    result = Translator(glossary, FakeNeural()).translate("cau a", "tyz", "en")
    assert result["route"] == "pivot"
    assert [step["engine"] for step in result["steps"]] == ["fake-ethnic", "glossary"]
    assert result["warnings"] == [UNREVIEWED_WARNING]


def test_glossary_hit_has_no_machine_warning():
    result = Translator(builtin_glossary(), FakeNeural()).translate("Xin chào", "vi", "en")
    assert result["warnings"] == []


def test_nllb_engine_offers_only_trained_languages(tmp_path: Path):
    from app.nllb import METADATA, NllbEngine

    base = {("vi", "en"), ("en", "vi")}
    (tmp_path / METADATA).write_text('{"languages": ["tyz"]}', encoding="utf-8")
    assert NllbEngine("unused-base", str(tmp_path)).pairs() == base
    engine = NllbEngine("unused-base", str(tmp_path), gated=False)
    assert engine.pairs() == base | {("tyz", "vi"), ("vi", "tyz")}
    assert engine.step_name("tyz", "vi") == "nllb-ethnic"
    assert engine.step_name("vi", "en") == "nllb"
    assert NllbEngine("unused-base").pairs() == base


def test_nllb_engine_serves_only_directions_that_beat_copying(tmp_path: Path):
    from app.checkpoints import write_metadata
    from app.nllb import NllbEngine

    write_metadata(
        tmp_path,
        {
            "languages": ["tyz"],
            "evaluation": {
                "tyz->vi": {"chrf": 41.0, "copy_chrf": 30.0},
                "vi->tyz": {"chrf": 28.0, "copy_chrf": 30.0},
            },
        },
    )
    assert ("tyz", "vi") in NllbEngine("unused-base", str(tmp_path)).pairs()
    assert ("vi", "tyz") not in NllbEngine("unused-base", str(tmp_path)).pairs()


def test_nllb_tokenizer_has_no_unknown_pieces_for_ethnic_letters():
    from app.paths import models_dir
    from app.textnorm import from_model, to_model

    folder = models_dir() / "nllb-200-distilled-600M"
    if not (folder / "tokenizer_config.json").is_file():
        pytest.skip("NLLB tokenizer chưa tải")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(folder)
    text = "Ƀuôn ƀu Ŏ ŏ Ĭ ĭ Ŭ ŭ Ĕ ĕ Ê ê Ơ ơ Ư ư Đ đ"
    ids = tokenizer(to_model(text), add_special_tokens=False).input_ids
    assert tokenizer.unk_token_id not in ids
    assert from_model(tokenizer.decode(ids)) == text


def test_held_out_filter_covers_both_sides():
    from training.finetune_nllb import held_out_sentences, leaks

    held = held_out_sentences([("tyz", "vi", "Pây hâư", "Đi đâu")])
    assert leaks(("tyz", "vi", "khác hẳn", "đi đâu"), held)
    assert leaks(("tyz", "vi", "pây hâư", "khác"), held)
    assert not leaks(("tyz", "vi", "khác", "khác"), held)


def test_akhar_thrah_never_reaches_latin_engines():
    with pytest.raises(RouteNotReady):
        Translator(builtin_glossary()).translate("ꨀꨁ", "cjm", "vi")


def test_one_word_input_is_not_sent_to_the_char_model(monkeypatch):
    from app import detect

    class Always:
        def predict(self, text):
            return "rad", 0.999

    monkeypatch.setattr(detect, "load_langid", lambda: Always())
    assert detect.detect_language("ok")["language"] == "und"
    assert detect.detect_language("ƀuôn sang kâo")["language"] == "rad"


def test_reimport_keeps_reviewer_edits(tmp_path: Path):
    import csv

    from training.import_speech import COLUMNS, read_previous

    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerow({column: "" for column in COLUMNS} | {"source_dataset": "a/b", "clip_id": "7", "verified": "true"})
    assert read_previous(manifest)[("a/b", "7")]["verified"] == "true"
    assert read_previous(tmp_path / "missing.csv") == {}


def test_split_sentences():
    from app.nllb import split_sentences

    assert split_sentences("Tôi đi. Bạn ở! Ừ?") == ["Tôi đi.", "Bạn ở!", "Ừ?"]
    assert split_sentences("không dấu chấm") == ["không dấu chấm"]
    long = ", ".join(["một vế câu dài"] * 40)
    assert len(split_sentences(long)) == 40


def test_langid_model_separates_languages(tmp_path: Path, monkeypatch):
    import json

    from app.langid import CharNgramModel, load_langid

    model = CharNgramModel.train(
        {
            "rad": ["ƀuôn sang kâo", "amai kâo dôk mă abăn", "ƀu hmâo klei"],
            "vi": ["chị tôi đang lấy chăn", "hôm nay trời đẹp", "các bạn khỏe không"],
        },
        min_count=1,
    )
    assert model.predict("amai kâo dôk")[0] == "rad"
    (tmp_path / "langid.json").write_text(json.dumps(model.to_json(), ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("ML_MODELS_DIR", str(tmp_path))
    load_langid.cache_clear()
    try:
        found = detect_language("amai kâo dôk mă abăn")
        assert found["language"] == "rad"
        assert found["method"] == "char_ngram"
        assert detect_language("Hello")["language"] == "en"
    finally:
        load_langid.cache_clear()


def test_training_rows_are_reviewed_split_and_bidirectional(tmp_path: Path):
    from training.finetune_nllb import both_directions, read_pairs

    path = tmp_path / "pairs.csv"
    path.write_text(
        "source_lang,target_lang,source,target,verified,split\n"
        "tyz,vi,pair_a,câu a,true,train\n"
        "tyz,vi,pair_b,câu b,false,train\n"
        "tyz,vi,pair_c,câu c,true,test\n"
        "vi,en,xin chào,Hello,true,train\n",
        encoding="utf-8",
    )
    rows = read_pairs([path], "train")
    assert rows == [("tyz", "vi", "pair_a", "câu a")]
    assert both_directions(rows) == [("tyz", "vi", "pair_a", "câu a"), ("vi", "tyz", "câu a", "pair_a")]


def test_import_split_keeps_declared_test_and_filters_noise():
    from training.import_public import assign_split, keep

    assert assign_split("test", "x", 0.5, has_dev=False) == "test"
    assert assign_split("train", "x", 0.5, has_dev=True) == "train"
    assert keep("một hai ba bốn năm sáu bảy tám chín mười", "một") is False
    assert keep("", "câu") is False
    assert keep("noọng", "em gái") is True
