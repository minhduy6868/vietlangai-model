import io
import math
import struct
import wave

import pytest

from app import io_engines
from app.asr import UNREVIEWED_SPEECH_TRANSLATION, CheckpointNotVerified, WhisperEngine
from app.audio import SAMPLE_RATE, AudioDecodeError, decode_audio
from app.checkpoints import write_metadata
from app.ocr import LATIN_OCR_WARNING, EasyOcrEngine


CACHED = (io_engines.asr_engine, io_engines.ocr_engine)


@pytest.fixture(autouse=True)
def fresh_engines(monkeypatch):
    for name in ("ML_ASR_ENGINE", "ML_ASR_MODEL", "ML_OCR_ENGINE"):
        monkeypatch.delenv(name, raising=False)
    for factory in CACHED:
        factory.cache_clear()
    yield
    for factory in CACHED:
        factory.cache_clear()


def wav_bytes(seconds: float, rate: int = 8000) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        frames = int(seconds * rate)
        handle.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i / 10))) for i in range(frames)))
    return buffer.getvalue()


def good_metadata(**overrides) -> dict:
    meta = {
        "task": "speech_translation",
        "language": "hmn",
        "text_language": "vi",
        "whisper_language": "vi",
        "test_wer": 0.4,
        "trained_on_unverified": False,
    }
    meta.update(overrides)
    return meta


def test_error_rates():
    from training.speech_metrics import error_rates

    assert error_rates(["xin chào bạn"], ["Xin chào, bạn!"]) == {"wer": 0.0, "cer": 0.0}
    rates = error_rates(["xin chao"], ["xin chào"])
    assert rates["wer"] == 0.5
    assert 0 < rates["cer"] < 0.5


def test_looks_vietnamese_keeps_hmong_out():
    from training.import_speech import looks_vietnamese

    for text in ["Mưa to quá.", "Cảm ơn giao dịch viên.", "Nhẹ tay thôi.", "Tôi đi chợ"]:
        assert looks_vietnamese(text), text
    for text in ["Kuv mus tsev", "Nyob zoo os", "Koj puas noj mov lawm"]:
        assert not looks_vietnamese(text), text


def test_usable_drops_misaligned_clips():
    from training.finetune_whisper import usable

    assert usable({"duration": "2.0", "text": "Tôi đi chợ"})
    assert not usable({"duration": "0.5", "text": "Câu này quá dài so với độ dài của đoạn âm thanh"})
    assert not usable({"duration": "45", "text": "dài quá"})
    assert not usable({"duration": "", "text": "x"})
    assert not usable({"duration": "abc", "text": "x"})


@pytest.mark.parametrize(
    "meta",
    [
        {},
        good_metadata(test_wer=None),
        good_metadata(task="translate"),
        good_metadata(trained_on_unverified=True),
        {key: value for key, value in good_metadata().items() if key != "trained_on_unverified"},
    ],
)
def test_whisper_refuses_unchecked_checkpoints(tmp_path, meta):
    if meta:
        write_metadata(tmp_path, meta)
    with pytest.raises(CheckpointNotVerified):
        WhisperEngine(str(tmp_path))


def test_whisper_accepts_checked_checkpoint(tmp_path):
    write_metadata(tmp_path, good_metadata())
    engine = WhisperEngine(str(tmp_path))
    assert engine.text_language == "vi"
    assert engine.spoken_language == "hmn"
    assert engine.warning == UNREVIEWED_SPEECH_TRANSLATION


def test_unconfigured_engines_report_unloaded():
    speech = io_engines.transcribe(b"abc", "a.wav")
    assert speech["engine"] == "unloaded" and speech["text"] == ""
    image = io_engines.read_text(b"abc", "a.png")
    assert image["engine"] == "unloaded" and image["text"] == ""
    assert io_engines.transcribe(b"", "a.wav")["warnings"] == ["Tệp âm thanh trống."]


def test_asr_env_with_unchecked_checkpoint_stays_unloaded(tmp_path, monkeypatch):
    write_metadata(tmp_path, good_metadata(trained_on_unverified=True))
    monkeypatch.setenv("ML_ASR_ENGINE", "whisper")
    monkeypatch.setenv("ML_ASR_MODEL", str(tmp_path))
    assert io_engines.transcribe(b"abc", "a.wav")["engine"] == "unloaded"


class FakeWhisper:
    name = "whisper"
    text_language = "vi"
    spoken_language = "hmn"
    warning = UNREVIEWED_SPEECH_TRANSLATION

    def __init__(self):
        self.lengths = []

    def transcribe(self, samples):
        self.lengths.append(len(samples))
        return "tôi đi chợ"


def test_transcribe_with_fake_engine(monkeypatch):
    fake = FakeWhisper()
    monkeypatch.setattr(io_engines, "asr_engine", lambda: fake)
    monkeypatch.setattr(io_engines, "MAX_AUDIO_SECONDS", 1)
    result = io_engines.transcribe(wav_bytes(2.0), "a.wav")
    assert result["text"] == "tôi đi chợ"
    assert result["language"] == "vi" and result["spoken_language"] == "hmn"
    assert UNREVIEWED_SPEECH_TRANSLATION in result["warnings"]
    assert len(result["warnings"]) == 2
    assert fake.lengths == [SAMPLE_RATE]


def test_transcribe_rejects_garbage_audio(monkeypatch):
    monkeypatch.setattr(io_engines, "asr_engine", lambda: FakeWhisper())
    result = io_engines.transcribe(b"not audio at all", "a.wav")
    assert result["text"] == "" and result["engine"] == "whisper"


def test_decode_audio_resamples_to_16k():
    samples = decode_audio(wav_bytes(1.0, rate=8000))
    assert abs(len(samples) - SAMPLE_RATE) < SAMPLE_RATE * 0.02
    with pytest.raises(AudioDecodeError):
        decode_audio(b"\x00" * 64)


class FakeReader:
    def readtext(self, image, detail, paragraph):
        box = lambda x, y: [[x, y], [x + 50, y], [x + 50, y + 15], [x, y + 15]]  # noqa: E731
        return [
            (box(200, 62), "ƀuôn", 0.6),
            (box(10, 60), "Hlăm", 0.8),
            (box(10, 10), "Kâo", 1.0),
        ]


def test_ocr_orders_lines_top_to_bottom_then_left_to_right():
    engine = EasyOcrEngine()
    engine._reader = FakeReader()
    assert engine.read(b"img") == ("Kâo Hlăm ƀuôn", 0.8)


def test_read_text_with_fake_engine(monkeypatch):
    engine = EasyOcrEngine()
    engine._reader = FakeReader()
    monkeypatch.setattr(io_engines, "ocr_engine", lambda: engine)
    result = io_engines.read_text(png_bytes(40, 20), "a.png")
    assert result["text"] == "Kâo Hlăm ƀuôn"
    assert result["warnings"] == [LATIN_OCR_WARNING]
    assert io_engines.read_text(b"img", "a.png")["text"] == ""


def test_load_image_shrinks_and_rejects():
    from app.ocr import MAX_SIDE, ImageRejected, load_image

    assert max(load_image(png_bytes(MAX_SIDE * 2, 10)).shape[:2]) == MAX_SIDE
    with pytest.raises(ImageRejected):
        load_image(png_bytes(5000, 5000))
    with pytest.raises(ImageRejected):
        load_image(b"not an image")


def test_decode_audio_stops_at_the_cap():
    samples = decode_audio(wav_bytes(3.0), max_seconds=1)
    assert len(samples) == SAMPLE_RATE + 1


def png_bytes(width: int, height: int) -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buffer, format="PNG")
    return buffer.getvalue()
