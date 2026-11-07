import os
from functools import lru_cache

from app.asr import CheckpointNotVerified, WhisperEngine
from app.audio import SAMPLE_RATE, AudioDecodeError, decode_audio
from app.ocr import LATIN_OCR_WARNING, EasyOcrEngine, ImageRejected, load_image

MAX_AUDIO_SECONDS = 120


class EngineUnavailable(Exception):
    pass


@lru_cache
def asr_engine() -> WhisperEngine:
    kind = os.getenv("ML_ASR_ENGINE", "unloaded")
    if kind != "whisper":
        raise EngineUnavailable(
            "Mô hình giọng nói chưa được nạp. Fine-tune Whisper và đặt ML_ASR_ENGINE=whisper khi checkpoint đã kiểm tra."
        )
    model_dir = os.getenv("ML_ASR_MODEL", "")
    try:
        return WhisperEngine(model_dir)
    except CheckpointNotVerified as exc:
        raise EngineUnavailable(f"Checkpoint {model_dir or '(trống)'} chưa có WER trên tập test nên không được nạp.") from exc


@lru_cache
def ocr_engine() -> EasyOcrEngine:
    if os.getenv("ML_OCR_ENGINE", "unloaded") != "easyocr":
        raise EngineUnavailable("OCR chưa được nạp. Đặt ML_OCR_ENGINE=easyocr cho ảnh chữ Latin; chữ Chăm Akhar Thrah là mô hình riêng.")
    return EasyOcrEngine()


def _unloaded(message: str, size: int, filename: str) -> dict:
    return {"text": "", "engine": "unloaded", "warnings": [message, f"Đã nhận {size} byte từ {filename}."]}


def transcribe(audio: bytes, filename: str) -> dict:
    if not audio:
        return {"text": "", "engine": "unloaded", "warnings": ["Tệp âm thanh trống."]}
    try:
        engine = asr_engine()
    except EngineUnavailable as exc:
        return _unloaded(str(exc), len(audio), filename)
    try:
        samples = decode_audio(audio, max_seconds=MAX_AUDIO_SECONDS)
    except AudioDecodeError as exc:
        return {"text": "", "engine": engine.name, "warnings": [str(exc)]}
    warnings = [engine.warning]
    if len(samples) > MAX_AUDIO_SECONDS * SAMPLE_RATE:
        samples = samples[: MAX_AUDIO_SECONDS * SAMPLE_RATE]
        warnings.append(f"Chỉ nhận {MAX_AUDIO_SECONDS} giây đầu.")
    text = engine.transcribe(samples)
    return {
        "text": text,
        "engine": engine.name,
        "language": engine.text_language,
        "spoken_language": engine.spoken_language,
        "warnings": warnings,
    }


def read_text(image: bytes, filename: str) -> dict:
    if not image:
        return {"text": "", "engine": "unloaded", "warnings": ["Tệp ảnh trống."]}
    try:
        engine = ocr_engine()
    except EngineUnavailable as exc:
        return _unloaded(str(exc), len(image), filename)
    try:
        pixels = load_image(image)
    except ImageRejected as exc:
        return {"text": "", "engine": engine.name, "warnings": [str(exc)]}
    text, confidence = engine.read(pixels)
    return {"text": text, "engine": engine.name, "confidence": confidence, "warnings": [LATIN_OCR_WARNING]}
