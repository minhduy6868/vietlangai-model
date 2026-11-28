import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.catalog import load_catalog
from app.detect import detect_language
from app.io_engines import read_text, transcribe
from app.translate import RouteNotReady, Translator, UnknownLanguage, build_translator

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

_translator: Translator | None = None
_translator_lock = threading.Lock()


def translator() -> Translator:
    global _translator
    with _translator_lock:
        if _translator is None:
            _translator = build_translator()
    return _translator


def _prewarm_translator() -> None:
    tr = translator()
    neural = getattr(tr, "neural", None)
    if neural is not None and hasattr(neural, "_load"):
        neural._load()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if os.getenv("ML_PREWARM") == "1" and os.getenv("ML_MT_ENGINE") in ("nllb", "mt5"):
        threading.Thread(target=_prewarm_translator, daemon=True, name="ml-prewarm").start()
    yield


app = FastAPI(title="Ngon Sac Viet ML", version="0.1.0", lifespan=lifespan)


class TranslateIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    source_lang: str = "auto"
    target_lang: str


class DetectIn(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "ml"}


@app.get("/v1/languages")
def languages() -> list[dict]:
    return load_catalog()["languages"]


@app.post("/v1/detect")
def detect(body: DetectIn) -> dict:
    return detect_language(body.text)


@app.post("/v1/translate")
def translate(body: TranslateIn) -> dict:
    try:
        return translator().translate(body.text, body.source_lang, body.target_lang)
    except UnknownLanguage as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "unknown_language", "message": f"Không có mã ngôn ngữ {exc.code}."},
        ) from exc
    except RouteNotReady as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "model_not_ready",
                "message": "Chưa có mô hình hoặc cặp câu đã rà soát cho hướng dịch này.",
                "source_lang": exc.source,
                "target_lang": exc.target,
            },
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "empty_text", "message": "Văn bản trống."},
        ) from exc


async def read_upload(file: UploadFile) -> bytes:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail={"code": "file_too_large", "message": f"Tệp vượt quá {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."},
        )
    return data


@app.post("/v1/ocr")
async def ocr(file: UploadFile = File(...)) -> dict:
    data = await read_upload(file)
    return await run_in_threadpool(read_text, data, file.filename or "image")


@app.post("/v1/asr")
async def asr(file: UploadFile = File(...)) -> dict:
    data = await read_upload(file)
    return await run_in_threadpool(transcribe, data, file.filename or "audio")
