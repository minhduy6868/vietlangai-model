import threading

from app.audio import SAMPLE_RATE
from app.checkpoints import read_metadata
from app.textnorm import clean

CHUNK_SECONDS = 30
TASKS = ("asr", "speech_translation")
UNREVIEWED_ASR = "Bản ghi lời do máy nhận dạng, nên nghe lại trước khi dùng."
UNREVIEWED_SPEECH_TRANSLATION = "Câu tiếng Việt do máy dịch thẳng từ giọng nói, chưa được người bản ngữ rà soát."
PRETRAINED_ASR_MODELS = {
    "vinai/PhoWhisper-small",
    "vinai/PhoWhisper-base",
    "openai/whisper-small",
    "openai/whisper-base",
    "openai/whisper-medium",
}


class CheckpointNotVerified(Exception):
    pass


def _load_whisper(model_source: str):
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(model_source)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_source, dtype=dtype).to(device).eval()
    return model, processor


def _transcribe_chunks(model, processor, samples, whisper_language: str | None) -> str:
    import torch

    step = CHUNK_SECONDS * SAMPLE_RATE
    pieces = []
    for start in range(0, len(samples), step):
        chunk = samples[start : start + step]
        if len(chunk) < SAMPLE_RATE // 4:
            continue
        features = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        features = features.to(model.device, model.dtype)
        kwargs = {"task": "transcribe", "max_new_tokens": 225}
        if whisper_language:
            kwargs["language"] = whisper_language
        with torch.inference_mode():
            ids = model.generate(features, **kwargs)
        pieces.append(processor.batch_decode(ids, skip_special_tokens=True)[0])
    return clean(" ".join(pieces))


class PretrainedWhisperEngine:
    """Published Whisper/PhoWhisper checkpoint for live Vietnamese/English ASR."""

    name = "whisper"
    task = "asr"

    def __init__(self, model_id: str, whisper_language: str = "vi") -> None:
        self.model_id = model_id
        self.whisper_language = whisper_language
        self.spoken_language = whisper_language
        self.text_language = whisper_language
        self._bundle = None
        self._lock = threading.Lock()

    @property
    def warning(self) -> str:
        return UNREVIEWED_ASR

    def transcribe(self, samples) -> str:
        with self._lock:
            if self._bundle is None:
                self._bundle = _load_whisper(self.model_id)
            model, processor = self._bundle
            return _transcribe_chunks(model, processor, samples, self.whisper_language)


class WhisperEngine:
    """Fine-tuned Whisper for one spoken language.

    `asr` checkpoints write the spoken language; `speech_translation`
    checkpoints write Vietnamese. The checkpoint must carry ngonsacviet.json
    with the task, the spoken language, a WER measured on the held-out split,
    and trained_on_unverified=false.
    """

    name = "whisper"

    def __init__(self, model_dir: str) -> None:
        meta = read_metadata(model_dir)
        if meta.get("task") not in TASKS or not meta.get("language") or meta.get("test_wer") is None:
            raise CheckpointNotVerified(model_dir)
        if meta.get("trained_on_unverified", True):
            raise CheckpointNotVerified(model_dir)
        self.model_dir = model_dir
        self.task: str = meta["task"]
        self.spoken_language: str = meta["language"]
        self.text_language: str = meta.get("text_language") or meta["language"]
        self.whisper_language: str | None = meta.get("whisper_language")
        self.test_wer: float = meta["test_wer"]
        self._bundle = None
        self._lock = threading.Lock()

    @property
    def warning(self) -> str:
        return UNREVIEWED_SPEECH_TRANSLATION if self.task == "speech_translation" else UNREVIEWED_ASR

    def transcribe(self, samples) -> str:
        with self._lock:
            if self._bundle is None:
                self._bundle = _load_whisper(self.model_dir)
            model, processor = self._bundle
            return _transcribe_chunks(model, processor, samples, self.whisper_language)
