import threading

from app.audio import SAMPLE_RATE
from app.checkpoints import read_metadata
from app.textnorm import clean

CHUNK_SECONDS = 30
TASKS = ("asr", "speech_translation")
UNREVIEWED_ASR = "Bản ghi lời do máy nhận dạng, nên nghe lại trước khi dùng."
UNREVIEWED_SPEECH_TRANSLATION = "Câu tiếng Việt do máy dịch thẳng từ giọng nói, chưa được người bản ngữ rà soát."


class CheckpointNotVerified(Exception):
    pass


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
        import torch

        step = CHUNK_SECONDS * SAMPLE_RATE
        pieces = []
        with self._lock:
            model, processor = self._load()
            for start in range(0, len(samples), step):
                chunk = samples[start : start + step]
                if len(chunk) < SAMPLE_RATE // 4:
                    continue
                features = processor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
                features = features.to(model.device, model.dtype)
                kwargs = {"task": "transcribe", "max_new_tokens": 225}
                if self.whisper_language:
                    kwargs["language"] = self.whisper_language
                with torch.inference_mode():
                    ids = model.generate(features, **kwargs)
                pieces.append(processor.batch_decode(ids, skip_special_tokens=True)[0])
        return clean(" ".join(pieces))

    def _load(self):
        if self._bundle is None:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            processor = AutoProcessor.from_pretrained(self.model_dir)
            model = AutoModelForSpeechSeq2Seq.from_pretrained(self.model_dir, dtype=dtype).to(device).eval()
            self._bundle = (model, processor)
        return self._bundle
