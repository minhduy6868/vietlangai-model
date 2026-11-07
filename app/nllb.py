import re
import threading
import unicodedata

from app.checkpoints import METADATA, read_metadata
from app.textnorm import from_model, to_model

MAX_TOKENS = 200
MAX_CHARS = 400
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s+")

BASE_TAGS = {"vi": "vie_Latn", "en": "eng_Latn"}
NEW_TAGS = {
    "tyz": "tyz_Latn",
    "rad": "rad_Latn",
    "jra": "jra_Latn",
    "cjm": "cjm_Latn",
    "hmn": "hmn_Latn",
}
# Embedding rows averaged to start each new tag. Acehnese is the only Chamic
# language in NLLB-200; Tày borrows Vietnamese spelling and its Tai relatives.
INIT_FROM = {
    "tyz_Latn": ("vie_Latn", "tha_Thai", "lao_Laoo"),
    "rad_Latn": ("ace_Latn", "vie_Latn"),
    "jra_Latn": ("ace_Latn", "ind_Latn"),
    "cjm_Latn": ("ace_Latn", "zsm_Latn"),
    "hmn_Latn": ("vie_Latn", "eng_Latn"),
}
__all__ = ["BASE_TAGS", "INIT_FROM", "METADATA", "NEW_TAGS", "NllbEngine", "read_metadata", "tag_for"]


def tag_for(code: str) -> str:
    return BASE_TAGS.get(code) or NEW_TAGS[code]


def split_sentences(text: str) -> list[str]:
    """Translate sentence by sentence so long input is not cut at MAX_TOKENS.

    A sentence longer than MAX_CHARS is split again at commas and semicolons.
    """
    parts = []
    for sentence in _SENTENCE_END.split(text.strip()):
        if len(sentence) > MAX_CHARS:
            parts.extend(_CLAUSE_END.split(sentence))
        else:
            parts.append(sentence)
    return [part for part in parts if part] or [text]


class NllbEngine:
    """NLLB-200 base for vi <-> en, plus an optional LoRA adapter for ethnic <-> vi.

    The adapter is switched off for vi <-> en so the base model keeps its quality.
    Ethnic pairs come from the checkpoint metadata (the adapter folder, or the
    base folder for a fully fine-tuned model). A direction is served only when
    training/evaluate.py measured it on the test split and it beat copying the
    source; `gated=False` lists every trained direction, for evaluation.
    """

    name = "nllb"

    def __init__(self, base: str, adapter: str | None = None, beams: int = 4, gated: bool = True) -> None:
        self.base = base
        self.adapter = adapter
        self.beams = beams
        meta = read_metadata(adapter or base)
        self.languages: list[str] = meta.get("languages", [])
        self.evaluation: dict = meta.get("evaluation", {})
        self.gated = gated
        self._bundle = None
        self._lock = threading.Lock()

    def pairs(self) -> set[tuple[str, str]]:
        pairs = {("vi", "en"), ("en", "vi")}
        for code in self.languages:
            for direction in ((code, "vi"), ("vi", code)):
                if not self.gated or self.passed(*direction):
                    pairs.add(direction)
        return pairs

    def passed(self, source: str, target: str) -> bool:
        scores = self.evaluation.get(f"{source}->{target}")
        return bool(scores) and scores["chrf"] > scores["copy_chrf"]

    def step_name(self, source: str, target: str) -> str:
        return "nllb-ethnic" if source in NEW_TAGS or target in NEW_TAGS else "nllb"

    def translate(self, text: str, source: str, target: str) -> str:
        sentences = split_sentences(text)
        return " ".join(self.translate_batch(sentences, source, target))

    def translate_batch(self, texts: list[str], source: str, target: str) -> list[str]:
        import torch

        # The tokenizer's src_lang and the adapter switch are shared state.
        with self._lock:
            model, tokenizer = self._load()
            tokenizer.src_lang = tag_for(source)
            inputs = tokenizer(
                [to_model(text) for text in texts],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_TOKENS,
            ).to(model.device)
            forced = tokenizer.convert_tokens_to_ids(tag_for(target))
            budget = min(MAX_TOKENS, 3 * inputs["input_ids"].shape[1] + 10)
            kwargs = {"forced_bos_token_id": forced, "num_beams": self.beams, "max_new_tokens": budget}
            with torch.inference_mode():
                if self.adapter and self.step_name(source, target) == "nllb":
                    with model.disable_adapter():
                        output = model.generate(**inputs, **kwargs)
                else:
                    output = model.generate(**inputs, **kwargs)
            decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
        if target in NEW_TAGS:
            return [from_model(text) for text in decoded]
        return [unicodedata.normalize("NFC", text) for text in decoded]

    def _load(self):
        if self._bundle is None:
            import torch
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            tokenizer = AutoTokenizer.from_pretrained(self.adapter or self.base)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.base, dtype=dtype)
            if self.adapter:
                from peft import PeftModel

                model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
                model = PeftModel.from_pretrained(model, self.adapter)
            model.to(device).eval()
            self._bundle = (model, tokenizer)
        return self._bundle
