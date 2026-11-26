import re
import threading
import unicodedata
from pathlib import Path

from app.checkpoints import METADATA, read_metadata
from app.textnorm import from_model, to_model

MAX_TOKENS = 512
MAX_CHARS = 400
CHUNK_MAX_CHARS = 480
_PARA_BREAK = re.compile(r"\n\s*\n+")
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
__all__ = [
    "BASE_TAGS",
    "INIT_FROM",
    "METADATA",
    "NEW_TAGS",
    "NllbEngine",
    "group_sentences",
    "init_new_tag_embeddings",
    "read_metadata",
    "split_sentences",
    "tag_for",
]


def tag_for(code: str) -> str:
    return BASE_TAGS.get(code) or NEW_TAGS[code]


def init_new_tag_embeddings(model, tokenizer) -> list[int]:
    """Initialize new ethnic language tag rows from related NLLB language embeddings."""
    import torch

    weight = model.get_input_embeddings().weight
    ids: list[int] = []
    with torch.no_grad():
        for tag, sources in INIT_FROM.items():
            tag_id = tokenizer.convert_tokens_to_ids(tag)
            source_ids = [tokenizer.convert_tokens_to_ids(s) for s in sources]
            weight[tag_id] = weight[source_ids].float().mean(dim=0).to(weight.dtype)
            ids.append(tag_id)
    return ids


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


def group_sentences(sentences: list[str], max_chars: int = CHUNK_MAX_CHARS) -> list[str]:
    """Merge consecutive short sentences up to max_chars so nearby context stays intact."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sent in sentences:
        s = sent.strip()
        if not s:
            continue
        added = len(s) + (1 if current else 0)
        if current and current_len + added > max_chars:
            chunks.append(" ".join(current))
            current = [s]
            current_len = len(s)
        else:
            current.append(s)
            current_len += added
    if current:
        chunks.append(" ".join(current))
    return chunks


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
        self._is_peft = bool(self.adapter and (Path(self.adapter) / "adapter_config.json").is_file())
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
        if not scores or scores.get("chrf", 0.0) <= scores.get("copy_chrf", 0.0):
            return False
        ratio = scores.get("length_ratio")
        if ratio is not None and not (0.55 <= ratio <= 1.6):
            return False
        rep = scores.get("repetition_rate")
        return not (rep is not None and rep >= 0.25)

    def step_name(self, source: str, target: str) -> str:
        return "nllb-ethnic" if source in NEW_TAGS or target in NEW_TAGS else "nllb"

    def translate(self, text: str, source: str, target: str) -> str:
        paragraphs = [p for p in _PARA_BREAK.split(text.strip()) if p.strip()] or [text]
        para_chunks = [group_sentences(split_sentences(p)) for p in paragraphs]
        flat = [chunk for chunks in para_chunks for chunk in chunks]
        if not flat:
            return ""
        translated = self.translate_batch(flat, source, target)
        out_paras = []
        idx = 0
        for chunks in para_chunks:
            count = len(chunks)
            out_paras.append(" ".join(translated[idx : idx + count]))
            idx += count
        return "\n\n".join(out_paras)

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
            budget = min(MAX_TOKENS, max(32, int(2.2 * inputs["input_ids"].shape[1]) + 24))
            kwargs = {
                "forced_bos_token_id": forced,
                "num_beams": self.beams,
                "max_new_tokens": budget,
                "length_penalty": 1.1,
                "no_repeat_ngram_size": 3,
                "repetition_penalty": 1.1,
                "early_stopping": True,
            }
            with torch.inference_mode():
                if self._is_peft and self.step_name(source, target) == "nllb":
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
            tok_source = (
                self.adapter
                if self.adapter and (Path(self.adapter) / "tokenizer_config.json").is_file()
                else self.base
            )
            tokenizer = AutoTokenizer.from_pretrained(tok_source)
            # low_cpu_mem_usage + fp16 on CUDA keeps the 600M weights off the pagefile.
            if self._is_peft:
                from peft import PeftModel

                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.base,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                )
                model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
                init_new_tag_embeddings(model, tokenizer)
                model = PeftModel.from_pretrained(model, self.adapter)
            elif self.adapter and (Path(self.adapter) / "config.json").is_file():
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.adapter,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                )
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.base,
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True,
                )
            if device == "cuda":
                model.to(device)
            else:
                model = model.to(device)
            model.eval()
            self._bundle = (model, tokenizer)
        return self._bundle
