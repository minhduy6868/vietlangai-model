import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.catalog import ethnic_codes, language_by_code
from app.detect import detect_language, is_cham_script
from app.nllb import NllbEngine
from app.paths import default_nllb_base, public_pairs_path, reviewed_pairs_path

_PARA_BREAK = re.compile(r"\n\s*\n+")
_EDGE_PUNCT = ".!?…,:;\"'()[]{}"


class RouteNotReady(Exception):
    def __init__(self, source: str, target: str) -> None:
        self.source = source
        self.target = target
        super().__init__(f"No reviewed translator for {source} -> {target}")


class UnknownLanguage(Exception):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def _strip_edge_punct(text: str) -> str:
    return normalize(text).strip(_EDGE_PUNCT).strip()


def _clean_input(text: str) -> str:
    paragraphs = []
    for block in _PARA_BREAK.split(text.strip()):
        normalized = " ".join(block.split())
        if normalized:
            paragraphs.append(normalized)
    return "\n\n".join(paragraphs)


@dataclass
class Glossary:
    pairs: dict[tuple[str, str], dict[str, str]] = field(default_factory=dict)

    def direct_pairs(self) -> set[tuple[str, str]]:
        return {pair for pair, table in self.pairs.items() if table}

    def translate(self, text: str, source: str, target: str) -> str | None:
        table = self.pairs.get((source, target), {})
        if not table:
            return None
        key = normalize(text)
        hit = table.get(key)
        if hit is not None:
            return hit
        stripped = _strip_edge_punct(key)
        return table.get(stripped) if stripped else None


def builtin_glossary() -> Glossary:
    return Glossary(
        {
            ("vi", "en"): {
                "xin chào": "Hello",
                "cảm ơn": "Thank you",
                "tạm biệt": "Goodbye",
            },
            ("en", "vi"): {
                "hello": "Xin chào",
                "thank you": "Cảm ơn",
                "goodbye": "Tạm biệt",
            },
        }
    )


def _add_pair(glossary: Glossary, source_lang: str, target_lang: str, source: str, target: str) -> None:
    source_lang = source_lang.strip()
    target_lang = target_lang.strip()
    source_text = source.strip()
    target_text = target.strip()
    if not source_lang or not target_lang or not source_text or not target_text:
        return
    pair = (source_lang, target_lang)
    table = glossary.pairs.setdefault(pair, {})
    norm_src = normalize(source_text)
    table[norm_src] = target_text
    stripped_src = _strip_edge_punct(norm_src)
    if stripped_src and stripped_src != norm_src:
        table.setdefault(stripped_src, target_text)
    # Exact-match glossaries are directional; mirror ethnic ↔ vi for lookup both ways.
    ethnic = set(ethnic_codes())
    if {source_lang, target_lang} & ethnic and "vi" in {source_lang, target_lang}:
        reverse = (target_lang, source_lang)
        rev_table = glossary.pairs.setdefault(reverse, {})
        norm_tgt = normalize(target_text)
        rev_table[norm_tgt] = source_text
        stripped_tgt = _strip_edge_punct(norm_tgt)
        if stripped_tgt and stripped_tgt != norm_tgt:
            rev_table.setdefault(stripped_tgt, source_text)


def _load_pairs_file(glossary: Glossary, path) -> None:
    if not path.is_file():
        return
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            verified = (row.get("verified") or "").strip().lower()
            if verified not in {"1", "true", "yes"}:
                continue
            _add_pair(
                glossary,
                row.get("source_lang", ""),
                row.get("target_lang", ""),
                row.get("source", ""),
                row.get("target", ""),
            )


def load_glossary() -> Glossary:
    """Load builtin, DB-exported reviewed pairs, then verified public bitext."""
    glossary = builtin_glossary()
    _load_pairs_file(glossary, reviewed_pairs_path())
    _load_pairs_file(glossary, public_pairs_path())
    return glossary


def plan_route(source: str, target: str, direct: set[tuple[str, str]]) -> list[tuple[str, str]]:
    if source == target:
        return []
    if (source, target) in direct:
        return [(source, target)]
    if (
        source != "vi"
        and target != "vi"
        and (source, "vi") in direct
        and ("vi", target) in direct
    ):
        return [(source, "vi"), ("vi", target)]
    raise RouteNotReady(source, target)


class NeuralEngine:
    """Optional mT5 checkpoint for ethnic <-> vi. The glossary stays in front of it."""

    name = "mt5"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._bundle = None

    def pairs(self) -> set[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for code in ethnic_codes():
            pairs.add((code, "vi"))
            pairs.add(("vi", code))
        return pairs

    def step_name(self, source: str, target: str) -> str:
        return self.name

    def translate(self, text: str, source: str, target: str) -> str:
        model, tokenizer = self._load()
        prefixed = f"{source} {target}: {text}"
        inputs = tokenizer(prefixed, return_tensors="pt", truncation=True, max_length=128)
        output = model.generate(**inputs, max_new_tokens=128)
        return tokenizer.batch_decode(output, skip_special_tokens=True)[0]

    def _load(self):
        if self._bundle is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._bundle = (model, tokenizer)
        return self._bundle


UNREVIEWED_WARNING = "Bản dịch máy cho tiếng dân tộc, chưa được người bản ngữ rà soát."


class Translator:
    def __init__(self, glossary: Glossary, neural: NeuralEngine | NllbEngine | None = None) -> None:
        self.glossary = glossary
        self.neural = neural
        self.codes = set(language_by_code())

    def direct_pairs(self) -> set[tuple[str, str]]:
        pairs = set(self.glossary.direct_pairs())
        if self.neural is not None:
            pairs |= self.neural.pairs()
        return pairs

    def translate(self, text: str, source: str, target: str) -> dict:
        cleaned = _clean_input(text)
        if not cleaned:
            raise ValueError("empty")
        detected = None
        if source == "auto":
            found = detect_language(cleaned)
            detected = found["language"]
            if detected == "und":
                raise RouteNotReady("und", target)
            source = detected
        if source not in self.codes or target not in self.codes:
            missing = source if source not in self.codes else target
            raise UnknownLanguage(missing)
        if any(is_cham_script(char) for char in cleaned):
            # Every engine here reads Latin script; Akhar Thrah needs its own model.
            raise RouteNotReady(source, target)
        planned = plan_route(source, target, self.direct_pairs())
        current = cleaned
        steps = []
        warnings: list[str] = []
        ethnic = set(ethnic_codes())
        for step_source, step_target in planned:
            hit = self.glossary.translate(current, step_source, step_target)
            if hit is not None:
                current = hit
                engine = "glossary"
            elif self.neural is not None and (step_source, step_target) in self.neural.pairs():
                current = self.neural.translate(current, step_source, step_target)
                engine = self.neural.step_name(step_source, step_target)
                if {step_source, step_target} & ethnic and UNREVIEWED_WARNING not in warnings:
                    warnings.append(UNREVIEWED_WARNING)
            else:
                raise RouteNotReady(step_source, step_target)
            steps.append(
                {
                    "source": step_source,
                    "target": step_target,
                    "engine": engine,
                    "text": current,
                }
            )
        route = "direct" if len(planned) == 1 else "pivot"
        if not planned:
            route = "identity"
        engines = [step["engine"] for step in steps]
        return {
            "text": current,
            "source_lang": source,
            "target_lang": target,
            "detected_lang": detected,
            "route": route,
            "engine": "+".join(engines) if engines else "identity",
            "warnings": warnings,
            "steps": steps,
        }


def _adapter_ready(path: str | None) -> str | None:
    """Use the path when PEFT adapter or full seq2seq weights are on disk."""
    if not path:
        return None
    folder = Path(path)
    if (folder / "adapter_config.json").is_file():
        return path
    has_weights = (folder / "model.safetensors").is_file() or (folder / "pytorch_model.bin").is_file()
    if (folder / "config.json").is_file() and has_weights:
        return path
    return None


def build_translator() -> Translator:
    import os

    kind = os.getenv("ML_MT_ENGINE", "glossary")
    neural = None
    if kind == "mt5":
        neural = NeuralEngine(os.getenv("ML_MT_MODEL") or "google/mt5-small")
    elif kind == "nllb":
        beams = max(1, int(os.getenv("ML_NLLB_BEAMS", "4")))
        neural = NllbEngine(
            os.getenv("ML_MT_MODEL") or default_nllb_base(),
            _adapter_ready(os.getenv("ML_MT_ADAPTER") or None),
            beams=beams,
        )
    elif kind != "glossary":
        raise ValueError(f"Unknown ML_MT_ENGINE: {kind}")
    return Translator(load_glossary(), neural)
