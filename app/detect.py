import re

from app.langid import load_langid

MIN_CONFIDENCE = 0.9
# Below this the per-n-gram fit is too noisy for the character model to answer.
MIN_LETTERS = 8

_VI_WORDS = {
    "và",
    "của",
    "là",
    "những",
    "không",
    "các",
    "một",
    "trong",
    "cho",
    "được",
    "này",
    "với",
    "xin",
    "chào",
}
_EN_WORDS = {"the", "and", "hello", "thank", "you", "goodbye", "good", "morning"}
_TOKEN = re.compile(r"\w+", re.UNICODE)


def is_cham_script(char: str) -> bool:
    return "\uAA00" <= char <= "\uAA5F"


def detect_language(text: str) -> dict:
    """Guess the language of a short text.

    Cham script is decided from characters. When models/langid.json exists, a
    character n-gram model separates the trained Latin-script languages and
    answers only above MIN_CONFIDENCE. Otherwise, or below that, the lexicon
    rule answers vi/en and refuses to guess ethnic Latin text.
    """
    if any(is_cham_script(char) for char in text):
        return {"language": "cjm", "confidence": 0.99, "method": "script"}
    tokens = [token.casefold() for token in _TOKEN.findall(text)]
    if not tokens:
        return {"language": "und", "confidence": 0.0, "method": "empty"}
    vi_hits = sum(token in _VI_WORDS for token in tokens)
    en_hits = sum(token in _EN_WORDS for token in tokens)
    model = load_langid()
    long_enough = len(tokens) >= 2 and sum(len(token) for token in tokens) >= MIN_LETTERS
    if model is not None and long_enough and not (en_hits and en_hits > vi_hits):
        language, confidence = model.predict(text)
        if confidence >= MIN_CONFIDENCE:
            return {"language": language, "confidence": round(confidence, 3), "method": "char_ngram"}
    if vi_hits >= 1 and vi_hits >= en_hits:
        return {
            "language": "vi",
            "confidence": min(0.55 + 0.1 * vi_hits, 0.9),
            "method": "lexicon",
        }
    if en_hits >= 1 and en_hits > vi_hits:
        return {"language": "en", "confidence": 0.6, "method": "lexicon"}
    return {"language": "und", "confidence": 0.2, "method": "unclassified"}
