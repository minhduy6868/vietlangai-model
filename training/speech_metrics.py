import re
import unicodedata

_PUNCT = re.compile(r"[^\w\s']", re.UNICODE)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    return " ".join(_PUNCT.sub(" ", text).split())


def edit_distance(a: list, b: list) -> int:
    previous = list(range(len(b) + 1))
    for i, item in enumerate(a, start=1):
        current = [i]
        for j, other in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (item != other)))
        previous = current
    return previous[-1]


def error_rates(hypotheses: list[str], references: list[str]) -> dict:
    word_errors = word_total = char_errors = char_total = 0
    for hyp, ref in zip(hypotheses, references):
        hyp, ref = normalize(hyp), normalize(ref)
        word_errors += edit_distance(hyp.split(), ref.split())
        word_total += len(ref.split())
        char_errors += edit_distance(list(hyp), list(ref))
        char_total += len(ref)
    return {
        "wer": round(word_errors / max(word_total, 1), 4),
        "cer": round(char_errors / max(char_total, 1), 4),
    }
