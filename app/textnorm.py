import unicodedata

_PUNCT = str.maketrans(
    {
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u02bc": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\u00a0": " ",
    }
)

# NLLB has no piece for these letters, and its tokenizer recomposes letter +
# combining breve back into the missing capital. Each one borrows a letter that
# NLLB knows and the five languages never use.
_MODEL_IN = str.maketrans({"ƀ": "ɓ", "Ƀ": "Ɓ", "Ĕ": "Ë", "Ŏ": "Ő", "Ĭ": "Ï", "Ŭ": "Ü"})
_MODEL_OUT = str.maketrans({"ɓ": "ƀ", "Ɓ": "Ƀ", "Ë": "Ĕ", "Ő": "Ŏ", "Ï": "Ĭ", "Ü": "Ŭ"})


def clean(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_PUNCT)
    return " ".join(text.split())


def to_model(text: str) -> str:
    return clean(text).translate(_MODEL_IN)


def from_model(text: str) -> str:
    return unicodedata.normalize("NFC", text.translate(_MODEL_OUT))
