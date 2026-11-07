import os
from pathlib import Path


def catalog_path() -> Path:
    override = os.environ.get("LANGUAGES_CATALOG")
    if override:
        return Path(override)
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "shared" / "languages.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("shared/languages.json")


def models_dir() -> Path:
    return Path(os.environ.get("ML_MODELS_DIR") or Path(__file__).resolve().parents[1] / "models")


def default_nllb_base() -> str:
    local = models_dir() / "nllb-200-distilled-600M"
    return str(local) if (local / "model.safetensors").is_file() else "facebook/nllb-200-distilled-600M"


def reviewed_pairs_path() -> Path:
    override = os.environ.get("REVIEWED_PAIRS")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / "data" / "reviewed_pairs.csv"
