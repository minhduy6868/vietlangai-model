import os
from pathlib import Path


def _load_dotenv() -> None:
    """Load ai-model/.env into os.environ without adding a dependency."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


_load_dotenv()


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


def public_pairs_path() -> Path:
    override = os.environ.get("PUBLIC_PAIRS")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[1] / "data" / "public" / "pairs.csv"
