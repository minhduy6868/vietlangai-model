import json
from pathlib import Path

METADATA = "ngonsacviet.json"


def read_metadata(folder: str | Path) -> dict:
    path = Path(folder) / METADATA
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_metadata(folder: str | Path, data: dict) -> None:
    (Path(folder) / METADATA).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
