from pathlib import Path

from app.translate import _adapter_ready


def test_adapter_ready_requires_peft_config(tmp_path: Path):
    assert _adapter_ready(None) is None
    assert _adapter_ready(str(tmp_path)) is None
    (tmp_path / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert _adapter_ready(str(tmp_path)) == str(tmp_path)
