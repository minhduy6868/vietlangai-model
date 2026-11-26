"""End-to-end Kaggle GPU automation for Ngon Sac Viet ethnic MT training.

Usage:
    python scripts/kaggle_auto_train.py
    python scripts/kaggle_auto_train.py --dry-run
    python scripts/kaggle_auto_train.py --install-zip path/to/nllb-ethnic-ready.zip
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.checkpoints import METADATA, read_metadata
from app.translate import _adapter_ready

NOTEBOOK = ROOT / "notebooks" / "kaggle" / "train_nllb_ethnic.ipynb"
TEMPLATE_META = ROOT / "notebooks" / "kaggle" / "train-kernel-metadata.json"
DEFAULT_OUTPUT_DIR = ROOT / "models" / "nllb-ethnic"
DEFAULT_REPORT_PATH = (
    (ROOT.parent / "reports" / "eval" / "nllb-ethnic.json")
    if (ROOT.parent / "reports").is_dir()
    else (ROOT / "reports" / "eval" / "nllb-ethnic.json")
)


def _credential_paths() -> list[Path]:
    home = Path.home()
    return [
        home / ".kaggle" / "kaggle.json",
        home / ".config" / "kaggle" / "kaggle.json",
    ]


def _token_paths() -> list[Path]:
    home = Path.home()
    return [
        home / ".kaggle" / "access_token",
        home / ".config" / "kaggle" / "access_token",
    ]


def resolve_kaggle_username(explicit: str | None = None) -> str | None:
    if explicit and explicit.strip():
        return explicit.strip()
    env_user = os.getenv("KAGGLE_USERNAME", "").strip()
    if env_user:
        return env_user
    for path in _credential_paths():
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                user = (data.get("username") or "").strip()
                if user:
                    return user
            except (OSError, json.JSONDecodeError):
                continue
    if has_kaggle_credentials():
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()
            user = (api.get_config_value("username") or "").strip()
            if user:
                return user
        except (ImportError, OSError, ValueError, RuntimeError, KeyError):
            return None
    return None


def has_kaggle_credentials() -> bool:
    if any(p.is_file() for p in _credential_paths() + _token_paths()):
        return True
    if os.getenv("KAGGLE_API_TOKEN", "").strip():
        return True
    has_user = bool(os.getenv("KAGGLE_USERNAME", "").strip())
    has_key = bool(os.getenv("KAGGLE_KEY", "").strip())
    return has_user and has_key


def require_kaggle_cli() -> str:
    kaggle_bin = shutil.which("kaggle")
    if not kaggle_bin:
        raise SystemExit(
            "Chưa cài Kaggle CLI. Cài bằng:\n"
            "  pip install kaggle\n"
            "Sau đó tải kaggle.json từ Kaggle Account -> API -> Create New Token "
            f"và lưu vào: {Path.home() / '.kaggle' / 'kaggle.json'}"
        )
    if not has_kaggle_credentials():
        raise SystemExit(
            "Chưa có thông tin đăng nhập Kaggle.\n"
            "Vào Kaggle -> Settings -> API -> Create New Token để tải kaggle.json, "
            f"đặt tại {Path.home() / '.kaggle' / 'kaggle.json'} (hoặc đặt KAGGLE_USERNAME + KAGGLE_API_TOKEN)."
        )
    return kaggle_bin


def build_staging(staging_dir: Path, kernel_id: str, gpu: str) -> Path:
    if not NOTEBOOK.is_file():
        raise FileNotFoundError(f"Thiếu notebook: {NOTEBOOK}")
    if not TEMPLATE_META.is_file():
        raise FileNotFoundError(f"Thiếu metadata mẫu: {TEMPLATE_META}")

    staging_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(NOTEBOOK, staging_dir / NOTEBOOK.name)
    meta = json.loads(TEMPLATE_META.read_text(encoding="utf-8"))
    meta["id"] = kernel_id
    meta["machine_shape"] = gpu
    meta_path = staging_dir / "kernel-metadata.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta_path


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    if proc.returncode != 0:
        raise SystemExit(f"Lệnh thất bại ({proc.returncode}): {' '.join(cmd)}\n{output.strip()}")
    return output.strip()


def push_kernel(kaggle_bin: str, staging_dir: Path) -> None:
    print(f"Đang đẩy notebook lên Kaggle từ {staging_dir}...")
    out = _run([kaggle_bin, "kernels", "push", "-p", str(staging_dir)])
    print(out)


def wait_kernel(kaggle_bin: str, kernel_id: str, poll_seconds: int) -> None:
    print(f"Đang chờ Kaggle chạy {kernel_id} (kiểm tra mỗi {poll_seconds}s)...")
    while True:
        status_out = _run([kaggle_bin, "kernels", "status", kernel_id])
        lower = status_out.lower()
        print(f"[{time.strftime('%H:%M:%S')}] {status_out}")
        if "complete" in lower:
            return
        if "error" in lower or "cancel" in lower:
            raise SystemExit(f"Kaggle kernel dừng với trạng thái lỗi:\n{status_out}")
        time.sleep(poll_seconds)


def _find_checkpoint_root(extract_dir: Path) -> Path:
    markers = ("adapter_config.json", "config.json", METADATA)
    for marker in markers:
        if (extract_dir / marker).is_file():
            return extract_dir
    for child in sorted(extract_dir.iterdir()):
        if child.is_dir() and any((child / marker).is_file() for marker in markers):
            return child
    raise SystemExit(f"Không tìm thấy checkpoint hợp lệ bên trong {extract_dir}")


def install_bundle(
    zip_path: Path,
    target_dir: Path = DEFAULT_OUTPUT_DIR,
    report_src: Path | None = None,
    report_dest: Path = DEFAULT_REPORT_PATH,
) -> dict:
    if not zip_path.is_file():
        raise SystemExit(f"Không tìm thấy file zip: {zip_path}")

    with tempfile.TemporaryDirectory(prefix="nsv_install_") as tmp:
        extract_dir = Path(tmp) / "extracted"
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        src_root = _find_checkpoint_root(extract_dir)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_root, target_dir)

    if report_src and report_src.is_file():
        report_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report_src, report_dest)

    ready = _adapter_ready(str(target_dir))
    if not ready:
        raise SystemExit(
            f"Đã giải nén vào {target_dir} nhưng thiếu weights (adapter_config.json hoặc config.json + weights)."
        )
    meta = read_metadata(target_dir)
    print(f"Đã cài checkpoint thành công vào: {target_dir}")
    print(json.dumps(meta.get("evaluation", {}), ensure_ascii=False, indent=2))
    return meta


def download_and_install(
    kaggle_bin: str,
    kernel_id: str,
    target_dir: Path,
    report_dest: Path,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="nsv_kout_") as tmp:
        out_dir = Path(tmp)
        print(f"Đang tải output từ {kernel_id}...")
        _run([kaggle_bin, "kernels", "output", kernel_id, "-p", str(out_dir)])
        zip_path = out_dir / "nllb-ethnic-ready.zip"
        if not zip_path.is_file():
            matches = list(out_dir.rglob("*.zip"))
            if not matches:
                raise SystemExit(f"Không thấy nllb-ethnic-ready.zip trong output của {kernel_id}.")
            zip_path = matches[0]
        report_src = out_dir / "eval_report.json"
        return install_bundle(zip_path, target_dir, report_src=report_src, report_dest=report_dest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tự động train NLLB Việt <-> Dân tộc trên Kaggle GPU")
    parser.add_argument("--username", help="Kaggle username (mặc định đọc từ kaggle.json)")
    parser.add_argument("--slug", default="ngonsacviet-train-nllb-ethnic-long-sentences")
    parser.add_argument("--gpu", default="NvidiaTeslaT4", choices=["NvidiaTeslaT4", "NvidiaTeslaP100"])
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--push-only", action="store_true", help="Chỉ đẩy notebook lên Kaggle, không chờ")
    parser.add_argument("--wait-only", action="store_true", help="Chỉ chờ kernel đang chạy và tải về")
    parser.add_argument("--install-zip", type=Path, help="Giải nén và kích hoạt file zip đã tải tay")
    parser.add_argument("--target", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Kiểm tra tạo staging không gọi Kaggle")
    args = parser.parse_args()

    if args.install_zip:
        install_bundle(args.install_zip, target_dir=args.target, report_dest=args.report)
        return

    username = resolve_kaggle_username(args.username) or "demo-user"
    kernel_id = f"{username}/{args.slug}"

    if args.dry_run:
        with tempfile.TemporaryDirectory(prefix="nsv_stage_") as tmp:
            meta_path = build_staging(Path(tmp), kernel_id, args.gpu)
            print(f"Dry-run OK: {meta_path}")
            print(meta_path.read_text(encoding="utf-8"))
        return

    kaggle_bin = require_kaggle_cli()
    if not resolve_kaggle_username(args.username):
        raise SystemExit("Không xác định được Kaggle username. Truyền --username <ten> hoặc cấu hình kaggle.json.")

    if not args.wait_only:
        with tempfile.TemporaryDirectory(prefix="nsv_stage_") as tmp:
            staging = Path(tmp)
            build_staging(staging, kernel_id, args.gpu)
            push_kernel(kaggle_bin, staging)

    if args.push_only:
        print(f"Đã đẩy lên Kaggle: https://www.kaggle.com/code/{kernel_id}")
        return

    wait_kernel(kaggle_bin, kernel_id, args.poll_seconds)
    download_and_install(kaggle_bin, kernel_id, args.target, args.report)


if __name__ == "__main__":
    main()
