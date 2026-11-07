import threading

LATIN_OCR_WARNING = "Chữ nhận từ ảnh bằng OCR Latin; các chữ ƀ, ĕ, ŏ, ŭ của Ê Đê và Jrai có thể bị đọc sai."
MAX_PIXELS = 4096 * 4096
MAX_SIDE = 2048


class ImageRejected(Exception):
    pass


def load_image(image: bytes):
    """Decode with a pixel cap and shrink large photos before OCR."""
    import io

    import numpy as np
    from PIL import Image, ImageOps

    try:
        with Image.open(io.BytesIO(image)) as picture:
            if picture.width * picture.height > MAX_PIXELS:
                raise ImageRejected("Ảnh quá lớn.")
            picture = ImageOps.exif_transpose(picture).convert("RGB")
            picture.thumbnail((MAX_SIDE, MAX_SIDE))
            # EasyOCR reads arrays as OpenCV BGR.
            return np.ascontiguousarray(np.asarray(picture)[..., ::-1])
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ImageRejected("Không đọc được tệp ảnh.") from exc


class EasyOcrEngine:
    """Server fallback for Latin-script photos when the phone has no on-device OCR.

    Uses EasyOCR's Vietnamese model, which covers the Quốc ngữ letters shared by
    Tày, H'Mông and the Chamic Latin orthographies. Akhar Thrah is not read here.
    """

    name = "easyocr"

    def __init__(self, languages: tuple[str, ...] = ("vi", "en")) -> None:
        self.languages = list(languages)
        self._reader = None
        self._lock = threading.Lock()

    def read(self, image) -> tuple[str, float]:
        with self._lock:
            results = self._load().readtext(image, detail=1, paragraph=False)
        if not results:
            return "", 0.0
        lines = sorted(results, key=lambda item: (round(min(p[1] for p in item[0]) / 20), min(p[0] for p in item[0])))
        text = " ".join(item[1].strip() for item in lines if item[1].strip())
        confidence = sum(float(item[2]) for item in lines) / len(lines)
        return text, round(confidence, 3)

    def _load(self):
        if self._reader is None:
            import easyocr
            import torch

            self._reader = easyocr.Reader(self.languages, gpu=torch.cuda.is_available(), verbose=False)
        return self._reader
