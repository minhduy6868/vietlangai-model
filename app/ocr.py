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


def order_ocr_lines(results) -> list[str]:
    """Group OCR boxes into reading lines using each box's adaptive height."""
    items = []
    for box, text, conf in results:
        cleaned = text.strip()
        if not cleaned:
            continue
        ys = [float(p[1]) for p in box]
        xs = [float(p[0]) for p in box]
        y_top, y_bottom = min(ys), max(ys)
        items.append(((y_top + y_bottom) / 2.0, max(1.0, y_bottom - y_top), min(xs), cleaned))
    if not items:
        return []
    items.sort(key=lambda item: (item[0], item[2]))

    lines: list[list[tuple[float, str]]] = []
    line_y = items[0][0]
    line_h = items[0][1]
    current: list[tuple[float, str]] = []
    for y_center, height, x_left, text in items:
        if current and abs(y_center - line_y) > 0.6 * max(line_h, height):
            current.sort(key=lambda word: word[0])
            lines.append(current)
            current = [(x_left, text)]
            line_y = y_center
            line_h = height
        else:
            count = len(current)
            line_y = ((line_y * count) + y_center) / (count + 1)
            line_h = max(line_h, height)
            current.append((x_left, text))
    if current:
        current.sort(key=lambda word: word[0])
        lines.append(current)
    return [" ".join(word for _, word in line) for line in lines]


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
        lines = order_ocr_lines(results)
        if not lines:
            return "", 0.0
        non_empty = [item for item in results if item[1].strip()]
        text = " ".join(lines)
        confidence = sum(float(item[2]) for item in non_empty) / len(non_empty)
        return text, round(confidence, 3)

    def _load(self):
        if self._reader is None:
            import easyocr
            import torch

            self._reader = easyocr.Reader(self.languages, gpu=torch.cuda.is_available(), verbose=False)
        return self._reader
