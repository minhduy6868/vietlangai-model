import io

SAMPLE_RATE = 16000


class AudioDecodeError(Exception):
    pass


def decode_audio(data: bytes, rate: int = SAMPLE_RATE, max_seconds: float | None = None):
    """Decode any container PyAV/FFmpeg reads (wav, m4a, aac, ogg, webm, mp3) to mono float32.

    With max_seconds, decoding stops once that much audio is collected and the
    result is one sample longer than the cap, so callers can tell it was cut.
    """
    import av
    import numpy as np

    limit = int(max_seconds * rate) + 1 if max_seconds else None
    chunks = []
    total = 0
    try:
        with av.open(io.BytesIO(data)) as container:
            stream = next((s for s in container.streams if s.type == "audio"), None)
            if stream is None:
                raise AudioDecodeError("Tệp không có luồng âm thanh.")
            resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
            for frame in container.decode(stream):
                for out in resampler.resample(frame):
                    chunk = out.to_ndarray().reshape(-1)
                    chunks.append(chunk)
                    total += len(chunk)
                if limit and total >= limit:
                    break
            else:
                for out in resampler.resample(None):
                    chunks.append(out.to_ndarray().reshape(-1))
    except AudioDecodeError:
        raise
    except Exception as exc:
        raise AudioDecodeError("Không đọc được tệp âm thanh.") from exc
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    samples = np.concatenate(chunks).astype(np.float32)
    return samples[:limit] if limit else samples
