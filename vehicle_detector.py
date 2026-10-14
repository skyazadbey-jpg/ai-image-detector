"""Optional dominant-car evidence, using pinned YOLOX-Nano COCO weights.

No images leave the server. Model download, if needed, uses a fixed HTTPS URL
and a pinned SHA-256. This does not detect fraud, damage or vehicle identity.
"""
import hashlib
import io
from pathlib import Path
import threading
import time

ROOT = Path(__file__).resolve().parent
MODEL_SHA256 = 'c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d'
MODEL_URL = 'https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx'
_session = None
_lock = threading.Lock()
_retry_after = 0.0


def model_path():
    global _retry_after
    path = ROOT / 'yolox_nano.onnx'
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == MODEL_SHA256:
        return path
    if time.monotonic() < _retry_after:
        raise RuntimeError('Vehicle model temporarily unavailable')
    import requests
    temporary = path.with_suffix('.download')
    try:
        size = 0
        digest = hashlib.sha256()
        started = time.monotonic()
        with requests.get(MODEL_URL, stream=True, timeout=(8, 20)) as response:
            response.raise_for_status()
            with temporary.open('wb') as handle:
                for chunk in response.iter_content(64 * 1024):
                    if time.monotonic() - started > 30:
                        raise TimeoutError('Vehicle model download timed out')
                    size += len(chunk)
                    if size > 5 * 1024 * 1024:
                        raise ValueError('Vehicle model exceeds expected size')
                    digest.update(chunk)
                    handle.write(chunk)
        if digest.hexdigest() != MODEL_SHA256:
            raise ValueError('Vehicle model integrity check failed')
        temporary.replace(path)
        return path
    except Exception:
        _retry_after = time.monotonic() + 300
        raise
    finally:
        temporary.unlink(missing_ok=True)


def select_car(values, width, height, ratio):
    import numpy as np
    if values.shape != (3549, 85) or not np.isfinite(values).all():
        raise ValueError('Invalid vehicle model response')
    grids, strides = [], []
    for stride in (8, 16, 32):
        x, y = np.meshgrid(np.arange(416 // stride), np.arange(416 // stride))
        grid = np.stack((x, y), 2).reshape(-1, 2)
        grids.append(grid)
        strides.append(np.full((len(grid), 1), stride))
    stride = np.concatenate(strides)
    centers = (values[:, :2] + np.concatenate(grids)) * stride / ratio
    size = np.exp(np.clip(values[:, 2:4], -20, 20)) * stride / ratio
    # Objectness times the COCO car class (index 2, starting at column 5).
    confidence = values[:, 4] * values[:, 7]
    for index in np.argsort(confidence)[-12:][::-1]:
        score = float(confidence[index])
        if score < .5:
            continue
        x1, y1 = centers[index] - size[index] / 2
        x2, y2 = centers[index] + size[index] / 2
        area = max(0, min(width, x2) - max(0, x1)) * max(0, min(height, y2) - max(0, y1)) / (width * height)
        cx, cy = centers[index] / [width, height]
        if area >= .15 and .15 <= cx <= .85 and .15 <= cy <= .85:
            return {'confidence': round(score, 4), 'area': round(float(area), 4),
                    'box': [round(float(v), 2) for v in (max(0, x1), max(0, y1), min(width, x2), min(height, y2))],
                    'model': 'YOLOX-Nano', 'revision': '0.1.1rc0', 'class': 'car'}
    return None


def dominant_car(contents):
    global _session
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    if not contents or len(contents) > 20 * 1024 * 1024:
        raise ValueError('Invalid image size')
    with _lock:
        if _session is None:
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            options.enable_cpu_mem_arena = options.enable_mem_pattern = False
            _session = ort.InferenceSession(str(model_path()), sess_options=options, providers=['CPUExecutionProvider'])
            if _session.get_inputs()[0].shape != [1, 3, 416, 416]:
                _session = None
                raise ValueError('Invalid vehicle model input')
        with Image.open(io.BytesIO(contents)) as source:
            width, height = source.size
            if not width or not height or width * height > 20_000_000:
                raise ValueError('Invalid image dimensions')
            ratio = min(416 / width, 416 / height)
            image = source.convert('RGB').resize((max(1, int(width * ratio)), max(1, int(height * ratio))), Image.Resampling.BILINEAR)
            tensor = np.full((416, 416, 3), 114, dtype=np.float32)
            tensor[:image.height, :image.width] = np.asarray(image)[:, :, ::-1]
        values = _session.run(None, {'images': tensor.transpose(2, 0, 1)[None].copy()})[0][0]
        return select_car(values, width, height, ratio)
