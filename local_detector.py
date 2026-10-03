"""Pinned ConvNeXt inference, no external image uploads or user-image storage."""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import threading

MODEL_ID = 'xRayon/convnext-ai-images-detector'
MODEL_REVISION = '1b4d270be0590cde4320fa7503d68b44bb7ee77e'
MODEL_SHA256 = '2b8b40817e30b1cc8b32dbb703cfad223255cb6657f4f12b792c166f6e9d3ab4'
MODEL_FILENAME = 'detector_model.onnx'
ROOT = Path(__file__).resolve().parent
_session = None
_lock = threading.Lock()
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 20_000_000

def _digest(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()

def assemble_model():
    """Stream verified small repository parts to a single model; never unpickle."""
    target = ROOT / MODEL_FILENAME
    if target.is_file() and _digest(target) == MODEL_SHA256:
        return target
    manifest = json.loads((ROOT / 'detector_model_manifest.json').read_text(encoding='utf-8'))
    if manifest.get('sha256') != MODEL_SHA256 or manifest.get('revision') != MODEL_REVISION:
        raise RuntimeError('Analiz modeli sürümü doğrulanamadı.')
    temporary = target.with_suffix('.assembling')
    try:
        combined = hashlib.sha256()
        with temporary.open('wb') as output:
            for index, part in enumerate(manifest['parts']):
                name = f'detector_model.part{index:02d}'
                if part['name'] != name:
                    raise RuntimeError('Analiz modeli parça sırası geçersiz.')
                source = ROOT / name
                if source.stat().st_size != part['bytes'] or _digest(source) != part['sha256']:
                    raise RuntimeError('Analiz modeli eksik veya bozuk. Model parçalarını yeniden yükleyin.')
                with source.open('rb') as file:
                    while chunk := file.read(1024 * 1024):
                        combined.update(chunk)
                        output.write(chunk)
        if temporary.stat().st_size != manifest['bytes'] or combined.hexdigest() != MODEL_SHA256:
            raise RuntimeError('Analiz modeli bütünlük kontrolünü geçemedi.')
        temporary.replace(target)
        return target
    except (OSError, KeyError, ValueError) as error:
        raise RuntimeError('Analiz modeli dosyaları eksik veya bozuk.') from error
    finally:
        temporary.unlink(missing_ok=True)

def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort
        path = assemble_model()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.enable_cpu_mem_arena = False
        options.enable_mem_pattern = False
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if len(inputs) != 1 or inputs[0].name != 'pixel_values' or inputs[0].shape != [1, 3, 256, 256] or outputs[0].name != 'logits':
            raise RuntimeError('Analiz modeli girdi biçimi geçersiz.')
        _session = session
    return _session

def _preprocess(contents):
    import numpy as np
    from PIL import Image
    if not contents or len(contents) > MAX_BYTES:
        raise ValueError('Görsel boş veya 20 MB sınırını aşıyor.')
    with Image.open(io.BytesIO(contents)) as image:
        w, h = image.size
        if w * h > MAX_PIXELS:
            raise ValueError('Görsel 20 megapiksel sınırını aşıyor. Daha küçük bir kopya yükleyin.')
        # Same Resize(288), CenterCrop(256), RGB and ImageNet normalization as
        # the tested publisher pipeline. Do not use JPEG decoder downsampling.
        size = (288, int(288 * h / w)) if w <= h else (int(288 * w / h), 288)
        if max(size) > 8192:
            raise ValueError('Görselin en-boy oranı desteklenmiyor.')
        source = image if image.mode == 'RGB' else image.convert('RGB')
        resized = source.resize(size, Image.Resampling.BILINEAR)
        left = round((size[0] - 256) / 2)
        top = round((size[1] - 256) / 2)
        crop = resized.crop((left, top, left + 256, top + 256))
        values = np.asarray(crop, dtype=np.float32) / 255
    values = (values - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
    return np.transpose(values, (2, 0, 1))[None].copy()

def analyze(contents):
    """Reject overlapping work instead of queuing decoded images in scarce RAM."""
    if not _lock.acquire(blocking=False):
        raise RuntimeError('Analiz motoru şu anda meşgul. Birkaç saniye sonra tekrar deneyin.')
    try:
        import numpy as np
        session = _get_session()
        tensor = _preprocess(contents)
        logits = session.run(['logits'], {'pixel_values': tensor})[0]
        if logits.shape != (1, 2) or not np.isfinite(logits).all():
            raise RuntimeError('Analiz modeli geçerli bir sonuç üretmedi.')
        probabilities = np.exp(logits[0] - np.max(logits[0]))
        score = float(probabilities[1] / probabilities.sum()) * 100
        if not math.isfinite(score) or not 0 <= score <= 100:
            raise RuntimeError('Analiz modeli skoru geçersiz.')
        return score, [{'model': MODEL_ID, 'revision': MODEL_REVISION,
            'variant': 'onnx-dynamic-int8', 'ai_score': round(score, 1), 'weight': 1.0}]
    finally:
        _lock.release()
