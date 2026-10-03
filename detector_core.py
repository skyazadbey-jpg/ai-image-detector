"""Strict response parsing; intentionally independent of FastAPI and databases."""
import json
import math
import os

AI_LABELS = {'artificial', 'fake', 'ai', 'generated', 'deepfake', 'ai-generated', 'ai generated', 'artificial intelligence'}
REAL_LABELS = {'human', 'hum', 'real', 'authentic', 'natural'}

def parse_prediction(result, model_id=''):
    if isinstance(result, list) and result and isinstance(result[0], list):
        result = result[0]
    if not isinstance(result, list) or not result:
        raise ValueError('Empty prediction')
    # Optional per-model maps, e.g. {"org/model": {"label_0": "AI", "label_1": "REAL"}}.
    maps = json.loads(os.getenv('AI_LABEL_MAPS', '{}')).get(model_id, {})
    ai, real = [], []
    for item in result:
        label = str(item['label']).strip().lower()
        score = float(item['score'])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid score')
        mapped = str(maps.get(label, '')).upper()
        if label in AI_LABELS or mapped == 'AI':
            ai.append(score)
        elif label in REAL_LABELS or mapped == 'REAL':
            real.append(score)
        else:
            raise ValueError('Unknown label; configure an explicit per-model map')
    if len(ai) > 1 or len(real) > 1:
        raise ValueError('Ambiguous class mapping')
    if ai and real and abs(ai[0] + real[0] - 1) > 0.02:
        raise ValueError('Inconsistent binary probabilities')
    return 100 * (ai[0] if ai else 1 - real[0])

def decide_verdict(score, threshold=60, margin=10, uncertain=False):
    if uncertain or abs(score - threshold) <= margin:
        return 'UNCERTAIN'
    return 'AI' if score > threshold else 'REAL'
