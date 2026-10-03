"""Run from a scheduled job; never imports app.py or touches users.db."""
import argparse
import csv
import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
import requests
from PIL import Image
from detector_core import parse_prediction

def models(raw):
    items = []
    for item in raw.split(','):
        if item.strip():
            name = item.strip().rsplit(':', 1)[0]
            if name not in items:
                items.append(name)
    return items

def infer(model, data, mime):
    response = requests.post('https://router.huggingface.co/hf-inference/models/' + model,
        headers={'Authorization': 'Bearer ' + os.environ['HF_TOKEN'], 'Content-Type': mime},
        data=data, timeout=30)
    response.raise_for_status()
    return parse_prediction(response.json(), model)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='CSV columns: path,label; labels AI/REAL. Verified held-out images only.')
    parser.add_argument('--output', default='maintenance-report.json')
    args = parser.parse_args()
    if not os.getenv('HF_TOKEN') or not os.getenv('AI_MODELS'):
        parser.error('HF_TOKEN and AI_MODELS are required')
    active = models(os.environ['AI_MODELS'])
    candidates = models(os.getenv('AI_CANDIDATE_MODELS', ''))
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'models': {},
              'automatic_promotion': False, 'note': 'Health probes do not measure detection accuracy.'}
    image = Image.new('RGB', (384, 384), (128, 128, 128))
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    dataset = []
    if args.dataset:
        source = Path(args.dataset).resolve()
        with source.open(encoding='utf-8-sig', newline='') as file:
            for row in csv.DictReader(file):
                label = row['label'].strip().upper()
                if label not in ('AI', 'REAL'):
                    raise ValueError('Dataset labels must be AI or REAL')
                path = (source.parent / row['path']).resolve()
                with Image.open(path) as sample:
                    mime = 'image/jpeg' if sample.format == 'JPEG' else 'image/png'
                    if sample.format not in ('JPEG', 'PNG'):
                        raise ValueError('Benchmark images must be JPEG or PNG')
                dataset.append((path.read_bytes(), mime, label))
        if not any(label == 'AI' for _, _, label in dataset) or not any(label == 'REAL' for _, _, label in dataset):
            raise ValueError('Both classes are required')
    threshold = float(os.getenv('DECISION_THRESHOLD', '60'))
    failed = False
    for model in dict.fromkeys(active + candidates):
        entry = {'role': 'active' if model in active else 'candidate'}
        try:
            if not dataset:
                infer(model, buf.getvalue(), 'image/png')
                entry['status'] = 'reachable'
            else:
                tp = tn = fp = fn = 0
                for data, mime, label in dataset:
                    predicted_ai = infer(model, data, mime) > threshold
                    tp += predicted_ai and label == 'AI'
                    fp += predicted_ai and label == 'REAL'
                    tn += not predicted_ai and label == 'REAL'
                    fn += not predicted_ai and label == 'AI'
                entry.update(status='evaluated', count=len(dataset),
                    balanced_accuracy=(tp / (tp + fn) + tn / (tn + fp)) / 2,
                    false_positive_rate=fp / (fp + tn), false_negative_rate=fn / (fn + tp),
                    tp=tp, tn=tn, fp=fp, fn=fn)
        except Exception as error:
            # Never include API bodies, request headers or token values in reports.
            entry.update(status='failed', error_type=type(error).__name__)
            failed = failed or model in active
        report['models'][model] = entry
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Maintenance report saved. Active model failures:', failed)
    return 1 if failed else 0

if __name__ == '__main__':
    sys.exit(main())
