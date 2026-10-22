"""Measure actual HTTP detections on a frozen labeled monitoring batch.

Serving AP uses the deployed confidence threshold, unlike offline training AP.
Existing source-label omissions remain a limitation; this is not quality approval.
"""

import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import requests

from src.monitoring.health import iso, model_identity, save


def prepare_batch(project, output):
    splits = json.loads((project / 'data/splits.json').read_text())['splits']
    names = splits['monitoring_current']
    others = {n for key, values in splits.items() if key != 'monitoring_current' for n in values}
    if len(names) < 200 or len(set(names)) != len(names) or set(names) & others:
        raise ValueError('Need at least 200 distinct held-out monitoring images')
    batch, hashes = [], {}
    for name in names:
        if Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('Invalid image identifier')
        image = project / f'data/processed/images/monitoring_current/{name}.png'
        label = project / f'data/processed/labels/monitoring_current/{name}.txt'
        body, text = image.read_bytes(), label.read_bytes()
        hashes[name] = {'image': hashlib.sha256(body).hexdigest(), 'label': hashlib.sha256(text).hexdigest()}
        with Image.open(BytesIO(body)) as im:
            width, height = im.size
        labels = []
        for line in text.decode().splitlines():
            values = [float(v) for v in line.split()]
            if len(values) != 5 or not np.isfinite(values).all() or values[0] not in (0, 1, 2):
                raise ValueError('Invalid monitoring label')
            cls, x, y, w, h = values
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError('Invalid normalized box')
            labels.append([cls, (x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height])
        batch.append((name, body, np.asarray(labels).reshape(-1, 5), width, height))
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    save(output / 'batch-sha256.json', hashes)
    return batch, digest


def matches(labels, detections):
    """One-to-one class-aware IoU matching, highest overlap first at each threshold."""
    correct = np.zeros((len(detections), 10), dtype=bool)
    if not len(labels) or not len(detections):
        return correct
    a, b = labels[:, 1:5], detections[:, :4]
    intersection = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:]) -
                              np.maximum(a[:, None, :2], b[None, :, :2])).prod(2)
    union = (a[:, 2:] - a[:, :2]).prod(1)[:, None] + (b[:, 2:] - b[:, :2]).prod(1)[None, :] - intersection
    iou = intersection / np.maximum(union, 1e-9)
    compatible = labels[:, 0, None] == detections[None, :, 5]
    for column, threshold in enumerate(np.linspace(.5, .95, 10)):
        rows, cols = np.where((iou >= threshold) & compatible)
        used_labels, used_predictions = set(), set()
        for index in np.argsort(-iou[rows, cols]):
            row, col = int(rows[index]), int(cols[index])
            if row not in used_labels and col not in used_predictions:
                correct[col, column] = True
                used_labels.add(row)
                used_predictions.add(col)
    return correct


def measure(batch, manifest, output, base='http://127.0.0.1:8000'):
    settings_path = Path(__file__).resolve().parents[2] / 'artifacts/ultralytics-settings'
    settings_path.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(settings_path)
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    from ultralytics.utils.metrics import ap_per_class

    identity = {k: manifest[k] for k in ('model_name', 'model_version', 'checkpoint_sha256', 'deployment_mode')}
    inference_hash = hashlib.sha256(json.dumps(manifest['inference'], sort_keys=True).encode()).hexdigest()
    records, stats = [], []
    with requests.Session() as session:
        session.trust_env = False
        for index, (name, body, labels, width, height) in enumerate(batch):
            response = session.post(base + '/predict', data=body, headers={'Content-Type': 'image/png'}, timeout=45)
            response.raise_for_status()
            prediction = response.json()
            if any(prediction.get(k) != v for k, v in identity.items()) or prediction.get('inference_sha256') != inference_hash:
                raise ValueError('Deployment changed during serving audit')
            if (prediction['width'], prediction['height']) != (width, height):
                raise ValueError('Prediction dimensions differ from input')
            boxes = np.asarray([[*d['xyxy'], d['confidence'], d['class_id']] for d in prediction['detections']], dtype=float).reshape(-1, 6)
            if not np.isfinite(boxes).all() or any(not (0 <= row[4] <= 1 and row[5] in (0, 1, 2)
                    and 0 <= row[0] <= row[2] <= width and 0 <= row[1] <= row[3] <= height) for row in boxes):
                raise ValueError('Invalid prediction boxes')
            stats.append((matches(labels, boxes), boxes[:, 4], boxes[:, 5], labels[:, 0]))
            records.append({'image_id': name, 'prediction': prediction})
            if (index + 1) % 50 == 0:
                print(f'Labeled serving audit: {index + 1}/{len(batch)} images.', flush=True)
    tp, confidence, classes, truth = [np.concatenate([s[i] for s in stats]) for i in range(4)]
    if not len(truth):
        raise ValueError('Audit has no labels')
    result = ap_per_class(tp, confidence, classes, truth, plot=False)
    report = {'map50_95': float(result[5].mean()), 'images': len(batch),
              'per_class': [{'class_id': int(cls), 'map50_95': float(ap.mean())} for cls, ap in zip(result[6], result[5])],
              'manifest': manifest, 'predictions': records, 'generated_at': iso(time.time())}
    save(output, report)
    return report


def comparison(project, batch_sha, before, after):
    if before['images'] != after['images'] or before['images'] < 200:
        raise ValueError('Incompatible audit batches')
    baseline, candidate = before['manifest'], after['manifest']
    return {'status': 'available', 'generated_at': iso(time.time()), 'candidate': model_identity(candidate),
            'baseline': {k: baseline[k] for k in ('model_name', 'model_version', 'checkpoint_sha256')},
            'images': before['images'], 'split_sha256': hashlib.sha256((project / 'data/splits.json').read_bytes()).hexdigest(),
            'baseline_batch_sha256': batch_sha, 'candidate_batch_sha256': batch_sha,
            'baseline_map50_95': before['map50_95'], 'candidate_map50_95': after['map50_95'],
            'evaluation': 'Actual HTTP detections; AP50-95 at each deployment confidence setting',
            'quality_approval': False, 'limitations': 'Source labels are incomplete; scoped configuration-fault demonstration only'}
