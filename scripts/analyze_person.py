"""Diagnose the existing baseline on train-label statistics and validation predictions only."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.monitoring.health import save
from src.tracking.mlflow_utils import file_hash, local_store


def iou(box, boxes):
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    low = np.maximum(np.asarray(box[:2]), boxes[:, :2])
    high = np.minimum(np.asarray(box[2:]), boxes[:, 2:])
    intersection = np.prod(np.maximum(0, high - low), axis=1)
    area = (box[2] - box[0]) * (box[3] - box[1])
    other = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
    return intersection / np.maximum(area + other - intersection, 1e-12)


def counts(truth, predictions, cutoff, class_id=2):
    """Confidence-ordered one-to-one IoU>=.5 counts, not a new AP implementation."""
    gt = [row['box'] for row in truth if row['class_id'] == class_id]
    chosen = sorted((p for p in predictions if p['class_id'] == class_id and p['confidence'] >= cutoff),
                    key=lambda p: -p['confidence'])
    matched = set()
    for pred in chosen:
        overlaps = iou(pred['box'], gt)
        eligible = [index for index in range(len(gt)) if index not in matched and overlaps[index] >= .5]
        if eligible:
            matched.add(max(eligible, key=lambda index: overlaps[index]))
    return {'tp': len(matched), 'fp': len(chosen) - len(matched), 'fn': len(gt) - len(matched)}


def labels(project, split, name, size):
    result = []
    for line in (project / 'data/processed/labels' / split / f'{name}.txt').read_text().splitlines():
        class_id, x, y, width, height = map(float, line.split())
        result.append({'class_id': int(class_id), 'box': [size * (x - width / 2), size * (y - height / 2),
                                                       size * (x + width / 2), size * (y + height / 2)]})
    return result


def metadata(project):
    digest = hashlib.sha256()
    files = sorted(p for p in (project / 'dataset').rglob('*') if p.is_file())
    for path in files:
        stat = path.stat()
        digest.update(f'{path.relative_to(project).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n'.encode())
    return {'files': len(files), 'metadata_sha256': digest.hexdigest()}


def analyze(project, output):
    config = yaml.safe_load((project / 'configs/config.yaml').read_text())
    size = yaml.safe_load((project / 'params.yaml').read_text())['training']['image_size']
    splits = json.loads((project / 'data/splits.json').read_text())['splits']
    protected = set(splits['test']) | set(splits['monitoring_reference']) | set(splits['monitoring_current'])
    if protected & (set(splits['train']) | set(splits['validation'])):
        raise ValueError('Diagnostic input membership overlaps protected pools')
    before = metadata(project)
    weights = project / 'artifacts/training/baseline/fit/weights/best.pt'
    weight_hash = file_hash(weights)
    recorded = json.loads((project / 'artifacts/training/baseline/metrics.json').read_text())
    if weight_hash != recorded['checkpoint_sha256']:
        raise ValueError('Baseline checkpoint differs from its recorded evaluation')
    client, _, registry = local_store(project)
    aliases = dict(client.get_registered_model(registry['model_name']).aliases)
    summary = {'started_at': datetime.now(timezone.utc).isoformat(), 'checkpoint_sha256': weight_hash,
               'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=project, text=True).strip(),
               'source_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=project)),
               'script_sha256': file_hash(Path(__file__)), 'split_sha256': file_hash(project / 'data/splits.json'),
               'settings': {'image_size': size, 'device': 'cpu', 'cpu_threads': 2, 'confidence_floor': .001,
                            'iou_nms': .7, 'max_detections': 300, 'match_iou': .5,
                            'confidence_cutoffs': [.001, .01, .05, .1, .25]},
               'recorded_evaluation': recorded, 'label_statistics': {}, 'operating_counts': {}}
    for split in ('train', 'validation'):
        boxes, images = Counter(), Counter()
        person_sizes = []
        for name in splits[split]:
            truth = labels(project, split, name, size)
            boxes.update(row['class_id'] for row in truth)
            images.update({row['class_id'] for row in truth})
            for row in truth:
                if row['class_id'] == 2:
                    x1, y1, x2, y2 = row['box']
                    person_sizes.append(min(x2 - x1, y2 - y1))
        summary['label_statistics'][split] = {'images': len(splits[split]), 'class_boxes': dict(boxes),
            'class_images': dict(images), 'person_min_side_pixels_quantiles': np.quantile(person_sizes, [0, .25, .5, .75, 1]).tolist(),
            'person_min_side_under16': sum(v < 16 for v in person_sizes),
            'person_min_side_under32': sum(v < 32 for v in person_sizes)}

    # All settings/cache directories stay in the report directory; predict() does not write label caches.
    os.environ.update(YOLO_CONFIG_DIR=str(output / 'ultralytics-settings'), YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false')
    import torch
    from ultralytics import YOLO, settings
    settings.update({'sync': False})
    torch.set_num_threads(2)
    model = YOLO(str(weights))
    if {int(k): v for k, v in model.names.items()} != config['dataset']['classes']:
        raise ValueError('Checkpoint class mapping differs from dataset configuration')
    observations = []
    person_confidence = []
    failures = Counter()
    for index, name in enumerate(splits['validation'], 1):
        truth = labels(project, 'validation', name, size)
        result = model.predict(str(project / 'data/processed/images/validation' / f'{name}.png'),
            device='cpu', imgsz=size, conf=.001, iou=.7, max_det=300, rect=False, half=False,
            augment=False, verbose=False, save=False)[0]
        predictions = [{'class_id': int(c), 'confidence': float(s), 'box': b.tolist()}
                      for b, c, s in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(),
                                         result.boxes.conf.cpu().numpy())]
        person_confidence.extend(p['confidence'] for p in predictions if p['class_id'] == 2)
        person_truth = [row for row in truth if row['class_id'] == 2]
        coverage = []
        for gt in person_truth:
            overlaps = iou(gt['box'], [p['box'] for p in predictions])
            matching = [p for p, overlap in zip(predictions, overlaps) if overlap >= .5]
            own = [p for p in matching if p['class_id'] == 2]
            maximum = max((p['confidence'] for p in own), default=0.)
            reason = 'person_overlap_at_serving_cutoff' if maximum >= .25 else \
                'person_overlap_only_below_serving_cutoff' if own else \
                'other_class_overlap_only' if matching else 'no_retained_box_with_iou50'
            failures[reason] += 1
            coverage.append({'box': gt['box'], 'reason': reason, 'best_person_confidence_at_iou50': maximum})
        observations.append({'image_id': name, 'truth': truth, 'predictions': predictions,
                             'person_coverage': coverage, 'hit_max_detections': len(predictions) == 300})
        if index % 50 == 0:
            print(f'validation: {index}/{len(splits["validation"])} images', flush=True)
    for cutoff in summary['settings']['confidence_cutoffs']:
        total = Counter()
        for row in observations:
            total.update(counts(row['truth'], row['predictions'], cutoff))
        summary['operating_counts'][str(cutoff)] = {**total,
            'precision': total['tp'] / (total['tp'] + total['fp']) if total['tp'] + total['fp'] else 0.,
            'recall': total['tp'] / (total['tp'] + total['fn']) if total['tp'] + total['fn'] else 0.}
    summary['person_ground_truth_coverage'] = dict(failures)
    summary['person_candidate_confidence_quantiles'] = np.quantile(person_confidence, [0, .5, .9, .99, 1]).tolist() if person_confidence else []
    summary['images_hitting_max_detections'] = sum(row['hit_max_detections'] for row in observations)
    summary['person_image_ids'] = [row['image_id'] for row in observations if row['person_coverage']]
    if metadata(project) != before or file_hash(weights) != weight_hash or \
            dict(client.get_registered_model(registry['model_name']).aliases) != aliases:
        raise ValueError('Dataset, model bytes or aliases changed during analysis')
    status = json.loads(subprocess.check_output([sys.executable, '-m', 'dvc', 'status', '--json'], cwd=project, text=True))
    if status:
        raise ValueError('DVC outputs changed during analysis')
    summary.update(completed_at=datetime.now(timezone.utc).isoformat(), preservation=before, dvc_status=status,
                   registry_aliases=aliases, final_test_evaluated=False, training_performed=False,
                   limitations=['Confidence-ordered IoU50 counts are diagnostic; recorded Ultralytics AP remains the model-selection metric.',
                                'Coverage categories are per ground-truth box and need not be one-to-one matches.',
                                'Retained low-confidence candidates are not proof of reliable person detection.',
                                'Visual annotation review is separate from inference metrics.'])
    save(output / 'observations.json', observations)
    save(output / 'summary.json', summary)
    print(json.dumps({'operating_counts': summary['operating_counts'], 'coverage': dict(failures)}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((PROJECT / 'reports').resolve()):
        parser.error('Diagnostic output must stay under project reports/')
    output.mkdir(parents=True, exist_ok=False)
    analyze(PROJECT, output)


if __name__ == '__main__':
    main()
