"""One local demonstration model, independent of Docker deployment and registry aliases."""
import hashlib
import json
from pathlib import Path
from threading import RLock
from time import perf_counter

from fastapi import HTTPException
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
import yaml

from src.serving.api import decode_image, load_model, predict_image
from src.serving.deploy import stage_model, write_receipt

lock = RLock()
selected = {'model': None, 'manifest': None}
confidence_guard = {'consecutive_low': 0, 'last_max_confidence': None, 'last_trigger': None, 'last_error': None}
registry = CollectorRegistry()
requests = Counter('va_demo_requests', 'Local demonstration requests by actual model and outcome',
                   ('model_id','version','outcome'), registry=registry)
latency = Histogram('va_demo_request_seconds', 'Local demo request time including decoding and waiting',
                    ('model_id','version'), registry=registry)
inference = Histogram('va_demo_inference_seconds', 'Local demo model inference time',
                      ('model_id','version'), registry=registry)
active = Gauge('va_demo_active_model', 'Currently selected demonstration checkpoint; separate from deployment',
               ('model_id','version','generation','checkpoint_sha256'), registry=registry)


def catalog(project):
    plan = json.loads((project/'configs/retraining.json').read_text())
    rows = []
    for job in plan['jobs']:
        for generation, name in (('historical', job['historical_run']), ('retrained', job['run'])):
            directory = project/'artifacts/training'/name
            receipt = directory/'mlflow.json'
            try:
                registration = json.loads(receipt.read_text())
            except (OSError,ValueError):
                registration = None
            available = bool(registration and registration.get('verified') and (directory/'fit/weights/best.pt').is_file())
            rows.append({'model_id':job['model_id'],'generation':generation,'run':name,
                'available':available,'status':'Registered; load to verify' if available else 'Training or registration incomplete',
                'model_version':registration.get('model_version') if registration else None,
                'checkpoint_sha256':registration.get('checkpoint_sha256') if registration else None})
    return rows


def fallback_policy(project=None):
    project = Path(project) if project is not None else Path(__file__).resolve().parents[2]
    config = yaml.safe_load((project/'configs/config.yaml').read_text(encoding='utf-8'))
    policy = config['demo']['automatic_fallback']
    if type(policy.get('enabled')) is not bool or not 0 < policy.get('max_confidence_threshold', 0) <= 1 or \
            type(policy.get('consecutive_predictions')) is not int or policy['consecutive_predictions'] < 1 or \
            policy.get('candidate_model_id') == policy.get('fallback_model_id'):
        raise ValueError('Invalid automatic demonstration fallback policy')
    return policy


def health(project=None):
    with lock:
        if selected['manifest'] is None:
            raise HTTPException(503, 'Select an available demonstration model')
        policy = fallback_policy(project)
        armed = (policy['enabled'] and selected['manifest']['demo_model_id'] == policy['candidate_model_id'] and
                 selected['manifest']['generation'] == policy['generation'])
        return {'status':'ready', **selected['manifest'], 'automatic_fallback': {
            'enabled': policy['enabled'], 'armed': armed,
            'threshold': policy['max_confidence_threshold'],
            'required_consecutive_predictions': policy['consecutive_predictions'],
            'consecutive_low_confidence_predictions': confidence_guard['consecutive_low'],
            'candidate_model_id': policy['candidate_model_id'],
            'fallback_model_id': policy['fallback_model_id'],
            'generation': policy['generation'], 'last_max_confidence': confidence_guard['last_max_confidence'],
            'last_trigger': confidence_guard['last_trigger'], 'last_error': confidence_guard['last_error']}}


def choose(project, model_id, generation, expected_checksum=None):
    if model_id not in ('1','2','3','4','5') or generation not in ('historical','retrained'):
        raise HTTPException(400, 'Choose Model 1–5 and an available model set')
    with lock:
        row = next(r for r in catalog(project) if r['model_id']==model_id and r['generation']==generation)
        if not row['available']:
            raise HTTPException(409, 'This model has not completed training and registration; selection is unchanged')
        if expected_checksum and expected_checksum != row['checkpoint_sha256']:
            raise ValueError('Saved demonstration checkpoint identity changed')
        staged, manifest = stage_model(project, project, str(row['model_version']), experimental=True)
        manifest = json.loads(json.dumps(manifest))  # JSON class keys are strings on disk.
        if manifest['checkpoint_sha256'] != row['checkpoint_sha256']:
            raise ValueError('Registry checkpoint differs from the chosen experiment')
        candidate, verified = load_model(staged)  # Includes checkpoint hash, class validation and warm-up.
        if verified != manifest:
            raise ValueError('Loaded checkpoint manifest changed')
        manifest = {**manifest, 'demo_model_id':model_id,'generation':generation,
                    'selection_mode':'demonstration','run':row['run']}
        # Publish only after successful verification; failure retains the old in-memory model.
        write_receipt(project/'artifacts/serving/demo-selection.json',manifest)
        selected.update(model=candidate,manifest=manifest)
        confidence_guard.update(consecutive_low=0, last_max_confidence=None, last_error=None)
        active.clear()
        active.labels(model_id,str(manifest['model_version']),generation,manifest['checkpoint_sha256']).set(1)
        return health()


def predict(body, project=None):
    started = perf_counter()
    with lock:
        manifest = selected['manifest']
        if manifest is None:
            raise HTTPException(503, 'Select an available demonstration model first')
        settings = manifest['inference']
        if len(body)>settings['max_upload_bytes']:
            raise HTTPException(413, 'Image must be under 4 MB')
        image = decode_image(body, settings['max_image_pixels'])
        before = perf_counter()
        detections = predict_image(selected['model'], image, settings)
        seconds = perf_counter()-before
        inference.labels(manifest['demo_model_id'],str(manifest['model_version'])).observe(seconds)
        maximum = max((float(d['confidence']) for d in detections), default=0.0)
        policy = fallback_policy(project)
        applies = (policy['enabled'] and manifest['demo_model_id'] == policy['candidate_model_id'] and
                   manifest['generation'] == policy['generation'])
        confidence_guard['last_max_confidence'] = maximum
        confidence_guard['last_error'] = None
        if applies and maximum < policy['max_confidence_threshold']:
            confidence_guard['consecutive_low'] += 1
        else:
            confidence_guard['consecutive_low'] = 0
        fallback_event = {'triggered': False, 'armed': applies,
                          'threshold': policy['max_confidence_threshold'],
                          'consecutive_low_confidence_predictions': confidence_guard['consecutive_low'],
                          'required_consecutive_predictions': policy['consecutive_predictions'],
                          'max_detection_confidence': maximum}
        if applies and confidence_guard['consecutive_low'] >= policy['consecutive_predictions']:
            source = {'demo_model_id': manifest['demo_model_id'], 'model_version': manifest['model_version'],
                      'checkpoint_sha256': manifest['checkpoint_sha256']}
            try:
                restored = choose(project or Path(__file__).resolve().parents[2], policy['fallback_model_id'],
                                  policy['generation'])
                fallback_event.update(triggered=True, from_model=source,
                    to_model={'demo_model_id': restored['demo_model_id'], 'model_version': restored['model_version'],
                              'checkpoint_sha256': restored['checkpoint_sha256'],
                              'generation': restored['generation']},
                    reason=f"{policy['consecutive_predictions']} consecutive images had maximum detection confidence below {policy['max_confidence_threshold']:.2f}")
                confidence_guard.update(consecutive_low=0, last_trigger=fallback_event)
            except Exception as error:
                confidence_guard.update(consecutive_low=0, last_error=type(error).__name__)
                fallback_event.update(error=type(error).__name__)
        response = {**{k:manifest[k] for k in ('model_name','model_version','checkpoint_sha256','deployment_mode',
                                         'demo_model_id','generation','selection_mode','run')},
            'inference_sha256':hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest(),
            'confidence_threshold':settings['confidence'],'inference_ms':seconds*1000,
            'processing_ms':(perf_counter()-started)*1000,'width':image.shape[1],'height':image.shape[0],
            'detections':detections,'audit_requested':False,'automatic_fallback':fallback_event}
        return response


def metrics():
    return generate_latest(registry)
