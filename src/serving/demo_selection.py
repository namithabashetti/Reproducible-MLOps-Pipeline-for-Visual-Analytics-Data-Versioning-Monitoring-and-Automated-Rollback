"""One local demonstration model, independent of Docker deployment and registry aliases."""
import hashlib
import json
from threading import RLock
from time import perf_counter

from fastapi import HTTPException
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from src.serving.api import decode_image, load_model, predict_image
from src.serving.deploy import stage_model, write_receipt

lock = RLock()
selected = {'model': None, 'manifest': None}
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


def health():
    with lock:
        if selected['manifest'] is None:
            raise HTTPException(503, 'Select an available demonstration model')
        return {'status':'ready', **selected['manifest']}


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
        active.clear()
        active.labels(model_id,str(manifest['model_version']),generation,manifest['checkpoint_sha256']).set(1)
        return health()


def predict(body):
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
        return {**{k:manifest[k] for k in ('model_name','model_version','checkpoint_sha256','deployment_mode',
                                         'demo_model_id','generation','selection_mode','run')},
            'inference_sha256':hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest(),
            'confidence_threshold':settings['confidence'],'inference_ms':seconds*1000,
            'processing_ms':(perf_counter()-started)*1000,'width':image.shape[1],'height':image.shape[0],
            'detections':detections,'audit_requested':False}


def metrics():
    return generate_latest(registry)
