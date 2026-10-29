"""Loopback-only presentation console; inference stays in the existing Docker API."""

import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import secrets
import re
import subprocess
import sys
from threading import Lock, Thread
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware
from src.monitoring.dashboard import evidence_metrics, project_status
import requests
from src.serving import demo_selection

ROOT = Path(__file__).resolve().parents[2]
API = 'http://127.0.0.1:8000'
MAX_BYTES = 4 * 1024 * 1024
TOKEN = secrets.token_urlsafe(32)
SAMPLES = {
    'css-e655ed922fb27bc2a40af8d2': ('New: workers in a row', 'Additional validation image: compare person boxes.'),
    'css-7fdd6457c593392b8fc47646': ('New: site inspection', 'Additional validation image: some distant people are still missed.'),
    'css-1e68af9b6ef60871ca6ec6ae': ('New: indoor ladder work', 'Additional validation image: two people at different heights.'),
    'css-3f01c8c0a88e712f7d82e4fa': ('New: outdoor ladder work', 'Additional validation image: helmet and uncovered head.'),
    'hard_hat_workers1017': ('Helmets and uncovered heads', 'Several workers indoors.'),
    'hard_hat_workers108': ('Two workers outdoors', 'Bright sky and clear helmets.'),
    'hard_hat_workers3058': ('Small, distant helmets', 'A harder scene with smaller objects.'),
    'hard_hat_workers4965': ('A group of workers', 'Several helmets in a busy scene.'),
    'hard_hat_workers2158': ('Partly hidden workers', 'Equipment covers part of the scene.'),
    'hard_hat_workers1209': ('A distant worker', 'A small worker near industrial equipment.'),
    'hard_hat_workers1215': ('Working at height', 'Two workers in different positions.'),
    'hard_hat_workers1234': ('Workers near branches', 'A busy background with several helmets.'),
}
SAMPLE_GUIDE = '''TEST IMAGES - MODEL COMPARISON AND AUTOMATIC ROLLBACK

Four css-* PNGs are additional validation images, outside training. They were
selected after comparing actual predictions to illustrate person-box improvements.
Eight hard_hat_workers* PNGs are from the existing held-out monitoring pool.
These selected examples are not an independent accuracy benchmark.
Person detection and source-label completeness remain limitations.

1. Open http://127.0.0.1:8767/ and choose a test image or upload a PNG.
2. The main page compares selected demo models; read the actual model label.
   This selection does not change the deployment or trigger its rollback.
3. Open /pipeline for the separate protected deployment (source model 4).
   Automatic protection starts with the application; wait for its active status.
4. Normal detection uses the candidate's real confidence setting, not a forced fault.
5. Try the SAME images on the candidate and after a recovery. No image is
   guaranteed to fail: these files illustrate detections, not an automatic trigger.
6. The controller measures a separate batch of 250 labeled images every five
   minutes. One uploaded image or missed camera frame cannot establish accuracy.
7. A qualifying quality drop or repeated readiness failures restores model 1
   and verifies real predictions and monitoring without a rollback click.
8. For a repeatable presentation, Simulate service outage is OPTIONAL: it stops
   the candidate service. The already-running watcher independently restores the
   fallback. This proves failure recovery, not natural model-quality degradation.

The copied demo registry calls its candidate version 2. Its trained source is
model 4 when selected. Existing trained model 2 has not been overwritten.
Model 4 trained for 60 epochs on 6,020 images, including 2,520 additional images.
The dashboard displays a shared 196-image evaluation with explicit settings.
Scores from older evaluations with different settings must not be mixed with it.
Model 4 has the highest shared score; person detection is still imperfect.
No production quality approval. Fresh protocol training is a separate experiment.
The 1,154-image protocol test was already evaluated by model 5. It is not a
fresh final test for the corrected-label experiment. Older models also trained
on some images reassigned to that split.
'''

app = FastAPI(title='Visual analytics live console', docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost', 'host.docker.internal'])
lock = Lock()
job = {'process': None, 'output': None, 'misses': 0, 'candidate_version': '3', 'mode':'quality', 'pid':None}
JOB_FILE = ROOT / 'reports/live-console-job.json'
if JOB_FILE.exists():
    try:
        saved = json.loads(JOB_FILE.read_text())
        saved_name = saved['directory']
        if re.fullmatch(r'live-demo-[0-9a-f]{12}', saved_name):
            job['output'] = ROOT / 'reports' / saved_name
            if saved.get('candidate_version') in ('2', '3', '4'):
                job['candidate_version'] = saved['candidate_version']
            job.update(mode=saved.get('mode','quality'),pid=saved.get('pid'))
    except (ValueError, KeyError, TypeError):
        pass


def active_job():
    if job['process'] is not None:
        return job['process'].poll() is None
    if job.get('pid') and job['output']:
        import psutil
        try:
            command = psutil.Process(job['pid']).cmdline()
            return (str(ROOT/'scripts/demo_rollback.py') in command and
                    str(job['output']) in command and job.get('mode') in command)
        except (psutil.Error, OSError):
            pass
    return False


def launch_demo(mode, candidate_version):
    output = ROOT / 'reports' / ('live-demo-' + uuid4().hex[:12])
    log = output.with_suffix('.log')
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('wb') as stream:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'scripts/demo_rollback.py'),
            '--output', str(output), '--mode', mode, '--candidate-version', candidate_version],
            cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    job.update(process=process,output=output,misses=0,candidate_version=candidate_version,mode=mode,pid=int(process.pid))
    from src.serving.deploy import write_receipt
    write_receipt(JOB_FILE, {'directory':output.name,'candidate_version':candidate_version,'mode':mode,'pid':int(process.pid)})
    return output


@app.on_event('startup')
def start_automatic_protection():
    """Automatic college-demo protection starts with the application, without a UI click."""
    with lock:
        if active_job() or (ROOT/'artifacts/serving/deployment.lock').exists():
            return
        if job['output'] and not read_json(job['output']/'result.json'):
            return  # Preserve an interrupted deployment for inspection.
        launch_demo('automatic','4')


@app.on_event('startup')
def load_saved_demo_selection():
    def load():
        import logging
        try:
            saved = read_json(ROOT/'artifacts/serving/demo-selection.json') or {}
            demo_selection.choose(ROOT, saved.get('demo_model_id','4'), saved.get('generation','historical'),
                                  saved.get('checkpoint_sha256'))
        except Exception:
            logging.exception('Demonstration model could not be loaded; no fallback was substituted')
    Thread(target=load,name='load-demo-model',daemon=True).start()


@app.middleware('http')
async def local_controls(request, call_next):
    if request.method != 'GET':
        if not secrets.compare_digest(request.headers.get('x-console-token', ''), TOKEN):
            return Response('Reload the local console before using controls', status_code=403)
        origin = request.headers.get('origin')
        if origin and origin != str(request.base_url).rstrip('/'):
            return Response('Cross-origin controls are disabled', status_code=403)
    response = await call_next(request)
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    return response



@app.middleware('http')
async def demo_request_metrics(request, call_next):
    if request.method != 'POST' or request.url.path != '/predict':
        return await call_next(request)
    from time import perf_counter
    started,status=perf_counter(),500
    manifest=demo_selection.selected.get('manifest') or {}
    labels=(manifest.get('demo_model_id','unselected'),str(manifest.get('model_version','none')))
    try:
        response=await call_next(request)
        status=response.status_code
        return response
    finally:
        labels=getattr(request.state,'demo_identity',labels)
        outcome='server_error' if status>=500 else 'client_error' if status>=400 else 'success'
        demo_selection.requests.labels(*labels,outcome).inc()
        demo_selection.latency.labels(*labels).observe(perf_counter()-started)


def call_api(path, **kwargs):
    try:
        with requests.Session() as session:
            session.trust_env = False
            response = session.request('POST' if 'data' in kwargs else 'GET', API + path, timeout=45, **kwargs)
        if response.status_code != 200:
            raise HTTPException(response.status_code, 'Inference API rejected the request: ' + response.text[:250])
        return response.json()
    except requests.RequestException as error:
        raise HTTPException(503, 'Inference API unavailable; a deployment may be in progress') from error


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, ValueError):
        return None


@app.get('/', response_class=HTMLResponse)
def home():
    return (Path(__file__).with_name('console.html').read_text(encoding='utf-8')).replace('__CONSOLE_TOKEN__', TOKEN)


@app.get('/pipeline', response_class=HTMLResponse)
def pipeline():
    return Path(__file__).with_name('pipeline.html').read_text(encoding='utf-8').replace('__CONSOLE_TOKEN__', TOKEN)


@app.get('/health')
def health():
    return demo_selection.health()


@app.get('/deployment/health')
def deployment_health():
    return call_api('/health')


@app.get('/models')
def available_models():
    return demo_selection.catalog(ROOT)


@app.post('/models/select')
async def select_model(request: Request):
    body = bytearray()
    async for chunk in request.stream():
        if len(body)+len(chunk)>1024:
            raise HTTPException(413,'Selection request too large')
        body.extend(chunk)
    try:
        value=json.loads(body)
        model_id,generation=value['model_id'],value['generation']
    except (ValueError,KeyError,TypeError):
        raise HTTPException(400,'Choose a model and model set')
    try:
        return await run_in_threadpool(demo_selection.choose,ROOT,model_id,generation)
    except HTTPException:
        raise
    except Exception as error:
        import logging
        logging.exception('Demonstration model switch failed')
        raise HTTPException(422,'Model verification failed; previous selection is unchanged ('+type(error).__name__+')') from error


@app.get('/project/status')
def project_evidence():
    return project_status(ROOT)


@app.get('/metrics')
def metrics():
    return Response(evidence_metrics(ROOT)+demo_selection.metrics(),media_type='text/plain; version=0.0.4; charset=utf-8')


@app.get('/sample')
def sample():
    return Response((ROOT / 'dataset/images/hard_hat_workers1015.png').read_bytes(), media_type='image/png')


def sample_bytes(image_id):
    if image_id not in SAMPLES:
        raise HTTPException(404, 'Unknown test image')
    folder = 'data/expanded-css-v1/images/validation' if image_id.startswith('css-') else 'data/processed/images/monitoring_current'
    path = ROOT / folder / (image_id + '.png')
    if not path.is_file():
        raise HTTPException(503, 'Test image is unavailable. Restore the existing processed dataset.')
    return path.read_bytes()


@app.get('/samples')
def samples():
    return [{'id': key, 'title': value[0], 'description': value[1], 'url': '/samples/' + key}
            for key, value in SAMPLES.items()]


@app.get('/samples/{image_id}')
def sample_image(image_id: str):
    return Response(sample_bytes(image_id), media_type='image/png')


@app.get('/sample-pack')
def sample_pack():
    output = BytesIO()
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        archive.writestr('START-HERE.txt', SAMPLE_GUIDE)
        for image_id in SAMPLES:
            archive.writestr(image_id + '.png', sample_bytes(image_id))
    return Response(output.getvalue(), media_type='application/zip',
                    headers={'Content-Disposition': 'attachment; filename="mlops-test-images.zip"'})


@app.get('/faculty-pack')
def faculty_pack():
    names = ('test_normal_01.png', 'test_changed_data_01.png', 'test_indoor_ladder_01.png',
             'test_outdoor_ladder_01.png', 'START-HERE.md', 'SOURCE-NOTICE.txt', 'sources.json')
    folder = ROOT / 'demo-images'
    if not all((folder / name).is_file() for name in names):
        raise HTTPException(503, 'Faculty images are unavailable. Restore demo-images.dvc.')
    output = BytesIO()
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        for name in names:
            archive.writestr(name, SAMPLE_GUIDE if name == 'START-HERE.md' else (folder / name).read_bytes())
    return Response(output.getvalue(), media_type='application/zip',
                    headers={'Content-Disposition': 'attachment; filename="faculty-test-images.zip"'})


def request_audit():
    output = job['output']
    progress = read_json(output / 'progress.json') if output else None
    if progress and progress['mode'] in ('quality','automatic') and progress['stage'] in ('candidate_live', 'awaiting_quality_check','automatic_monitoring'):
        (output / 'request-audit').touch(exist_ok=True)
        return True
    return False


@app.post('/deployment/predict')
async def deployment_predict(request: Request):
    content_type = request.headers.get('content-type', '').split(';')[0]
    if content_type not in ('image/png', 'image/jpeg'):
        raise HTTPException(415, 'Select a PNG or JPEG image')
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > MAX_BYTES:
            raise HTTPException(413, 'Image must be under 4 MB')
    value = await run_in_threadpool(call_api, '/predict', data=bytes(body), headers={'Content-Type': content_type})
    with lock:
        # Empty live frames are only an audit request, never accuracy evidence.
        output = job['output']
        receipt = read_json(output / 'isolated-project/artifacts/serving/deployment.json') if output else None
        manifest = receipt['model'] if receipt else {}
        expected = hashlib.sha256(json.dumps(manifest.get('inference'), sort_keys=True).encode()).hexdigest()
        eligible = (manifest.get('inference', {}).get('confidence') == 1.0 and
                    value.get('inference_sha256') == expected and
                    value.get('checkpoint_sha256') == manifest.get('checkpoint_sha256'))
        job['misses'] = job['misses'] + 1 if eligible and not any(d['class_name'] == 'helmet' for d in value['detections']) else 0
        value['audit_requested'] = request_audit() if job['misses'] >= 3 else False
    return value


@app.post('/predict')
async def predict(request: Request):
    if request.headers.get('content-type','').split(';')[0].strip() not in ('image/png','image/jpeg'):
        raise HTTPException(415,'Select a PNG or JPEG image')
    body=bytearray()
    async for chunk in request.stream():
        if len(body)+len(chunk)>MAX_BYTES:
            raise HTTPException(413,'Image must be under 4 MB')
        body.extend(chunk)
    value=await run_in_threadpool(demo_selection.predict,bytes(body),ROOT)
    request.state.demo_identity=(value['demo_model_id'],str(value['model_version']))
    return value


@app.post('/demo/start')
async def start_demo(request: Request):
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > 1024:
            raise HTTPException(413, 'Control request too large')
        body.extend(chunk)
    try:
        control = json.loads(body)
        mode = control['mode']
        candidate_version = control.get('candidate_version', '3')
    except (ValueError, KeyError, TypeError):
        raise HTTPException(400, 'Choose quality or outage')
    if mode not in ('quality', 'outage'):
        raise HTTPException(400, 'Choose quality or outage')
    if candidate_version not in ('2', '3', '4'):
        raise HTTPException(400, 'Choose trained model 2, 3 or 4')
    with lock:
        if active_job() or (ROOT / 'artifacts/serving/deployment.lock').exists():
            raise HTTPException(409, 'A deployment/demo is already running; wait for restoration')
        if job['output'] and job['process'] is None and not read_json(job['output'] / 'result.json'):
            raise HTTPException(409, 'Prior demo has no final receipt. Inspect its process and deployment before starting another.')
        output = launch_demo(mode,candidate_version)
    return {'status': 'starting', 'report_directory': str(output)}


@app.post('/demo/audit')
def audit_now():
    with lock:
        if not request_audit():
            raise HTTPException(409, 'Wait for the quality-demo candidate to become live')
    return {'status': 'audit_requested'}


@app.post('/automatic/test-outage')
def automatic_test_outage():
    with lock:
        progress = read_json(job['output']/'progress.json') if job['output'] else None
        if not active_job() or not progress or progress.get('stage') != 'automatic_monitoring':
            raise HTTPException(409,'Wait for automatic protection to be active on the candidate')
        (job['output']/'request-outage').touch(exist_ok=True)
    return {'status':'test_outage_requested','note':'The watcher decides recovery independently.'}


@app.get('/demo/status')
def demo_status():
    with lock:
        process, output = job['process'], job['output']
    if not output:
        return {'running': False, 'progress': {'stage': 'idle', 'message': 'Live detection is ready. Choose a demonstration when needed.'}}
    progress = read_json(output / 'progress.json') or {'stage': 'starting', 'message': 'Preparing isolated registry'}
    result = read_json(output / 'result.json')
    audit = read_json(output / 'audit.json')
    log = output.with_suffix('.log')
    with log.open('rb') as stream:
        offset = max(0, log.stat().st_size - 20000)
        stream.seek(offset)
        if offset:
            stream.readline()  # Drop the initial partial line/UTF-8 sequence.
        tail = stream.read().decode('utf-8', errors='replace')
    tail = '\n'.join(line for line in tail.replace('\r', '\n').splitlines()
                     if line.strip() and not line.startswith('Downloading artifacts:'))[-6000:]
    if process is not None:
        exit_code = process.poll()
    elif result:
        exit_code = 0 if result['status'] == 'passed' else 1
    else:
        exit_code = None if (ROOT / 'artifacts/serving/deployment.lock').exists() else 1
    if exit_code not in (None, 0):
        progress = {'stage': 'failed', 'message': 'Demo failed. Read the log and check API health; do not claim recovery.'}
    adopted = process is None and active_job()
    if adopted:
        exit_code = None
    if process is None and not result and not adopted:
        exit_code = 1  # A leftover lock is not evidence of a living watcher.
        progress = {'stage': 'needs_check', 'message': 'Previous test has no final result. Check its process and detector before starting another.'}
    watch = read_json(output/'watch.json')
    recovery = read_json(output/'automatic-recovery.json')
    if progress.get('stage') == 'automatic_recovered':
        progress = {**progress, 'message':'Automatic recovery completed. The fallback is selected; see live availability above. There is no earlier model for another rollback.'}
    return {'running': exit_code is None, 'exit_code': exit_code, 'progress': progress, 'audit': audit,
            'automatic':progress.get('mode')=='automatic','watch':watch,'automatic_recovery':recovery,
            'source_versions': result.get('source_versions') if result else ['1', job['candidate_version']],
            'result': {'status': result['status'], 'recovered_map50_95': result.get('recovered_map50_95'),
                       'quality_approval': False, 'original_api_restored': 'original_api_restored' in result} if result else None,
            'report_directory': str(output), 'log': tail}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8767)
