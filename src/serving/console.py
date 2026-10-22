"""Loopback-only presentation console; inference stays in the existing Docker API."""

import hashlib
import json
import os
from pathlib import Path
import secrets
import re
import subprocess
import sys
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware
import requests

ROOT = Path(__file__).resolve().parents[2]
API = 'http://127.0.0.1:8000'
MAX_BYTES = 4 * 1024 * 1024
TOKEN = secrets.token_urlsafe(32)
app = FastAPI(title='Visual analytics live console', docs_url=None, redoc_url=None, openapi_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=['127.0.0.1', 'localhost'])
lock = Lock()
job = {'process': None, 'output': None, 'misses': 0}
JOB_FILE = ROOT / 'reports/live-console-job.json'
if JOB_FILE.exists():
    try:
        saved_name = json.loads(JOB_FILE.read_text())['directory']
        if re.fullmatch(r'live-demo-[0-9a-f]{12}', saved_name):
            job['output'] = ROOT / 'reports' / saved_name
    except (ValueError, KeyError, TypeError):
        pass


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


@app.get('/health')
def health():
    return call_api('/health')


@app.get('/sample')
def sample():
    return Response((ROOT / 'dataset/images/hard_hat_workers1015.png').read_bytes(), media_type='image/png')


def request_audit():
    output = job['output']
    progress = read_json(output / 'progress.json') if output else None
    if progress and progress['mode'] == 'quality' and progress['stage'] in ('candidate_live', 'awaiting_quality_check'):
        (output / 'request-audit').touch(exist_ok=True)
        return True
    return False


@app.post('/predict')
async def predict(request: Request):
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


@app.post('/demo/start')
async def start_demo(request: Request):
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > 1024:
            raise HTTPException(413, 'Control request too large')
        body.extend(chunk)
    try:
        mode = json.loads(body)['mode']
    except (ValueError, KeyError, TypeError):
        raise HTTPException(400, 'Choose quality or outage')
    if mode not in ('quality', 'outage'):
        raise HTTPException(400, 'Choose quality or outage')
    with lock:
        if (job['process'] and job['process'].poll() is None) or (ROOT / 'artifacts/serving/deployment.lock').exists():
            raise HTTPException(409, 'A deployment/demo is already running; wait for restoration')
        if job['output'] and job['process'] is None and not read_json(job['output'] / 'result.json'):
            raise HTTPException(409, 'Prior demo has no final receipt. Inspect its process and deployment before starting another.')
        output = ROOT / 'reports' / ('live-demo-' + uuid4().hex[:12])
        log = output.with_suffix('.log')
        with log.open('wb') as stream:
            process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'scripts/demo_rollback.py'),
                '--output', str(output), '--mode', mode], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        job.update(process=process, output=output, misses=0)
        from src.serving.deploy import write_receipt
        write_receipt(JOB_FILE, {'directory': output.name})
    return {'status': 'starting', 'report_directory': str(output)}


@app.post('/demo/audit')
def audit_now():
    with lock:
        if not request_audit():
            raise HTTPException(409, 'Wait for the quality-demo candidate to become live')
    return {'status': 'audit_requested'}


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
    return {'running': exit_code is None, 'exit_code': exit_code, 'progress': progress, 'audit': audit,
            'result': {'status': result['status'], 'recovered_map50_95': result.get('recovered_map50_95'),
                       'quality_approval': False, 'original_api_restored': 'original_api_restored' in result} if result else None,
            'report_directory': str(output), 'log': tail}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8767)
