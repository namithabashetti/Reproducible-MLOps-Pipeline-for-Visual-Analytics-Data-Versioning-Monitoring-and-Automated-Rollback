"""Demonstrate real automatic recovery in a copied registry, then restore the local API.

The copied registry's aliases are test fixtures, never model-quality approvals.
Uses existing weights, images and Docker runtime; no training or remote transfer.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
from threading import Event, Thread
import time

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.monitoring.rollback import recovery_images, run, watch
from src.serving.deploy import deploy, deployment_lock, stage_model, write_receipt
from src.tracking.mlflow_utils import local_store


def demonstrate(output, docker):
    output = output.resolve()
    if not output.is_relative_to(ROOT / 'reports') or output == ROOT / 'reports':
        raise ValueError('Use a new demonstration directory under reports/')
    output.mkdir(parents=True, exist_ok=False)
    store = output / 'isolated-project'
    (store / 'configs').mkdir(parents=True)
    (store / 'data').mkdir()
    (store / 'dataset/images').mkdir(parents=True)
    (store / 'artifacts/mlflow').mkdir(parents=True)
    config = yaml.safe_load((ROOT / 'configs/config.yaml').read_text())
    original_client, _, registry = local_store(ROOT)
    original_aliases = dict(original_client.get_registered_model(registry['model_name']).aliases)
    original_receipt = json.loads((ROOT / 'artifacts/serving/deployment.json').read_text())
    if original_receipt['model']['deployment_mode'] != 'experimental':
        raise ValueError('This demonstration only temporarily replaces an experimental local API')
    write_receipt(output / 'original-deployment.json', original_receipt)
    config['mlflow']['model_name'] = registry['model_name'] + '-rollback-demo'
    (store / 'configs/config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copy2(ROOT / 'data/splits.json', store / 'data/splits.json')
    for name, body, _ in recovery_images(ROOT, config['rollback']['recovery_prediction_successes']):
        (store / 'dataset/images' / (name + '.png')).write_bytes(body)
    source_database = Path(original_client.tracking_uri[len('sqlite:///'):])
    with sqlite3.connect(source_database.as_uri() + '?mode=ro', uri=True) as source:
        with sqlite3.connect(store / 'artifacts/mlflow/mlflow.db') as destination:
            source.backup(destination)
    client, _, demo_registry = local_store(store)
    name = demo_registry['model_name']
    client.create_registered_model(name, description='Isolated operational recovery fixture; NOT quality-approved.')
    versions = []
    for original_version in ('1', '3'):
        original = original_client.get_model_version(registry['model_name'], original_version)
        versions.append(client.create_model_version(name, original.source, run_id=original.run_id,
            tags={**original.tags, 'approval_status': 'demo_fixture_only', 'source_model_version': original_version}))
    client.set_registered_model_alias(name, 'stable', versions[0].version)
    client.set_registered_model_alias(name, 'previous_stable', versions[0].version)
    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'scope': 'isolated operational integration demonstration',
              'quality_approval': False, 'main_registry_aliases_before': original_aliases,
              'source_versions': ['1', '3'], 'demo_model_name': name, 'status': 'running',
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    stop, failures = Event(), []
    thread = None
    changed_service = False
    with deployment_lock(ROOT):
        try:
            print('Deploying the verified predecessor in the isolated demo registry.', flush=True)
            staged, manifest = stage_model(ROOT, store, versions[0].version)
            changed_service = True
            with deployment_lock(store):
                predecessor = deploy(store, staged, manifest, original_receipt['image_id'], docker)
            report['predecessor'] = predecessor
            print('Deploying candidate and capturing the real predecessor rollback target.', flush=True)
            staged, manifest = stage_model(ROOT, store, versions[1].version, experimental=True)
            with deployment_lock(store):
                candidate = deploy(store, staged, manifest, original_receipt['image_id'], docker)
            assert candidate['rollback_target']['model']['checkpoint_sha256'] == predecessor['model']['checkpoint_sha256']
            report['candidate'] = candidate
            # Keep the real production grace period; do not fabricate timestamps or health evidence.
            state = json.loads(subprocess.check_output([docker, 'inspect', candidate['container_id']], text=True))[0]
            started = datetime.fromisoformat(state['State']['StartedAt'].replace('Z', '+00:00')).timestamp()
            print('Waiting for the configured startup grace period before the fault test.', flush=True)
            while time.time() < started + config['health']['startup_grace_seconds'] + 1:
                time.sleep(1)
            def controller():
                try:
                    watch(store, docker, output / 'watch.json', 10, stop=stop)
                except Exception as error:
                    failures.append(repr(error))
            thread = Thread(target=controller, name='rollback-demo-watch')
            thread.start()
            deadline = time.monotonic() + 90
            while not (output / 'watch.json').exists():
                if failures or time.monotonic() > deadline:
                    raise RuntimeError(f'Initial watcher check failed: {failures}')
                time.sleep(1)
            initial = json.loads((output / 'watch.json').read_text())
            assert initial['status'] == 'observe', initial
            assert initial['health']['signals']['readiness']['status'] == 'healthy'
            report['healthy_watch_check'] = initial
            print('Watcher is active. Stopping the candidate container to inject a real outage.', flush=True)
            subprocess.run([docker, 'stop', '--time', '5', candidate['container_id']], check=True, capture_output=True)
            report['fault_injected_at'] = datetime.now(timezone.utc).isoformat()
            deadline = time.monotonic() + 240
            while time.monotonic() < deadline:
                result = json.loads((output / 'watch.json').read_text())
                if result['status'] == 'recovered':
                    report['rollback'] = result
                    break
                if failures or result['status'] == 'failed':
                    raise RuntimeError(f'Automatic recovery failed: {failures or result}')
                time.sleep(1)
            else:
                raise TimeoutError('Automatic recovery did not complete before the deadline')
            stop.set()
            thread.join()
            recovery = report['rollback']['recovery']
            assert len(recovery['readiness']) == 3 and len(recovery['predictions']) == 20
            assert recovery['monitoring_resumed'] is True
            assert report['rollback']['restored_deployment']['model']['checkpoint_sha256'] == predecessor['model']['checkpoint_sha256']
            report['followup_check'] = run(store, docker)
            assert report['followup_check']['status'] == 'unavailable'  # No rollback ping-pong.
            report['status'] = 'passed'
            print('Automatic rollback passed: 3 readiness checks, 20 predictions, fresh Prometheus identity.', flush=True)
        except Exception as error:
            report.update(status='failed', error=repr(error))
            raise
        finally:
            stop.set()
            if thread is not None:
                thread.join()
            if changed_service:
                print('Restoring the original experimental API and main deployment receipt.', flush=True)
                current = subprocess.run([docker, 'inspect', 'visual-analytics-api'], capture_output=True, text=True)
                if current.returncode == 0:
                    current = json.loads(current.stdout)[0]
                    receipt_path = store / 'artifacts/serving/deployment.json'
                    known = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
                    if known is None or current['Id'] != known['container_id']:
                        raise RuntimeError('API changed outside this demonstration; refusing to remove it')
                    subprocess.run([docker, 'rm', '-f', current['Id']], check=True, capture_output=True)
                restored = deploy(ROOT, Path(original_receipt['staged_directory']), original_receipt['model'],
                                  original_receipt['image_id'], docker)
                report['original_api_restored'] = restored
            report['main_registry_aliases_after'] = dict(original_client.get_registered_model(registry['model_name']).aliases)
            assert report['main_registry_aliases_after'] == original_aliases
            report['completed_at'] = datetime.now(timezone.utc).isoformat()
            write_receipt(output / 'result.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--docker', default=str(Path(os.environ['LOCALAPPDATA']) / 'Programs/DockerDesktop/resources/bin/docker.exe'))
    args = parser.parse_args()
    result = demonstrate(args.output, args.docker)
    print(json.dumps({'status': result['status'], 'quality_approval': False, 'output': str(args.output)}))
