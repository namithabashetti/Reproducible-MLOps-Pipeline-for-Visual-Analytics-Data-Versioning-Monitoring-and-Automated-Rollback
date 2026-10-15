"""Validate the existing local pipeline and record acceptance gaps without deploying or training."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import requests
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data.validate import validate_dataset
from src.monitoring.audit import labeled_audit
from src.monitoring.drift import collect as collect_drift, export_report
from src.monitoring.health import capture_latency_baseline, collect, evaluate, save
from src.monitoring.rollback import current_container, validate_decision
from src.tracking.mlflow_utils import file_hash, local_store, read_completed_run, verify_registration


def metadata(directory, project):
    digest = hashlib.sha256()
    files = sorted(p for p in directory.rglob('*') if p.is_file())
    for path in files:
        stat = path.stat()
        digest.update(f'{path.relative_to(project).as_posix()}\0{stat.st_size}\0{stat.st_mtime_ns}\n'.encode())
    return {'files': len(files), 'metadata_sha256': digest.hexdigest()}


def wait_until(target):
    while time.time() < target:
        time.sleep(min(5, target - time.time()))


def dashboard_check(session, config):
    base = f"http://127.0.0.1:{config['monitoring']['grafana_port']}"
    response = session.get(base + '/api/dashboards/uid/visual-analytics-operations', timeout=10)
    response.raise_for_status()
    live = response.json()
    source = json.loads((PROJECT / 'configs/monitoring/grafana/dashboard.json').read_text())
    if any(live['meta'][key] for key in ('canEdit', 'canSave', 'canAdmin')) or \
            [(p['id'], p['targets']) for p in source['panels']] != \
            [(p['id'], p['targets']) for p in live['dashboard']['panels']]:
        raise ValueError('Grafana dashboard differs from the provisioned read-only configuration')
    results = []
    for panel in source['panels']:
        expression = panel['targets'][0]['expr']
        response = session.get(f"http://127.0.0.1:{config['monitoring']['prometheus_port']}/api/v1/query",
                               params={'query': expression}, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data['status'] != 'success' or not data['data']['result'] or data.get('warnings'):
            raise ValueError(f"No usable Prometheus data for {panel['title']}")
        if any(not math.isfinite(float(item['value'][1])) for item in data['data']['result']):
            raise ValueError(f"Non-finite monitoring value for {panel['title']}")
        now_ms = int(time.time() * 1000)
        response = session.post(base + '/api/ds/query', json={
            'queries': [{'refId': 'A', 'datasource': {'type': 'prometheus', 'uid': 'visual-analytics-prometheus'},
                         'expr': expression, 'instant': True, 'range': False, 'format': 'time_series',
                         'intervalMs': 10000, 'maxDataPoints': 600}],
            'from': str(now_ms - 900000), 'to': str(now_ms)}, timeout=20)
        response.raise_for_status()
        result = response.json()['results']['A']
        if result.get('status', 200) != 200 or not result.get('frames'):
            raise ValueError(f"Grafana query failed for {panel['title']}")
        results.append({'title': panel['title'], 'series': len(data['data']['result']), 'grafana_query': 'passed'})
    return {'panels': results, 'read_only': True, 'verification': 'dashboard API and panel queries; not visual browser review'}


def validate(project, output, reference_path):
    config = yaml.safe_load((project / 'configs/config.yaml').read_text())
    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=project, text=True).strip(),
              'source_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=project)),
              'validator_sha256': file_hash(Path(__file__)), 'checks': {}, 'acceptance': {}}
    checks = report['checks']
    try:
        raw = project / config['dataset']['raw_path']
        before = metadata(raw, project)
        expected_metadata = json.loads((project / 'reports/phase3-raw-metadata.json').read_text())
        if before['metadata_sha256'] != expected_metadata['metadata_sha256']:
            raise ValueError('Original dataset metadata differs from the preserved baseline')
        print('Validating the existing dataset and DVC state.', flush=True)
        data = validate_dataset(raw, set(config['dataset']['classes'].values()))
        save(output / 'dataset.json', data)
        expected = json.loads((project / 'reports/dataset-validation.json').read_text())
        pointer = yaml.safe_load((project / 'dataset.dvc').read_text())['outs'][0]
        if data['content_manifest_sha256'] != expected['content_manifest_sha256'] or \
                data['files'] != pointer['nfiles'] or data['total_bytes'] != pointer['size']:
            raise ValueError('Original dataset content/pointer mismatch')
        status = json.loads(subprocess.check_output([sys.executable, '-m', 'dvc', 'status', '--json'], cwd=project, text=True))
        if status:
            raise ValueError('DVC reports changed dependencies or outputs')
        checks['data'] = {'raw_dvc_md5': pointer['md5'], 'content_sha256': data['content_manifest_sha256'],
                          'dvc_status': status, 'provenance': config['dataset']['provenance_status']}
        restored = json.loads((project / 'reports/gdrive-restore.json').read_text())
        if not restored['verified'] or restored['source'] != 'google_drive' or \
                restored['dvc_md5'] != pointer['md5'] or restored['content_manifest_sha256'] != data['content_manifest_sha256']:
            raise ValueError('Prior Drive recovery evidence does not match this dataset')
        checks['storage'] = {'scope': 'historical content-verified Drive recovery; no new remote transfer', 'evidence': restored}
        processed = json.loads((project / 'reports/phase3-validation.json').read_text())
        if not processed['verified'] or processed['raw_content_manifest_sha256'] != data['content_manifest_sha256']:
            raise ValueError('Preprocessing evidence does not match this dataset')
        checks['preprocessing'] = {'scope': 'prior full audit plus current clean DVC dependency/output state', 'evidence': processed}

        print('Checking completed training, MLflow artifacts and the deployed checkpoint.', flush=True)
        directory = project / 'artifacts/training/baseline'
        lineage, metrics, _, _, _, _ = read_completed_run(directory)
        registration = json.loads((directory / 'mlflow.json').read_text())
        client, _, registry = local_store(project)
        verified = verify_registration(client, registration['model_name'], registration['model_version'], directory)
        lock = yaml.safe_load((project / 'dvc.lock').read_text())['stages']
        if lineage['dataset_dvc_md5'] != pointer['md5'] or \
                lineage['splits_dvc_md5'] != lock['split']['outs'][0]['md5'] or \
                lineage['processed_dvc_md5'] != lock['preprocess']['outs'][0]['md5']:
            raise ValueError('Training lineage does not match current data outputs')
        checks['training'] = {'scope': 'verified completed run; no new training or final-test evaluation',
                              'metrics': metrics, 'registry': verified}
        receipt_path = project / 'artifacts/serving/deployment.json'
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes)
        if receipt['model']['checkpoint_sha256'] != verified['checkpoint_sha256'] or \
                receipt['model']['model_version'] != verified['model_version'] or \
                file_hash(Path(receipt['staged_directory']) / 'best.pt') != verified['checkpoint_sha256']:
            raise ValueError('Deployed checkpoint does not match the verified registry version')
        docker = str(Path(os.environ['LOCALAPPDATA']) / 'Programs/DockerDesktop/resources/bin/docker.exe')
        state = current_container(docker, receipt)
        if not state['State']['Running']:
            raise ValueError('Start the existing API and monitoring containers before validation')
        api = f"http://127.0.0.1:{receipt.get('serving', config['serving'])['port']}"
        session = requests.Session()
        session.trust_env = False
        with session:
            response = session.get(api + '/health', timeout=15)
            response.raise_for_status()
            if response.json() != {'status': 'ready', **receipt['model']}:
                raise ValueError('Live model/settings differ from deployment receipt')
            checks['serving'] = {'container_id': receipt['container_id'], 'model': receipt['model'], 'ready': True}
            audit = labeled_audit(project, output / 'audit')
            reference = json.loads(reference_path.read_text())
            width = config['health']['window_seconds']
            if width != 120:
                raise ValueError('This demonstration schedule is calibrated for the approved 120-second windows')
            start = (int(time.time()) // width + 1) * width
            print(f'Two-window serving check starts at UTC epoch {start}.', flush=True)
            wait_until(start + 20)
            names = json.loads((project / 'data/splits.json').read_text())['splits']['validation'][:10]
            for index in range(110):
                response = session.post(api + '/predict', data=(raw / 'images' / f'{names[index % 10]}.png').read_bytes(),
                                        headers={'Content-Type': 'image/png'}, timeout=30)
                response.raise_for_status()
                if response.json()['checkpoint_sha256'] != receipt['model']['checkpoint_sha256']:
                    raise ValueError('Model changed during predictions')
            for _ in range(3):
                response = session.post(api + '/predict', data=b'corrupt', headers={'Content-Type': 'image/png'}, timeout=10)
                if response.status_code != 422:
                    raise ValueError('Expected client validation rejection')
            if time.time() >= start + 110:
                raise ValueError('First traffic batch exceeded its window; rerun when the laptop is idle')
            print('First window: 110 successful predictions and 3 client rejections.', flush=True)
            wait_until(start + 140)
            current = collect_drift(project, 'current', output / 'current.json')
            if time.time() >= start + 230:
                raise ValueError('Monitoring replay exceeded its window; rerun when the laptop is idle')
            drift = export_report(reference, current, config['drift'], output / 'drift')
            if drift['status'] != 'available':
                raise ValueError('Drift evidence is incomplete')
            print('Fresh current-pool replay and Evidently report complete.', flush=True)
            wait_until(start + 245)
            evidence = collect(project, config['health'])
            if [w.get('samples') for w in evidence['windows']] != [110., float(len(current['observations']))]:
                raise ValueError('Traffic windows are incomplete or include concurrent requests')
            baseline = capture_latency_baseline(evidence, config['health'])
            save(output / 'latency-baseline.json', baseline)
            evidence.update(drift=drift, audit=audit, latency_baseline=baseline)
            decision = evaluate(evidence, config['health'], config['drift'])
            decision['evidence'] = evidence
            save(output / 'health.json', decision)
            if decision['status'] == 'unhealthy':
                raise ValueError('Unexpected unhealthy result during normal-operation validation; inspect health.json')
            try:
                validate_decision(decision, receipt, current_container(docker, receipt), config, time.time())
            except ValueError as error:
                checks['rollback_gate'] = {'status': 'observe', 'reason': str(error), 'switch_attempted': False}
            else:
                raise ValueError('Rollback gate unexpectedly accepted normal-operation evidence')
            checks['monitoring'] = dashboard_check(session, config)
        checks['drift'] = {'status': drift['status'], 'drift_detected': drift['drift_detected'],
                           'reference_sha256': file_hash(reference_path), 'reference_path': str(reference_path),
                           'current_observations': len(current['observations']), 'report': 'drift/summary.json'}
        checks['health'] = {'status': decision['status'], 'reasons': decision['reasons'],
                            'traffic_counts': [w['samples'] for w in evidence['windows']],
                            'excluded_client_errors': 3, 'latency_baseline_p95_seconds': baseline['p95_seconds']}
        aliases = dict(client.get_registered_model(registry['model_name']).aliases)
        if receipt_bytes != receipt_path.read_bytes() or aliases != verified['registry_aliases'] or \
                current_container(docker, receipt)['State']['StartedAt'] != state['State']['StartedAt']:
            raise ValueError('Deployment, process or registry aliases changed during validation')
        after = metadata(raw, project)
        if after != before:
            raise ValueError('Original dataset metadata changed during validation')
        checks['preservation'] = {'dataset': after, 'deployment_unchanged': True, 'aliases_unchanged': True}
        criteria = {
            'AC-01': ('partial', 'DVC/content verified; exact Kaggle source/version remains unverified'),
            'AC-02': ('verified', 'Approved Google Drive amendment; historical full recovery matches current content'),
            'AC-03': ('verified', 'Full preprocessing audit and current clean DVC outputs'),
            'AC-04': ('verified', 'Completed 30-epoch YOLO/PyTorch run and verified source/checkpoint lineage'),
            'AC-05': ('verified', 'Finite recorded validation precision, recall and mAP; person quality remains weak'),
            'AC-06': ('verified', 'Real MLflow run, metrics, epoch history and retrieved artifact hashes'),
            'AC-07': ('verified', 'Explicit registered model version; registration does not confer stable approval'),
            'AC-09': ('verified', 'Actual Docker/FastAPI predictions match the registered checkpoint'),
            'AC-10': ('verified', 'Fresh Prometheus metrics and provisioned Grafana panel API queries'),
            'AC-11': ('verified', 'Fresh controlled current-pool replay against a preserved reference in Evidently'),
            'AC-12': ('verified', 'PSI and KS results with fixed documented population and threshold rules'),
            'AC-13': ('partial', 'Fresh combined decision and prior decision/outage tests; real stable quality comparison unavailable'),
            'AC-14': ('blocked', 'No live approved-predecessor switch demonstrated; stable target review is required'),
            'AC-15': ('blocked', 'Live monitoring works, but monitoring after an actual approved-model rollback is unverified')}
        ci = json.loads((project / 'reports/phase11-github-run.json').read_text())
        if ci['run']['conclusion'] != 'success':
            raise ValueError('Latest recorded implementation CI did not pass')
        checks['ci'] = {'scope': 'previous implementation validation; Phase 12 CI recorded separately', 'run': ci['run']}
        criteria['AC-08'] = ('verified', 'GitHub validation plus the recorded Phase 9 experimental deployment workflow')
        report['acceptance'] = {key: {'status': status, 'evidence': explanation}
                                for key, (status, explanation) in sorted(criteria.items())}
        report['status'] = 'completed_with_acceptance_gaps'
        report['limitations'] = ['This validates the existing baseline; it does not train, approve, deploy or roll back a model.',
                                 'Monitoring replay and initial thresholds are a local demonstration, not production validation.',
                                 'Saved health/drift reports expire; historical evidence must not authorize later actions.']
        print('Validation completed with explicit provenance, stable-quality and rollback acceptance gaps.', flush=True)
    except Exception as error:
        report.update(status='failed', error_type=type(error).__name__, reason=str(error))
        raise
    finally:
        report['completed_at'] = datetime.now(timezone.utc).isoformat()
        save(output / 'validation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory under reports/')
    parser.add_argument('--reference', type=Path, required=True, help='Preserved reference monitoring observations')
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((PROJECT / 'reports').resolve()):
        parser.error('Validation output must stay under the project reports directory')
    output.mkdir(parents=True, exist_ok=False)
    validate(PROJECT, output, args.reference.resolve())


if __name__ == '__main__':
    main()
