"""Rollback safety and recovery contracts with isolated files and controlled external boundaries."""

from contextlib import ExitStack
from copy import deepcopy
import hashlib
from io import StringIO
import json
from pathlib import Path
import subprocess
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import yaml

from src.monitoring.health import evaluate, iso, model_identity
from src.monitoring.rollback import prepare_target, recovery_images, run, validate_decision, verify_recovery
from src.serving.deploy import deploy, deployment_lock, snapshot_target, stage_model


class RollbackTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = yaml.safe_load((Path(__file__).resolve().parents[1] / 'configs/config.yaml').read_text())
        (self.root / 'configs').mkdir()
        (self.root / 'configs/config.yaml').write_text(yaml.safe_dump(self.config))
        self.directory = self.root / 'artifacts/serving'
        self.directory.mkdir(parents=True)
        from src.monitoring.health import timestamp
        self.now = timestamp(iso(time.time()))
        self.model = {'model_name': 'hard-hat-detector', 'model_version': '2', 'run_id': 'candidate',
                      'checkpoint_sha256': 'candidate', 'bundle_sha256': 'candidate_bundle',
                      'dataset_dvc_md5': 'raw', 'deployment_mode': 'experimental', 'classes': {'0': 'helmet'},
                      'inference': {'device': 'cpu', 'confidence': .25}}
        self.target_model = {**self.model, 'model_version': '1', 'run_id': 'stable',
                             'checkpoint_sha256': 'stable', 'bundle_sha256': 'stable_bundle',
                             'deployment_mode': 'approved', 'inference': {'device': 'cpu', 'confidence': .3}}
        self.target = {'model': self.target_model, 'source_container_id': 'b' * 64,
                       'image_id': 'sha256:' + '1' * 64, 'staged_directory': str(self.directory / 'v1'),
                       'serving': {'host_bind_address': '127.0.0.1', 'port': 8000}, 'captured_at': iso(self.now - 2000)}
        self.receipt = {'status': 'ready', 'container_id': 'a' * 64, 'image_id': 'sha256:' + '2' * 64,
                        'model': self.model, 'rollback_target': self.target, 'serving': self.target['serving']}
        self.receipt_path = self.directory / 'deployment.json'
        self.receipt_path.write_text(json.dumps(self.receipt))
        self.state = {'Id': self.receipt['container_id'], 'Image': self.receipt['image_id'],
                      'Config': {'Labels': {'visual-analytics.project': 'hard-hat-detector'}},
                      'State': {'Running': False, 'StartedAt': iso(self.now - 1000)}}
        self.evidence = {'identity': model_identity(self.model), 'deployment_id': self.receipt['container_id'],
                         'runtime_started': self.now - 1000, 'windows': [],
                         'readiness': [{'at': self.now - age, 'ready': False} for age in (20, 10, 0)]}
        # Match ISO microsecond precision of Docker timestamps in these fixtures.
        from src.monitoring.health import timestamp
        self.evidence['runtime_started'] = timestamp(self.state['State']['StartedAt'])
        self.report = evaluate(self.evidence, self.config['health'], self.config['drift'], self.now)
        self.report['evidence'] = self.evidence

    def mocks(self):
        stack = ExitStack()
        self.addCleanup(stack.close)
        mocked = {}
        for name, value in [('current_container', self.state), ('prepare_target', self.target),
                            ('recovery_images', [('image', b'png', 'sha')] * 20),
                            ('deploy', {'container_id': 'c' * 64}), ('verify_recovery', {})]:
            mocked[name] = stack.enter_context(patch('src.monitoring.rollback.' + name, return_value=deepcopy(value)))
        client = MagicMock()
        client.get_registered_model.return_value = SimpleNamespace(aliases={'stable': '1'})
        mocked['registry'] = client
        stack.enter_context(patch('src.monitoring.rollback.local_store', return_value=(client, None, {'model_name': 'hard-hat-detector'})))
        return mocked

    def test_recompute_fresh_decision_and_reject_stale_wrong_identity_restart_or_policy(self):
        self.assertEqual(validate_decision(self.report, self.receipt, self.state, self.config, self.now)['status'], 'unhealthy')
        for case in ('expired', 'future', 'identity', 'restart', 'policy', 'claim_only'):
            report, state = deepcopy(self.report), deepcopy(self.state)
            if case == 'expired': report['valid_until'] = iso(self.now - 1)
            if case == 'future': report['generated_at'] = iso(self.now + 30)
            if case == 'identity': report['evidence']['deployment_id'] = 'different'
            if case == 'restart': state['State']['StartedAt'] = iso(self.now - 1)
            if case == 'policy': report['policy'] = {}
            if case == 'claim_only': report['evidence']['readiness'][-1]['ready'] = True
            with self.subTest(case=case), self.assertRaises(ValueError):
                validate_decision(report, self.receipt, state, self.config, self.now)

    def test_drift_alone_and_missing_quality_do_not_switch(self):
        mocked = self.mocks()
        report = deepcopy(self.report)
        report['evidence']['readiness'][-1]['ready'] = True
        report['evidence']['drift'] = {'drift_detected': True}
        result = run(self.root, 'docker', report=report)
        self.assertEqual(result['status'], 'observe')
        mocked['deploy'].assert_not_called()
        self.assertFalse(list((self.directory / 'rollback').glob('attempt-*')))

    def test_target_requires_approved_recorded_version_and_pinned_image(self):
        with patch('src.monitoring.rollback.subprocess.check_output', return_value=self.target['image_id']), \
                patch('src.monitoring.rollback.stage_model', return_value=(Path(self.target['staged_directory']), self.target_model)) as stage:
            result = prepare_target(self.root, self.receipt, 'docker')
            self.assertEqual(result['model']['model_version'], '1')
            self.assertEqual(stage.call_args.args[2], '1')
            self.assertEqual(stage.call_args.kwargs['restore_manifest']['inference']['confidence'], .3)
            for target in (None, {**self.target, 'image_id': 'latest'},
                           {**self.target, 'model': self.model}):
                with self.assertRaises(ValueError):
                    prepare_target(self.root, {**self.receipt, 'rollback_target': target}, 'docker')
        with patch('src.monitoring.rollback.subprocess.check_output', side_effect=subprocess.CalledProcessError(1, 'docker')):
            with self.assertRaises(subprocess.CalledProcessError):
                prepare_target(self.root, self.receipt, 'docker')

    def test_success_uses_shared_deploy_and_durable_claim_precedes_switch(self):
        mocked = self.mocks()
        path = self.directory / 'rollback' / f"attempt-{self.receipt['container_id']}.json"
        def switch(*args, **kwargs):
            self.assertEqual(json.loads(path.read_text())['status'], 'started')
            self.assertEqual(kwargs['recovery_target']['model'], self.target_model)
            self.assertTrue((self.directory / 'deployment.lock').exists())
            return {'container_id': 'c' * 64}
        mocked['deploy'].side_effect = switch
        result = run(self.root, 'docker', report=self.report)
        self.assertEqual(result['status'], 'recovered')
        self.assertEqual(json.loads(path.read_text())['status'], 'recovered')
        self.assertEqual(run(self.root, 'docker', report=self.report)['status'], 'blocked')
        self.assertEqual(mocked['deploy'].call_count, 1)
        self.assertFalse((self.directory / 'deployment.lock').exists())

    def test_failed_deploy_or_recovery_cannot_retry_or_swap_back(self):
        mocked = self.mocks()
        mocked['verify_recovery'].side_effect = RuntimeError('prediction failed')
        result = run(self.root, 'docker', report=self.report)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['reason'], 'prediction failed')
        self.assertEqual(run(self.root, 'docker', report=self.report)['status'], 'blocked')
        mocked['deploy'].assert_called_once()
        self.assertEqual(json.loads(self.receipt_path.read_text()), self.receipt)

    def test_approval_revoked_during_collection_prevents_switch(self):
        mocked = self.mocks()
        mocked['registry'].get_registered_model.return_value = SimpleNamespace(aliases={})
        result = run(self.root, 'docker', report=self.report)
        self.assertEqual(result['status'], 'observe')
        self.assertIn('approval changed', result['reason'])
        mocked['deploy'].assert_not_called()

    def test_quality_baseline_change_prevents_switch(self):
        mocked = self.mocks()
        evidence = deepcopy(self.evidence)
        evidence['readiness'][-1]['ready'] = True
        baseline = {'model_name': 'hard-hat-detector', 'model_version': '1', 'checkpoint_sha256': 'stable'}
        evidence.update(stable_baseline=baseline, split_sha256='split', audit={
            'status': 'available', 'generated_at': iso(self.now), 'candidate': evidence['identity'],
            'baseline': baseline, 'images': 200, 'split_sha256': 'split', 'baseline_batch_sha256': 'same',
            'candidate_batch_sha256': 'same', 'baseline_map50_95': .8, 'candidate_map50_95': .6})
        report = evaluate(evidence, self.config['health'], self.config['drift'], self.now)
        report['evidence'] = evidence
        self.assertEqual(report['reasons'], ['quality'])
        mocked['registry'].get_model_version_by_alias.return_value = SimpleNamespace(version='3', tags={'checkpoint_sha256': 'new'})
        result = run(self.root, 'docker', report=report)
        self.assertEqual(result['status'], 'observe')
        self.assertIn('quality baseline changed', result['reason'])
        mocked['deploy'].assert_not_called()

    def test_missing_target_and_interrupted_attempt_are_fail_closed(self):
        mocked = self.mocks()
        mocked['prepare_target'].side_effect = ValueError('No approved target')
        self.assertEqual(run(self.root, 'docker', report=self.report)['status'], 'unavailable')
        mocked['deploy'].assert_not_called()
        path = self.directory / 'rollback' / f"attempt-{self.receipt['container_id']}.json"
        path.write_text('{"status":"started"}')
        self.assertEqual(run(self.root, 'docker', report=self.report)['status'], 'blocked')
        with deployment_lock(self.root):
            with self.assertRaises(FileExistsError):
                run(self.root, 'docker', report=self.report)
        self.assertFalse((self.directory / 'deployment.lock').exists())

    def test_predecessor_capture_preserves_numeric_model_settings_and_requires_readiness(self):
        receipt = {**self.receipt, 'model': self.target_model}
        self.receipt_path.write_text(json.dumps(receipt))
        old = deepcopy(self.state)
        old['State']['Running'] = True
        old['HostConfig'] = {'PortBindings': {'8000/tcp': [{'HostIp': '127.0.0.1', 'HostPort': '8000'}]}}
        with patch('src.serving.deploy.urlopen', return_value=StringIO(json.dumps({'status': 'ready', **self.target_model}))), \
                patch('src.serving.deploy.stage_model', return_value=(Path(self.target['staged_directory']), self.target_model)):
            target = snapshot_target(self.root, self.root, old, 'docker')
            self.assertEqual(target['image_id'], receipt['image_id'])
            self.assertEqual(target['model']['inference'], self.target_model['inference'])
            self.assertEqual(target['source_container_id'], old['Id'])
            old['State']['Running'] = False
            with self.assertRaisesRegex(ValueError, 'unready'):
                snapshot_target(self.root, self.root, old, 'docker')

    def test_validation_images_preserve_originals_and_exclude_final_test(self):
        (self.root / 'data').mkdir()
        directory = self.root / 'dataset/images'
        directory.mkdir(parents=True)
        names = [f'image{i}' for i in range(20)]
        (self.root / 'data/splits.json').write_text(json.dumps({'splits': {'validation': names, 'test': ['never']}}))
        for name in names: (directory / f'{name}.png').write_bytes(name.encode())
        images = recovery_images(self.root, 20)
        self.assertEqual([name for name, _, _ in images], names)
        for name, body, digest in images:
            self.assertEqual((directory / f'{name}.png').read_bytes(), body)
            self.assertEqual(hashlib.sha256(body).hexdigest(), digest)

    def recovery_session(self, wrong_prediction=False, stale_monitoring=False):
        receipt = {**self.receipt, 'container_id': 'c' * 64, 'model': self.target_model,
                   'started_at_utc': iso(self.now - 5)}
        session = MagicMock()
        inference_hash = hashlib.sha256(json.dumps(self.target_model['inference'], sort_keys=True).encode()).hexdigest()
        def response(value):
            obj = MagicMock()
            obj.json.return_value = value
            return obj
        def get(url, **kwargs):
            if url.endswith('/health'): return response({'status': 'ready', **self.target_model})
            self.assertIn('[15s]', kwargs['params']['query'])
            return response({'status': 'success', 'data': {'resultType': 'matrix', 'result': [{
                'metric': {'checkpoint_sha256': 'stable', 'inference_sha256': inference_hash},
                'values': [[self.now - (100 if stale_monitoring else 0), '1']]}]}})
        session.get.side_effect = get
        session.post.return_value = response({**self.target_model, 'inference_sha256': inference_hash,
                                              'detections': [] if not wrong_prediction else None})
        session.__enter__.return_value = session
        return receipt, session

    def test_recovery_requires_three_readiness_twenty_predictions_and_fresh_scrapes(self):
        receipt, session = self.recovery_session()
        images = [('image', b'png', 'sha')] * 20
        with patch('src.monitoring.rollback.requests.Session', return_value=session), \
                patch('src.monitoring.rollback.current_container'), patch('src.monitoring.rollback.time.sleep') as sleep:
            result = verify_recovery(self.root, receipt, images, self.config, 'docker')
        self.assertEqual(len(result['readiness']), 3)
        self.assertEqual(len(result['predictions']), 20)
        self.assertTrue(result['monitoring_resumed'])
        self.assertEqual(sleep.call_args_list[0].args, (10,))
        self.assertEqual(session.post.call_count, 20)

    def test_wrong_prediction_or_stale_monitoring_fails_recovery(self):
        for wrong_prediction, stale in ((True, False), (False, True)):
            receipt, session = self.recovery_session(wrong_prediction, stale)
            with patch('src.monitoring.rollback.requests.Session', return_value=session), \
                    patch('src.monitoring.rollback.current_container'), patch('src.monitoring.rollback.time.sleep'), \
                    patch('src.monitoring.rollback.time.monotonic', side_effect=[0, 1, 40]):
                with self.assertRaises((ValueError, RuntimeError)):
                    verify_recovery(self.root, receipt, [('image', b'png', 'sha')] * 20, self.config, 'docker')

    def test_shared_deploy_refuses_changed_or_expired_recovery_before_removal(self):
        staged = self.directory / 'v1'
        staged.mkdir()
        target = {**self.target, 'failed_container_id': self.receipt['container_id'],
                  'failed_runtime_started': self.state['State']['StartedAt'], 'decision_valid_until': iso(self.now - 1)}
        def command(args, **kwargs):
            if args[1:3] == ['network', 'inspect']:
                return SimpleNamespace(returncode=0, stdout=json.dumps([{'Labels': {'visual-analytics.project': 'hard-hat-detector'}}]))
            if args[1:3] == ['container', 'inspect']:
                return SimpleNamespace(returncode=0, stdout=json.dumps([self.state]))
            self.fail('No removal/start should be attempted')
        with patch('src.serving.deploy.subprocess.check_output', return_value=self.target['image_id']), \
                patch('src.serving.deploy.subprocess.run', side_effect=command):
            with self.assertRaisesRegex(ValueError, 'expired before replacement'):
                deploy(self.root, staged, self.target_model, self.target['image_id'], 'docker', recovery_target=target)
        self.assertEqual(json.loads(self.receipt_path.read_text()), self.receipt)

    def test_shared_procedure_records_target_before_replacement_and_receipts_after_probes(self):
        staged = self.directory / 'v2'
        staged.mkdir()
        calls = []
        def command(args, **kwargs):
            calls.append(args)
            if args[1:3] == ['network', 'inspect']:
                return SimpleNamespace(returncode=0, stdout=json.dumps([{'Labels': {'visual-analytics.project': 'hard-hat-detector'}}]))
            if args[1:3] == ['container', 'inspect']:
                return SimpleNamespace(returncode=0, stdout=json.dumps([self.state]))
            if args[1:3] == ['rm', '-f']:
                intent = json.loads(next(self.directory.glob('deployment-intent-*.json')).read_text())
                self.assertEqual(intent['rollback_target'], self.target)
                return SimpleNamespace(returncode=0)
            self.fail(str(args))
        def output(args, **kwargs):
            if args[1:3] == ['image', 'inspect']: return self.receipt['image_id']
            if args[1] == 'run': return 'c' * 64
            if args[1] == 'inspect': return 'true'
            if args[1] == 'rev-parse': return 'fixture'
            if args[1] == 'status': return ''
            self.fail(str(args))
        def response(request, **kwargs):
            if isinstance(request, str): return StringIO(json.dumps({'status': 'ready', **self.model}))
            return StringIO(json.dumps({**self.model, 'detections': []}))
        with patch('src.serving.deploy.snapshot_target', return_value=self.target), \
                patch('src.serving.deploy.subprocess.run', side_effect=command), \
                patch('src.serving.deploy.subprocess.check_output', side_effect=output), \
                patch('src.serving.deploy.urlopen', side_effect=response):
            with deployment_lock(self.root):
                receipt = deploy(self.root, staged, self.model, self.receipt['image_id'], 'docker')
        self.assertEqual(receipt['status'], 'ready')
        self.assertEqual(receipt['rollback_target'], self.target)
        self.assertEqual(json.loads(self.receipt_path.read_text()), receipt)
        self.assertEqual(sum(args[1] == 'rm' for args in calls), 1)


if __name__ == '__main__':
    unittest.main()
