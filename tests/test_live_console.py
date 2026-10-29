"""Live-console security, bounded controls, and actual detection matching contracts."""

from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile
import json

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from src.monitoring.serving_audit import matches, measure
from src.serving import console


class MatchingTests(unittest.TestCase):
    def test_automatic_session_keeps_fallback_after_recovery_without_another_target(self):
        from scripts.demo_rollback import automatic_monitor
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);store=root/'store';output=root/'output'
            (store/'artifacts/serving').mkdir(parents=True);output.mkdir()
            receipt={'container_id':'fallback','model':{'model_version':'1'}}
            (store/'artifacts/serving/deployment.json').write_text(json.dumps(receipt))
            (output/'watch.json').write_text('{"status":"recovered"}')
            cycles=[]
            def tick(_):
                cycles.append(1)
                (output/'watch.json').write_text('{"status":"unavailable","reason":"No earlier target"}')
                if len(cycles)==3:
                    (output/'stop').touch()
            with patch('scripts.demo_rollback.time.sleep',side_effect=tick), \
                    patch('src.monitoring.serving_audit.measure') as measure_call, \
                    patch('scripts.demo_rollback.deploy') as deploy_call:
                report=automatic_monitor(store,output,'docker',[],'batch',
                    {'manifest':{'model_version':'1'}},{'model':{'model_version':'2'}},{},[],lambda *args:None)
            self.assertEqual(len(cycles),3)
            self.assertEqual(report['status'],'stopped')
            self.assertTrue((output/'automatic-recovery.json').exists())
            self.assertEqual(json.loads((store/'artifacts/serving/deployment.json').read_text()),receipt)
            measure_call.assert_not_called()
            deploy_call.assert_not_called()

    def test_automatic_monitor_audits_without_injecting_a_fault(self):
        from scripts.demo_rollback import automatic_monitor
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);store=root/'store';output=root/'output'
            (store/'artifacts/serving').mkdir(parents=True);output.mkdir()
            receipt={'container_id':'candidate','model':{'model_version':'2'}}
            (store/'artifacts/serving/deployment.json').write_text(json.dumps(receipt))
            (output/'watch.json').write_text('{"status":"observe"}')
            baseline={'manifest':{'model_version':'1'}}
            with patch('src.monitoring.serving_audit.measure',return_value={'fixture':True}) as measure_call, \
                    patch('src.monitoring.serving_audit.comparison',return_value={'status':'available'}), \
                    patch('scripts.demo_rollback.subprocess.run') as command, \
                    patch('scripts.demo_rollback.time.sleep',side_effect=lambda _: (output/'stop').touch()):
                result=automatic_monitor(store,output,'docker',[], 'batch',baseline,receipt,{},[],lambda *a:None)
            measure_call.assert_called_once()
            command.assert_not_called()
            self.assertTrue((output/'audit.json').exists())
            self.assertEqual(result['status'],'stopped')

    def test_optional_outage_stops_service_but_does_not_call_deploy(self):
        from scripts.demo_rollback import automatic_monitor
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);store=root/'store';output=root/'output'
            (store/'artifacts/serving').mkdir(parents=True);output.mkdir()
            receipt={'container_id':'candidate','model':{'model_version':'2'}}
            (store/'artifacts/serving/deployment.json').write_text(json.dumps(receipt))
            (output/'watch.json').write_text('{"status":"observe"}')
            (output/'request-outage').touch()
            with patch('src.monitoring.rollback.current_container'),patch('scripts.demo_rollback.subprocess.run') as command, \
                    patch('scripts.demo_rollback.deploy') as deploy_call, \
                    patch('scripts.demo_rollback.time.sleep',side_effect=lambda _: (output/'stop').touch()):
                automatic_monitor(store,output,'docker',[],'batch',{'manifest':{'model_version':'1'}},receipt,{},[],lambda *a:None)
            self.assertEqual(command.call_args.args[0],['docker','stop','--time','5','candidate'])
            deploy_call.assert_not_called()
            self.assertTrue((output/'injected-outage.json').exists())

    def test_image_decoder_does_not_invoke_other_format_plugins(self):
        from fastapi import HTTPException
        from src.serving.api import decode_image
        for format_name in ('GIF', 'TIFF', 'BMP'):
            body = BytesIO()
            Image.new('RGB', (10, 10)).save(body, format=format_name)
            with self.assertRaises(HTTPException) as caught:
                decode_image(body.getvalue(), 1000)
            self.assertEqual(caught.exception.status_code, 422)

    def test_duplicates_wrong_class_and_localization(self):
        labels = np.array([[0, 0, 0, 20, 20]])
        detections = np.array([[0, 0, 20, 20, .9, 0], [0, 0, 20, 20, .8, 0], [0, 0, 20, 20, .99, 1]])
        correct = matches(labels, detections)
        self.assertTrue(correct[0].all())
        self.assertFalse(correct[1:].any())
        moved = np.array([[5, 0, 25, 20, .9, 0]])
        correct = matches(labels, moved)
        self.assertTrue(correct[0, 0])
        self.assertFalse(correct[0, -1])
        self.assertEqual(matches(labels, np.empty((0, 6))).shape, (0, 10))

    def test_switched_identity_aborts_audit(self):
        manifest = {'model_name': 'test', 'model_version': '1', 'checkpoint_sha256': 'abc',
                    'deployment_mode': 'experimental', 'inference': {'confidence': .25}}
        with tempfile.TemporaryDirectory() as temporary, patch('src.monitoring.serving_audit.requests.Session') as session:
            session.return_value.__enter__.return_value.post.return_value.json.return_value = {'model_version': '2'}
            with self.assertRaisesRegex(ValueError, 'Deployment changed'):
                measure([('test', b'png', np.array([[0, 0, 0, 20, 20]]), 20, 20)], manifest, Path(temporary) / 'audit.json')
            self.assertFalse((Path(temporary) / 'audit.json').exists())


class ConsoleTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(console.app, base_url='http://127.0.0.1:8767')
        self.headers = {'X-Console-Token': console.TOKEN}
        self.previous = console.job.copy()
        console.job.update(process=None, output=None, misses=0)

    def tearDown(self):
        console.job.update(self.previous)

    def test_controls_reject_cross_origin_and_missing_token(self):
        for headers in ({}, {**self.headers, 'Origin': 'https://example.com'}):
            self.assertEqual(self.client.post('/demo/start', json={'mode': 'quality'}, headers=headers).status_code, 403)
        self.assertEqual(self.client.get('/', headers={'Host': 'example.com'}).status_code, 400)
        self.assertEqual(self.client.post('/demo/start', json={'mode': 'arbitrary'}, headers=self.headers).status_code, 400)
        self.assertEqual(self.client.post('/demo/audit', headers=self.headers).status_code, 409)
        self.assertEqual(self.client.post('/demo/start', content=b'x' * 1025, headers=self.headers).status_code, 413)
        self.assertEqual(self.client.post('/automatic/test-outage',headers={}).status_code,403)
        self.assertEqual(self.client.post('/automatic/test-outage',headers=self.headers).status_code,409)

    def test_application_start_launches_automatic_protection_without_a_button(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(console,'ROOT',Path(temporary)), \
                patch.object(console,'launch_demo') as launch:
            console.start_automatic_protection()
            launch.assert_called_once_with('automatic','4')
            launch.reset_mock()
            with patch.object(console,'active_job',return_value=True):
                console.start_automatic_protection()
            launch.assert_not_called()

    def test_live_prediction_uses_api_without_starting_demo(self):
        prediction = {'model_name': 'test', 'model_version': '1', 'detections': []}
        image = BytesIO()
        Image.new('RGB', (16, 16)).save(image, format='PNG')
        with patch.object(console, 'call_api', return_value=prediction) as proxy, patch.object(console.subprocess, 'Popen') as launch:
            for _ in range(4):
                response = self.client.post('/deployment/predict', content=image.getvalue(), headers={**self.headers, 'Content-Type': 'image/png'})
                self.assertEqual(response.status_code, 200)
                self.assertFalse(response.json()['audit_requested'])
            self.assertEqual(proxy.call_count, 4)
            launch.assert_not_called()
        self.assertEqual(self.client.post('/predict', content=b'not png', headers=self.headers).status_code, 415)
        self.assertEqual(self.client.post('/predict', content=b'x' * (console.MAX_BYTES+1),
                         headers={**self.headers, 'Content-Type': 'image/png'}).status_code, 413)

    def test_page_explains_controlled_fault_and_no_quality_approval(self):
        page = self.client.get('/pipeline')
        self.assertEqual(page.status_code, 200)
        self.assertIn('confidence set to', page.text)
        self.assertIn('NO PRODUCTION APPROVAL', page.text)
        self.assertNotIn('__CONSOLE_TOKEN__', page.text)
        self.assertEqual(page.headers['cache-control'], 'no-store')

    def test_test_images_are_allowlisted_and_zip_has_real_image_bytes(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(console, 'ROOT', Path(temporary)):
            folder = Path(temporary) / 'data/processed/images/monitoring_current'
            folder.mkdir(parents=True)
            body = BytesIO()
            Image.new('RGB', (16, 16)).save(body, format='PNG')
            for image_id in console.SAMPLES:
                image_folder = Path(temporary) / 'data/expanded-css-v1/images/validation' if image_id.startswith('css-') else folder
                image_folder.mkdir(parents=True, exist_ok=True)
                (image_folder / (image_id + '.png')).write_bytes(body.getvalue())
            samples = self.client.get('/samples').json()
            self.assertEqual(len(samples), 12)
            self.assertEqual(self.client.get(samples[0]['url']).content, body.getvalue())
            self.assertEqual(self.client.get('/samples/not-allowlisted').status_code, 404)
            package = self.client.get('/sample-pack')
            self.assertEqual(package.status_code, 200)
            with ZipFile(BytesIO(package.content)) as archive:
                self.assertEqual(set(archive.namelist()), {'START-HERE.txt'} | {x + '.png' for x in console.SAMPLES})
                for image_id in console.SAMPLES:
                    self.assertEqual(archive.read(image_id + '.png'), body.getvalue())
                self.assertIn(b'not an independent accuracy benchmark', archive.read('START-HERE.txt'))
            (Path(temporary) / 'data/expanded-css-v1/images/validation' / (samples[0]['id'] + '.png')).unlink()
            self.assertEqual(self.client.get(samples[0]['url']).status_code, 503)

    def test_interrupted_job_is_not_presented_as_verified_progress(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(console, 'ROOT', Path(temporary)):
            root = Path(temporary)
            output = root / 'reports/live-demo-interrupted'
            output.mkdir(parents=True)
            output.with_suffix('.log').write_text('Stopped during baseline audit')
            (output / 'progress.json').write_text('{"stage":"baseline_audit","mode":"quality"}')
            (root / 'artifacts/serving').mkdir(parents=True)
            (root / 'artifacts/serving/deployment.lock').write_text('{}')
            console.job.update(process=None, output=output)
            status = self.client.get('/demo/status').json()
            self.assertEqual(status['progress']['stage'], 'needs_check')
            self.assertFalse(status['running'])
            self.assertFalse(status['automatic'])
            self.assertIsNone(status['result'])
            with patch.object(console.subprocess, 'Popen') as launch:
                self.assertEqual(self.client.post('/demo/start', json={'mode': 'quality'}, headers=self.headers).status_code, 409)
                launch.assert_not_called()
            self.assertTrue((root / 'artifacts/serving/deployment.lock').exists())

    def test_candidate_selection_reaches_existing_demo_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'reports').mkdir()
            with patch.object(console, 'ROOT', root), patch.object(console, 'JOB_FILE', root / 'reports/job.json'), \
                    patch.object(console.subprocess, 'Popen') as launch:
                for invalid in ('1', '5', '../other', None, 2, {}):
                    response = self.client.post('/demo/start', json={'mode': 'quality', 'candidate_version': invalid}, headers=self.headers)
                    self.assertEqual(response.status_code, 400)
                launch.assert_not_called()
                response = self.client.post('/demo/start', json={'mode': 'quality', 'candidate_version': '4'}, headers=self.headers)
                self.assertEqual(response.status_code, 200)
                commands = [call.args[0] for call in launch.call_args_list
                            if isinstance(call.args[0], list) and any(str(arg).endswith('demo_rollback.py') for arg in call.args[0])]
                self.assertEqual(len(commands), 1)
                command = commands[0]
                self.assertEqual(command[command.index('--candidate-version') + 1], '4')
                self.assertEqual(command[command.index('--mode') + 1], 'quality')


if __name__ == '__main__':
    unittest.main()
