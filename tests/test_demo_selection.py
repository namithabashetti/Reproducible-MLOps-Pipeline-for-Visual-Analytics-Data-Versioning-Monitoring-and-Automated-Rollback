"""Demo identity changes must not mutate deployment roles or mislabel predictions."""
from io import BytesIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from src.serving import console, demo_selection as demo


class DemoSelectionTests(unittest.TestCase):
    def setUp(self):
        self.previous=demo.selected.copy()
        self.previous_guard=demo.confidence_guard.copy()
        demo.selected.update(model=None,manifest=None)
        demo.confidence_guard.update(consecutive_low=0,last_max_confidence=None,last_trigger=None,last_error=None)
        self.client=TestClient(console.app,base_url='http://127.0.0.1:8767')
        self.headers={'X-Console-Token':console.TOKEN}

    def tearDown(self):
        demo.selected.update(self.previous)
        demo.confidence_guard.update(self.previous_guard)

    def manifest(self, version='1'):
        return {'model_name':'hard-hat-detector','model_version':version,'checkpoint_sha256':'sha'+version,
                'deployment_mode':'experimental','inference':{'confidence':.25,'max_upload_bytes':4194304,
                'max_image_pixels':4194304},'classes':{'0':'helmet','1':'head','2':'person'}}

    def test_switch_loads_requested_artifact_preserves_production_and_failed_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);folder=root/'artifacts/serving';folder.mkdir(parents=True)
            (folder/'deployment.json').write_text('{"production":"unchanged"}')
            manifests=[self.manifest('1'),self.manifest('4')]
            rows=[{'model_id':m['model_version'],'generation':'historical','available':True,
                   'model_version':m['model_version'],'checkpoint_sha256':m['checkpoint_sha256'],'run':'fixture'} for m in manifests]
            with patch.object(demo,'catalog',return_value=rows),patch.object(demo,'stage_model') as stage, \
                    patch.object(demo,'load_model') as load:
                staged_manifest={**manifests[0],'classes':{0:'helmet',1:'head',2:'person'}}
                stage.return_value=(root,staged_manifest);first=object();load.return_value=(first,manifests[0])
                result=demo.choose(root,'1','historical')
                stage.assert_called_once_with(root,root,'1',experimental=True)
                self.assertEqual(result['demo_model_id'],'1')
                self.assertEqual(result['selection_mode'],'demonstration')
                stage.return_value=(root,manifests[1]);load.side_effect=ValueError('warm-up failed')
                before=(folder/'demo-selection.json').read_bytes()
                with self.assertRaises(ValueError): demo.choose(root,'4','historical')
                self.assertIs(demo.selected['model'],first)
                self.assertEqual((folder/'demo-selection.json').read_bytes(),before)
                self.assertEqual((folder/'deployment.json').read_text(),'{"production":"unchanged"}')
                self.assertEqual(demo.health()['model_version'],'1')

    def test_prediction_reports_the_loaded_model_and_does_not_request_rollback(self):
        manifest={**self.manifest('7'),'demo_model_id':'2','generation':'retrained','selection_mode':'demonstration','run':'fixture'}
        model=object();demo.selected.update(model=model,manifest=manifest)
        with patch.object(demo,'decode_image',return_value=np.zeros((12,16,3),dtype=np.uint8)), \
                patch.object(demo,'predict_image',return_value=[]) as predict, \
                patch.object(console,'call_api') as production:
            response=self.client.post('/predict',content=b'png',headers={**self.headers,'Content-Type':'image/png'})
        self.assertEqual(response.status_code,200)
        value=response.json()
        self.assertEqual((value['demo_model_id'],value['model_version']),('2','7'))
        self.assertEqual(value['checkpoint_sha256'],'sha7')
        self.assertEqual(value['confidence_threshold'],.25)
        self.assertFalse(value['audit_requested'])
        self.assertEqual(value['detections'],[])
        self.assertIs(predict.call_args.args[0],model)
        production.assert_not_called()

    def test_three_low_confidence_predictions_automatically_switch_demo_fallback(self):
        manifest={**self.manifest('10'),'demo_model_id':'5','generation':'retrained',
                  'selection_mode':'demonstration','run':'retrain-model5-70e'}
        demo.selected.update(model=object(),manifest=manifest)
        fallback={**self.manifest('9'),'demo_model_id':'4','generation':'retrained',
                  'selection_mode':'demonstration','run':'retrain-model4-60e'}
        with patch.object(demo,'decode_image',return_value=np.zeros((12,16,3),dtype=np.uint8)), \
                patch.object(demo,'predict_image',return_value=[{'class_name':'helmet','confidence':.84}]), \
                patch.object(demo,'choose',return_value=fallback) as choose:
            first=demo.predict(b'png')
            second=demo.predict(b'png')
            third=demo.predict(b'png')
        self.assertFalse(first['automatic_fallback']['triggered'])
        self.assertEqual(second['automatic_fallback']['consecutive_low_confidence_predictions'],2)
        self.assertTrue(third['automatic_fallback']['triggered'])
        self.assertEqual(third['automatic_fallback']['from_model']['demo_model_id'],'5')
        self.assertEqual(third['automatic_fallback']['to_model']['demo_model_id'],'4')
        self.assertIn('3 consecutive images',third['automatic_fallback']['reason'])
        choose.assert_called_once_with(Path(__file__).resolve().parents[1],'4','retrained')

    def test_confidence_at_or_above_threshold_resets_consecutive_low_count(self):
        manifest={**self.manifest('10'),'demo_model_id':'5','generation':'retrained',
                  'selection_mode':'demonstration','run':'retrain-model5-70e'}
        demo.selected.update(model=object(),manifest=manifest)
        with patch.object(demo,'decode_image',return_value=np.zeros((12,16,3),dtype=np.uint8)), \
                patch.object(demo,'predict_image',side_effect=[
                    [{'class_name':'helmet','confidence':.84}],
                    [{'class_name':'helmet','confidence':.91}],
                    [{'class_name':'helmet','confidence':.84}],
                    [{'class_name':'helmet','confidence':.84}]]), \
                patch.object(demo,'choose') as choose:
            demo.predict(b'png'); reset=demo.predict(b'png'); demo.predict(b'png'); final=demo.predict(b'png')
        self.assertEqual(reset['automatic_fallback']['consecutive_low_confidence_predictions'],0)
        self.assertFalse(final['automatic_fallback']['triggered'])
        self.assertEqual(final['automatic_fallback']['consecutive_low_confidence_predictions'],2)
        choose.assert_not_called()

    def test_controls_types_size_missing_model_and_corrupt_images(self):
        self.assertEqual(self.client.post('/models/select',json={'model_id':'1','generation':'historical'}).status_code,403)
        self.assertEqual(self.client.post('/models/select',content=b'x'*1025,headers=self.headers).status_code,413)
        self.assertEqual(self.client.post('/models/select',json={'model_id':'9','generation':'historical'},headers=self.headers).status_code,400)
        with patch.object(demo,'catalog',return_value=[{'model_id':'1','generation':'retrained','available':False}]):
            self.assertEqual(self.client.post('/models/select',json={'model_id':'1','generation':'retrained'},headers=self.headers).status_code,409)
        self.assertEqual(self.client.get('/health').status_code,503)
        manifest={**self.manifest(),'demo_model_id':'1','generation':'historical','selection_mode':'demonstration','run':'fixture'}
        demo.selected.update(model=object(),manifest=manifest)
        self.assertEqual(self.client.post('/predict',content=b'x'*4194305,headers={**self.headers,'Content-Type':'image/png'}).status_code,413)
        self.assertEqual(self.client.post('/predict',content=b'bad',headers={**self.headers,'Content-Type':'image/png'}).status_code,422)
        self.assertEqual(self.client.post('/predict',content=b'gif',headers={**self.headers,'Content-Type':'image/gif'}).status_code,415)

    def test_simple_page_keeps_supporting_links_and_moves_pipeline_details(self):
        page=self.client.get('/').text
        for label in ('01 / Detection','06 / Supporting evidence','Monitoring dashboard','Training experiments','Detector health','modelChoice'):
            self.assertIn(label,page)
        self.assertNotIn('Recovery history',page)
        self.assertNotIn('id="modelEvidence"',page)
        self.assertIn('id="modelEvidence"',self.client.get('/pipeline').text)
        self.assertNotIn('__CONSOLE_TOKEN__',page)


if __name__=='__main__': unittest.main()
