"""The queue must retain exact targets, grouped partitions and single-owner execution."""
import json
from pathlib import Path
import tempfile
import unittest
from scripts.train_all_models import validate_plan, runner_lock


class QueueTests(unittest.TestCase):
    def fixture(self, root):
        source=root/'data/expanded-fixture';source.mkdir(parents=True)
        source.with_suffix('.dvc').write_text('outs:\n- md5: fixture.dir\n')
        manifest={'seed':42,'protocol':'grouped-70-15-15-v1','classes':{'0':'helmet','1':'head','2':'person'},
                  'selection':{s:[s+str(i) for i in range(n)] for s,n in [('train',5385),('validation',1154),('test',1154)]},
                  'related_groups':[['train0','train1']]}
        (source/'manifest.json').write_text(json.dumps(manifest))
        jobs=[]
        for number,epochs in zip('12345',(30,30,30,60,70)):
            directory=root/'artifacts/training'/('old'+number);directory.mkdir(parents=True)
            (directory/'lineage.json').write_text(json.dumps({'parameters':{'model':'yolov8n.pt'},'input_counts':{'train':10}}))
            (directory/'mlflow.json').write_text('{}')
            jobs.append({'model_id':number,'epochs':epochs,'historical_run':'old'+number})
        return {'dataset':'data/expanded-fixture','dataset_dvc_md5':'fixture.dir','seed':42,'test_policy':'reused test','jobs':jobs},manifest,source

    def test_exact_targets_and_counts_with_honest_test_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);plan,manifest,source=self.fixture(root)
            audit=validate_plan(root,plan,scan_labels=False)
            self.assertAlmostEqual(sum(audit['percentages'].values()),100)
            self.assertFalse(audit['pristine_test'])
            self.assertEqual(audit['counts']['train'],5385)
            plan['jobs'][4]['epochs']=30
            with self.assertRaisesRegex(ValueError,'epoch targets'): validate_plan(root,plan,False)

    def test_group_leakage_and_duplicate_ids_block_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);plan,manifest,source=self.fixture(root)
            manifest['related_groups']=[['train0','test0']]
            (source/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'Related images'): validate_plan(root,plan,False)
            manifest['selection']['test'][0]='train0'
            (source/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'overlap'): validate_plan(root,plan,False)

    def test_duplicate_supervisor_is_blocked_and_lock_is_released(self):
        with tempfile.TemporaryDirectory() as temporary:
            with runner_lock(Path(temporary)):
                with self.assertRaises(OSError):
                    with runner_lock(Path(temporary)): pass
            with runner_lock(Path(temporary)): pass


if __name__=='__main__': unittest.main()
