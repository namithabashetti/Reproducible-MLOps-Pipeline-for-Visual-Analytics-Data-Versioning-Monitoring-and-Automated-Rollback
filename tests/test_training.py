"""Check held-out isolation and metric recording without claiming mocked training."""

import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import yaml
import numpy as np

from src.training.train import prepare_data, resume_run
from src.training.evaluate import evaluate, validation_cohorts, protocol_test_membership


class TrainingTests(unittest.TestCase):
    def test_resume_refuses_finished_runs_and_changed_input_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            project=Path(temporary);run=project/'artifacts/training/interrupted'
            (run/'inputs').mkdir(parents=True)
            (run/'metrics.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'unfinished'):
                resume_run(project,run)
            (run/'metrics.json').unlink()
            (run/'lineage.json').write_text(json.dumps({'parameters':{'deterministic':False},
                'hardware':{'device':'cpu'},'input_counts':{'train':1,'validation':1}}))
            (run/'inputs/selection.json').write_text(json.dumps({'train':['a'],'validation':['b']}))
            (run/'inputs/input-sha256.json').write_text('{}')
            with patch('src.training.train.check_device',return_value={}), patch('ultralytics.settings.update'), \
                    patch.dict('os.environ',clear=False):
                with self.assertRaisesRegex(ValueError,'coverage'):
                    resume_run(project,run)

    def test_final_test_rejects_old_protocol_overlap_and_related_group_leakage(self):
        selected={'train':['a','b'],'validation':['c']}
        manifest={'protocol':'grouped-70-15-15-v1','selection':{**selected,'test':['d','e']},
                  'related_groups':[['a','b'],['d','e']]}
        self.assertEqual(protocol_test_membership(manifest,selected),['d','e'])
        for mutation in ('old', 'overlap', 'related'):
            changed=json.loads(json.dumps(manifest))
            if mutation=='old': changed.pop('protocol')
            elif mutation=='overlap': changed['selection']['test'].append('a')
            else: changed['related_groups'].append(['a','d'])
            with self.assertRaises(ValueError):
                protocol_test_membership(changed,selected)

    def test_comparison_cohorts_never_include_training_or_reserved_test(self):
        selected={'train':['train'],'validation':['old','new']}
        manifest={'selection':{**selected,'test':['reserved']},'source_images':{'new':{},'reserved':{}}}
        self.assertEqual(validation_cohorts(selected,manifest),{'combined':['old','new'],'original':['old'],'additional':['new']})
        manifest['selection']['test'].append('new')
        with self.assertRaisesRegex(ValueError,'frozen validation'):
            validation_cohorts(selected,manifest)

    def test_expanded_data_preserves_test_isolation_and_rejects_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'data/expanded-fixture'
            source.mkdir(parents=True)
            selected = {s:[f'{s}{i}' for i in range(3)] for s in ('train','validation','test')}
            (root/'data/splits.json').write_text(json.dumps({'seed':42,'splits':{}}))
            hashes = {}
            for split,ids in selected.items():
                for kind,suffix in (('images','.png'),('labels','.txt')):
                    (source/kind/split).mkdir(parents=True)
                    for i,name in enumerate(ids):
                        p=source/kind/split/(name+suffix)
                        p.write_text(f'{i} 0.5 0.5 0.2 0.2\n')
                        hashes[p.relative_to(source).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
            (source/'manifest.json').write_text(json.dumps({'selection':selected}))
            (source/'selection.json').write_text(json.dumps({s:selected[s] for s in ('train','validation')}))
            (source/'input-sha256.json').write_text(json.dumps(hashes))
            (source/'dataset.yaml').write_text(yaml.safe_dump({'train':'images/train','val':'images/validation','names':{0:'helmet',1:'head',2:'person'}}))
            for name in ('audit.json','README.dataset.txt','README.roboflow.txt'):
                (source/name).write_text('{}')
            output=root/'staged'
            _,counts=prepare_data(root,output,smoke=False,expanded_data=source)
            self.assertEqual(counts,{'train':3,'validation':3})
            self.assertEqual({p.name for p in (output/'images').iterdir()},{'train','validation'})
            self.assertFalse(any('/test/' in key for key in json.loads((output/'input-sha256.json').read_text())))
            (source/'labels/train/train0.txt').write_text('changed')
            with self.assertRaisesRegex(ValueError,'Expanded input changed'):
                prepare_data(root,root/'rejected',False,expanded_data=source)
            self.assertFalse((root/'rejected').exists())

    def test_smoke_inputs_cover_classes_and_exclude_held_out_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data/processed"
            manifest = {"seed": 42, "splits": {}}
            for split in ("train", "validation", "test", "monitoring_reference", "monitoring_current"):
                names = [f"{split}{i}" for i in range(70)]
                manifest["splits"][split] = names
                for category, suffix in (("images", ".png"), ("labels", ".txt")):
                    folder = data / category / split
                    folder.mkdir(parents=True)
                    for i, name in enumerate(names):
                        (folder / f"{name}{suffix}").write_text(f"{i % 3} 0.5 0.5 0.2 0.2\n")
            (root / "data/splits.json").write_text(json.dumps(manifest))
            (data / "dataset.yaml").write_text(yaml.safe_dump({
                "train": "images/train", "val": "images/validation",
                "names": {0: "helmet", 1: "head", 2: "person"}}))
            staged = root / "staged"
            config, counts = prepare_data(root, staged, smoke=True)
            self.assertEqual(counts, {"train": 64, "validation": 32})
            self.assertEqual(set(p.name for p in (staged / "images").iterdir()), {"train", "validation"})
            self.assertNotIn("test", yaml.safe_load(config.read_text()))
            for split in counts:
                observed = {int(p.read_text().split()[0]) for p in (staged / "labels" / split).glob("*.txt")}
                self.assertEqual(observed, {0, 1, 2})
            second = root / "second"
            prepare_data(root, second, smoke=True)
            self.assertEqual((staged / "selection.json").read_bytes(), (second / "selection.json").read_bytes())
            label = next((staged / "labels/train").glob('*.txt'))
            original = data / "labels/train" / label.name
            before = original.read_bytes()
            label.write_text("changed locally")
            self.assertEqual(original.read_bytes(), before)
            with self.assertRaises(FileExistsError):
                prepare_data(root, staged, smoke=True)

    def test_evaluation_records_validation_and_rejects_nonfinite_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            weights = output / 'test.pt'
            weights.write_bytes(b'test checkpoint identity only')
            result = Mock()
            result.box.mp, result.box.mr, result.box.map50, result.box.map = 0.5, 0.4, 0.3, 0.2
            result.summary.return_value = [{"Class": "helmet", "Images": np.int64(4), "mAP50": np.float64(0.3)}]
            with patch.dict('sys.modules', {'ultralytics': Mock()}) as modules:
                model = modules['ultralytics'].YOLO.return_value
                model.val.return_value = result
                report = evaluate(weights, output/'data.yaml', output, 'cpu', 416, 4)
                self.assertEqual(model.val.call_args.kwargs['split'], 'val')
                self.assertEqual(report['evaluation_split'], 'validation')
                self.assertFalse(report['stable_model'])
                self.assertEqual(len(report['checkpoint_sha256']), 64)
                result.box.map = float('nan')
                with self.assertRaisesRegex(ValueError, 'invalid detection metrics'):
                    evaluate(weights, output/'data.yaml', output, 'cpu', 416, 4)


if __name__ == '__main__':
    unittest.main()
