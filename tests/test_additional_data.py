"""External labels must not silently become valid or leak across evaluation pools."""

from pathlib import Path
import json
import hashlib
import tempfile
import unittest

from scripts.audit_additional_data import read_labels
from scripts.prepare_additional_data import select_additions, repartition


class AdditionalDataTests(unittest.TestCase):
    def test_new_protocol_preserves_groups_and_protected_evaluation_membership(self):
        with tempfile.TemporaryDirectory() as temporary:
            project=Path(temporary); base=project/'data/expanded-css-v1';base.mkdir(parents=True)
            selection={'train':[f'image{i}' for i in range(90)],
                       'validation':[f'image{i}' for i in range(90,95)],
                       'test':[f'image{i}' for i in range(95,100)]}
            groups=[['image0','image1'],['image2','image3']]
            (project/'data/splits.json').write_text(json.dumps({'splits':{**selection,'monitoring_reference':[],
                'monitoring_current':[]},'related_groups':groups}))
            hashes={}
            for split,ids in selection.items():
                for kind,suffix,body in [('images','.png',b'image fixture'),('labels','.txt',b'0 0.5 0.5 0.2 0.2\n1 0.2 0.2 0.1 0.1\n2 0.7 0.7 0.1 0.1\n')]:
                    (base/kind/split).mkdir(parents=True)
                    for name in ids:
                        relative=f'{kind}/{split}/{name}{suffix}'
                        (base/relative).write_bytes(body);hashes[relative]=hashlib.sha256(body).hexdigest()
            (base/'input-sha256.json').write_text(json.dumps(hashes))
            (base/'manifest.json').write_text(json.dumps({'selection':selection,'source_images':{},'limitations':[]}))
            (base/'audit.json').write_text(json.dumps({'additional_related_pairs':[]}))
            for name in ['README.dataset.txt','README.roboflow.txt','dataset.yaml']:(base/name).write_text('fixture')
            outputs=[project/'data/expanded-one',project/'data/expanded-two']
            for output in outputs:
                stats=repartition(project,output)
                self.assertEqual({s:r['total_images'] for s,r in stats.items()},{'train':70,'validation':15,'test':15})
            a=json.loads((outputs[0]/'manifest.json').read_text());b=json.loads((outputs[1]/'manifest.json').read_text())
            self.assertEqual(a['selection'],b['selection'])
            self.assertFalse(set(a['selection']['train']) & set(selection['validation']+selection['test']))
            membership={n:s for s,ids in a['selection'].items() for n in ids}
            self.assertTrue(all(len({membership[n] for n in g})==1 for g in groups))
            self.assertEqual(json.loads((base/'input-sha256.json').read_text()),hashes)

    def test_transitive_groups_are_excluded_without_moving_evaluation_images(self):
        audit = {'additional_images':[{'path':f'{split}/images/{name}.jpg','split':split}
                 for split,name in [('train','a'),('train','b'),('valid','c'),('train','d'),
                                    ('train','e'),('test','f'),('train','g'),('valid','h')]],
                 'additional_related_pairs':[
                     {'a':'train/images/a.jpg','b':'train/images/b.jpg'},
                     {'a':'train/images/b.jpg','b':'valid/images/c.jpg'},
                     {'a':'train/images/d.jpg','b':'train/images/e.jpg'}],
                 'original_related_pairs':[{'additional':'train/images/d.jpg'}],
                 'label_errors':[{'file':'train/labels/g.txt'}]}
        selected, excluded = select_additions(audit)
        self.assertEqual(selected, {'train':[], 'valid':['valid/images/h.jpg'], 'test':['test/images/f.jpg']})
        self.assertEqual(sum(len(g['images']) for g in excluded), 6)
        self.assertEqual({r for g in excluded for r in g['reasons']},
                         {'related_group_crosses_supplied_splits','possible_overlap_with_original_dataset','invalid_source_label'})

    def test_rejects_bad_classes_nonfinite_outside_and_duplicate_boxes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'labels.txt'
            path.write_text('0 0.5 0.5 1 1\n2 0.2 0.3 0.1 0.2\n'
                            '10 0.5 0.5 0.1 0.1\n0 nan 0.5 0.1 0.1\n'
                            '0 0.99 0.5 0.1 0.1\n0 0.5 0.5 0 1\n'
                            '0 0.5 0.5 1 1\n0 0.5 0.5\n')
            rows, errors = read_labels(path, 10)
            self.assertEqual(len(rows), 2)
            self.assertEqual([e['line'] for e in errors], [3, 4, 5, 6, 7, 8])
            path.write_text('')
            self.assertEqual(read_labels(path, 10), ([], []))


if __name__ == '__main__':
    unittest.main()
