"""Build an isolated three-class candidate from audited external data and frozen model-2 inputs."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import yaml

from scripts.audit_additional_data import read_labels
from src.data.preprocess import letterbox, yolo_labels

NAMES = {0: "helmet", 1: "head", 2: "person"}
SOURCE_NAMES = ["Hardhat", "Mask", "NO-Hardhat", "NO-Mask", "NO-Safety Vest", "Person",
                "Safety Cone", "Safety Vest", "machinery", "vehicle"]
MAPPING = {0: "helmet", 2: "head", 5: "person"}


def select_additions(audit):
    """Exclude whole related components; never move supplied evaluation images into training."""
    records = {r["path"]: r for r in audit["additional_images"]}
    parent = {name: name for name in records}

    def root(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for pair in audit["additional_related_pairs"]:
        parent[root(pair["a"])] = root(pair["b"])
    groups = defaultdict(list)
    for name in records:
        groups[root(name)].append(name)
    overlaps = {p["additional"] for p in audit["original_related_pairs"]}
    invalid = {e["file"].replace('/labels/', '/images/').rsplit('.', 1)[0] for e in audit["label_errors"]}
    selected, excluded = {s: [] for s in ("train", "valid", "test")}, []
    for group in groups.values():
        reasons = []
        if len({records[n]["split"] for n in group}) > 1:
            reasons.append("related_group_crosses_supplied_splits")
        if overlaps.intersection(group):
            reasons.append("possible_overlap_with_original_dataset")
        if any(n.rsplit('.', 1)[0] in invalid for n in group):
            reasons.append("invalid_source_label")
        if reasons:
            excluded.append({"images": sorted(group), "reasons": reasons})
        else:
            for name in group:
                selected[records[name]["split"]].append(name)
    return {s: sorted(v) for s, v in selected.items()}, excluded


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(project, report_path, output):
    audit = json.loads(report_path.read_text())
    source = Path(audit["source"])
    if audit["class_names"] != SOURCE_NAMES:
        raise ValueError("The source class order differs from the inspected dataset")
    if output.parent != (project / 'data').resolve() or not output.name.startswith('expanded-'):
        raise ValueError("Use a new data/expanded-* directory")
    if output.exists():
        raise FileExistsError("Preserve existing candidates; use a new version name")
    selected, excluded = select_additions(audit)
    records = {r["path"]: r for r in audit["additional_images"]}
    base = project / 'artifacts/training/fulltrain-v1/inputs'
    selection = json.loads((base / 'selection.json').read_text())
    base_hashes = json.loads((base / 'input-sha256.json').read_text())
    original_splits = json.loads((project / 'data/splits.json').read_text())["splits"]
    if set(selection['train']) != set(original_splits['train']) or not set(selection['validation']) <= set(original_splits['validation']):
        raise ValueError("The frozen base no longer matches original train/validation membership")
    # Verify inputs before exporting. The audit cannot authorize changed source bytes.
    for split in selected:
        for name in selected[split]:
            path = source / name
            label = path.parent.parent / 'labels' / (path.stem + '.txt')
            if sha(path) != records[name]['sha256'] or sha(label) != records[name]['label_sha256']:
                raise ValueError("Additional source changed after audit: " + name)
    expected = {f'{kind}/{split}/{name}{suffix}' for split, ids in selection.items() for name in ids
                for kind, suffix in (('images','.png'),('labels','.txt'))}
    if set(base_hashes) != expected or any(sha(base / name) != digest for name,digest in base_hashes.items()):
        raise ValueError("Frozen model-2 inputs failed their recorded hash checks")
    output.mkdir()
    for split, ids in selection.items():
        for kind,suffix in (('images','.png'),('labels','.txt')):
            folder = output / kind / split
            folder.mkdir(parents=True)
            for name in ids:
                shutil.copy2(base / kind / split / (name+suffix), folder / (name+suffix))
    counts, provenance = {}, {}
    for supplied_split, paths in selected.items():
        split = {'train':'train','valid':'validation','test':'test'}[supplied_split]
        selection.setdefault(split, [])
        for kind in ('images','labels'):
            (output / kind / split).mkdir(parents=True,exist_ok=True)
        boxes = Counter()
        for name in paths:
            record = records[name]
            image_path = source / name
            label_path = image_path.parent.parent / 'labels' / (image_path.stem+'.txt')
            image = cv2.imread(str(image_path))
            rows, invalid = read_labels(label_path, len(SOURCE_NAMES))
            if image is None or invalid:
                raise ValueError("Source failed final preparation check: " + name)
            height,width = image.shape[:2]
            objects = [(MAPPING[c],((x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height))
                       for c,x,y,w,h in rows if c in MAPPING]
            image, transform = letterbox(image,416,114)
            new_id = 'css-' + hashlib.sha256(name.encode()).hexdigest()[:24]
            if new_id in provenance:
                raise ValueError("Derived name collision")
            if not cv2.imwrite(str(output/'images'/split/(new_id+'.png')),image,[cv2.IMWRITE_PNG_COMPRESSION,3]):
                raise OSError("Could not write derived image")
            (output/'labels'/split/(new_id+'.txt')).write_text(yolo_labels(objects,{v:k for k,v in NAMES.items()},transform,416),encoding='utf-8')
            selection[split].append(new_id)
            provenance[new_id] = {"source":name,"image_sha256":record['sha256'],"label_sha256":record['label_sha256']}
            boxes.update(o[0] for o in objects)
        counts[split] = {'added_images':len(paths),'added_boxes':dict(boxes),'total_images':len(selection[split])}
    hashes = {p.relative_to(output).as_posix():sha(p) for kind in ('images','labels') for p in sorted((output/kind).rglob('*')) if p.is_file()}
    manifest = {'schema':1,'label_status':'source annotations as supplied; not fully reviewed',
                'base_run':'fulltrain-v1','base_input_hashes_sha256':sha(base/'input-sha256.json'),
                'audit_sha256':sha(report_path),'classes':NAMES,'class_mapping':MAPPING,
                'dropped_source_classes':[n for i,n in enumerate(SOURCE_NAMES) if i not in MAPPING],
                'mapping_note':'NO-Hardhat means an uncovered head in inspected examples. Source box conventions may differ; geometry is preserved.',
                'preprocessing':{'image_size':416,'pad_value':114,'upscale':False},
                'statistics':counts,'selection':selection,'source_images':provenance,'excluded_groups':excluded,
                'limitations':audit['limitations']+['Four mapping examples were inspected; this is not a full visual label review.',
                    'Original 100-image validation limitations remain. New validation is source-labeled, not independently adjudicated.',
                    'The external test split is reserved and is absent from the training YAML.']}
    for name,value in [('manifest.json',manifest),('selection.json',{s:selection[s] for s in ('train','validation')}),('input-sha256.json',hashes)]:
        (output/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    (output/'dataset.yaml').write_text(yaml.safe_dump({'train':'images/train','val':'images/validation','names':NAMES},sort_keys=False),encoding='utf-8')
    shutil.copy2(report_path,output/'audit.json')
    for name in ('README.dataset.txt','README.roboflow.txt'):
        shutil.copy2(source/name,output/name)
    return counts


def repartition(project, output):
    """Version a 70/15/15 protocol; old checkpoints must not use its final test."""
    from src.data.split import assign_splits

    base = project / 'data/expanded-css-v1'
    prior = json.loads((base / 'manifest.json').read_text())
    audit = json.loads((base / 'audit.json').read_text())
    original = json.loads((project / 'data/splits.json').read_text())
    if output.parent != (project / 'data').resolve() or not output.name.startswith('expanded-') or output.exists():
        raise ValueError('Use a new isolated data/expanded-* export')
    # Verify the exact existing derived inputs before reusing them.
    for relative, expected in json.loads((base / 'input-sha256.json').read_text()).items():
        if sha(base / relative) != expected:
            raise ValueError('Existing expanded input changed: ' + relative)
    locations = {n: (project/'data/processed', s) for s, ids in original['splits'].items() for n in ids}
    locations.update({n: (base, s) for s, ids in prior['selection'].items() for n in ids})
    parent = {n:n for n in locations}
    def root(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n
    def join(a,b):
        if a in parent and b in parent:
            parent[root(a)] = root(b)
    for group in original['related_groups']:
        for n in group[1:]: join(group[0], n)
    external_ids = {r['source']: n for n,r in prior['source_images'].items()}
    for pair in audit['additional_related_pairs']:
        if pair['a'] in external_ids and pair['b'] in external_ids:
            join(external_ids[pair['a']], external_ids[pair['b']])
    groups = defaultdict(list)
    for n in locations: groups[root(n)].append(n)
    fixed_test = set(original['splits']['test']) | set(prior['selection']['test'])
    fixed_validation = set(original['splits']['validation']) | set(prior['selection']['validation'])
    fixed_validation |= set(original['splits']['monitoring_reference']) | set(original['splits']['monitoring_current'])
    result = {'train':[], 'validation':[], 'test':[]}
    free = []
    for group in groups.values():
        tests, vals = bool(set(group)&fixed_test), bool(set(group)&fixed_validation)
        if tests and vals:
            raise ValueError('A related group crosses protected validation/test memberships')
        if tests or vals: result['test' if tests else 'validation'].extend(group)
        else: free.append(group)
    total = len(locations)
    targets = {'train':round(total*.70), 'validation':round(total*.15)}
    targets['test'] = total - sum(targets.values())
    labels = {}
    for n,(folder,split) in locations.items():
        rows, errors = read_labels(folder/'labels'/split/(n+'.txt'),3)
        if errors: raise ValueError('Invalid derived labels: '+n)
        labels[n] = Counter(NAMES[row[0]] for row in rows)
    remaining = {s:targets[s]-len(result[s]) for s in targets}
    assigned = assign_splits(free, {n:labels[n] for group in free for n in group}, remaining, NAMES.values(),42)
    for s in result: result[s] = sorted(result[s]+assigned[s])
    membership = {n:s for s,ids in result.items() for n in ids}
    assert len(membership)==total and all(len({membership[n] for n in g})==1 for g in groups.values())
    assert not set(result['train']) & (fixed_validation | fixed_test)
    output.mkdir()
    hashes, statistics = {}, {}
    for split, ids in result.items():
        counts = sum((labels[n] for n in ids),Counter())
        statistics[split] = {'total_images':len(ids),'percentage':100*len(ids)/total,'class_boxes':dict(counts)}
        for kind,suffix in [('images','.png'),('labels','.txt')]:
            (output/kind/split).mkdir(parents=True)
            for n in ids:
                folder, old_split = locations[n]
                dest = output/kind/split/(n+suffix)
                shutil.copy2(folder/kind/old_split/(n+suffix),dest)
                hashes[dest.relative_to(output).as_posix()] = sha(dest)
    manifest = {**prior,'protocol':'grouped-70-15-15-v1','seed':42,'selection':result,'statistics':statistics,
        'related_groups':[sorted(g) for g in groups.values() if len(g)>1],
        'previous_manifest_sha256':sha(base/'manifest.json'),
        'monitoring_membership':'Existing 500 monitoring images are validation replay subsets, never final-test evidence.',
        'evaluation_restriction':'Final test is valid only for fresh training on this protocol. Earlier models trained on some reassigned images.',
        'reassigned_prior_training':{s:len(set(result[s]) & set(prior['selection']['train'])) for s in ('validation','test')},
        'limitations':prior['limitations']+['Approximate 70/15/15 after integer rounding; image count is not independent scene count.']}
    for name,value in [('manifest.json',manifest),('selection.json',{s:result[s] for s in ('train','validation')}),('input-sha256.json',hashes)]:
        (output/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    for name in ['audit.json','dataset.yaml','README.dataset.txt','README.roboflow.txt']:
        shutil.copy2(base/name,output/name)
    return statistics


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit',type=Path)
    parser.add_argument('--repartition',action='store_true',help='New grouped 70/15/15 protocol; preserves existing exports')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    if not args.repartition and args.audit is None: parser.error('--audit is required for initial preparation')
    print(json.dumps(repartition(project,args.output.resolve()) if args.repartition else
                     prepare(project,args.audit.resolve(),args.output.resolve()),indent=2))
