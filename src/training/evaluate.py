"""Evaluate a chosen checkpoint on validation data and record real detection metrics."""

import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np


def evaluate(weights, data, output, device, image_size, batch_size, evaluation_split='validation', *, plots=False):
    os.environ.setdefault('YOLO_CONFIG_DIR',str(Path(__file__).resolve().parents[2]/'artifacts/ultralytics-settings'))
    os.environ.setdefault('YOLO_OFFLINE','true')
    os.environ.setdefault('YOLO_AUTOINSTALL','false')
    from ultralytics import YOLO

    if evaluation_split not in ('validation', 'test'):
        raise ValueError('Unsupported evaluation split')
    model = YOLO(str(weights))
    result = model.val(data=str(data), split="val", device=device, imgsz=image_size,
                       batch=batch_size, workers=0, plots=plots, verbose=False,
                       conf=.001, iou=.7, max_det=300, rect=False, half=False, augment=False,
                       project=str(output), name="evaluation", exist_ok=False)
    metrics = {"precision": float(result.box.mp), "recall": float(result.box.mr),
               "map50": float(result.box.map50), "map50_95": float(result.box.map)}
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in metrics.values()):
        raise ValueError("Evaluation produced invalid detection metrics")
    report = {
        "evaluation_split": evaluation_split, "metrics": metrics,
        "per_class": [{key: value.item() if isinstance(value, np.generic) else value
                       for key, value in row.items()} for row in result.summary()],
        "checkpoint": str(Path(weights).resolve()),
        "checkpoint_sha256": hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
        "stable_model": False,
        "note": ("Validation metrics are used for selection, not an unbiased final-test estimate."
                 if evaluation_split == 'validation' else
                 "Held-out protocol test; checkpoint and evaluation settings were frozen before scoring. Source-label limitations remain."),
    }
    if plots and getattr(result, 'confusion_matrix', None) is not None:
        report['confusion_matrix'] = result.confusion_matrix.matrix.tolist()
        report['confusion_matrix_classes'] = [*model.names.values(), 'background']
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def protocol_test_membership(manifest, selected):
    """Only a freshly trained grouped protocol can use the reassigned final test."""
    if manifest.get('final_test_status'):
        raise ValueError('This test pool was already consumed; it is not a fresh final test')
    if manifest.get('protocol') != 'grouped-70-15-15-v1':
        raise ValueError('Final test requires fresh grouped-protocol training')
    splits = manifest['selection']
    if set(splits) != {'train', 'validation', 'test'} or selected != {k:splits[k] for k in ('train','validation')}:
        raise ValueError('Training selection differs from the frozen protocol')
    owner = {}
    for split, names in splits.items():
        for name in names:
            if name in owner or Path(name).name != name or '/' in name or '\\' in name:
                raise ValueError('Duplicate, overlapping or invalid protocol image ID')
            owner[name] = split
    if not splits['test']:
        raise ValueError('Empty final test')
    for group in manifest['related_groups']:
        if any(name not in owner for name in group) or len({owner[name] for name in group}) != 1:
            raise ValueError('Related images cross protocol splits')
    return splits['test']


def evaluate_protocol_test(project, directory):
    """Evaluate once in a fixed run directory; never select a model using test scores."""
    import shutil
    import subprocess
    import sys
    import yaml
    from src.tracking.mlflow_utils import file_hash, read_completed_run, local_store, verify_registration

    lineage, _, _, _, _, _ = read_completed_run(directory)
    if lineage['smoke_test'] or 'expanded_data' not in lineage:
        raise ValueError('Expected a completed full protocol experiment')
    manifest = json.loads((directory/'inputs/manifest.json').read_text())
    selected = json.loads((directory/'inputs/selection.json').read_text())
    names = protocol_test_membership(manifest, selected)
    source = (project/lineage['expanded_data']['path']).resolve()
    if not source.is_relative_to((project/'data').resolve()):
        raise ValueError('Dataset escaped project data directory')
    if file_hash(source/'manifest.json') != file_hash(directory/'inputs/manifest.json'):
        raise ValueError('Test manifest differs from training protocol')
    pointer = source.with_suffix('.dvc')
    if yaml.safe_load(pointer.read_text())['outs'][0]['md5'] != lineage['dataset_dvc_md5']:
        raise ValueError('Test DVC identity differs from training protocol')
    if json.loads(subprocess.check_output([sys.executable,'-m','dvc','status',str(pointer),'--json'],cwd=project,text=True)):
        raise ValueError('Protocol dataset changed; final test refused')
    registration = json.loads((directory/'mlflow.json').read_text())
    client, _, registry = local_store(project)
    verify_registration(client, registry['model_name'], registration['model_version'], directory)
    output = directory/'final-test'
    output.mkdir(exist_ok=False)  # Refuse a second peek or overwrite, even after interruption.
    weights = directory/'fit/weights/best.pt'
    plan = {'checkpoint_sha256':file_hash(weights),'dataset_dvc_md5':lineage['dataset_dvc_md5'],
            'model_version':registration['model_version'],'test_images':len(names),
            'settings':{'confidence':.001,'iou':.7,'max_det':300,'rect':False,'half':False,
                        'augment':False,'image_size':lineage['parameters']['image_size'],'batch':4},
            'selection_sha256':hashlib.sha256(json.dumps(names).encode()).hexdigest()}
    (output/'frozen-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    hashes = json.loads((source/'input-sha256.json').read_text())
    copied = {}
    for kind,suffix in (('images','.png'),('labels','.txt')):
        destination=output/'inputs'/kind
        destination.mkdir(parents=True)
        for name in names:
            relative=f'{kind}/test/{name}{suffix}'
            target=destination/(name+suffix)
            shutil.copy2(source/relative,target)
            copied[relative]=file_hash(target)
            if copied[relative] != hashes[relative]:
                raise ValueError('Final-test input hash mismatch')
    data=output/'inputs/dataset.yaml'
    # The evaluator uses its val loader, pointed exclusively at the frozen test copy.
    data.write_text(yaml.safe_dump({'path':str((output/'inputs').resolve()),'train':'images','val':'images',
                                   'names':{0:'helmet',1:'head',2:'person'}}))
    report=evaluate(weights,data,output,lineage['hardware']['device'],lineage['parameters']['image_size'],4,'test')
    if file_hash(weights) != plan['checkpoint_sha256'] or any(
            file_hash(output/'inputs'/Path(relative).parts[0]/Path(relative).name) != value
            for relative,value in copied.items()):
        raise ValueError('Final-test checkpoint or copied input changed')
    report.update(protocol=manifest['protocol'],images=len(names),model_version=registration['model_version'],
                  dataset_dvc_md5=lineage['dataset_dvc_md5'],production_approval=False)
    (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'input-sha256.json').write_text(json.dumps(copied,indent=2)+'\n')
    for metric,value in report['metrics'].items():
        client.log_metric(registration['run_id'],'protocol_final_test/'+metric,value)
    for name in ('metrics.json','frozen-plan.json','input-sha256.json'):
        client.log_artifact(registration['run_id'],str(output/name),'protocol-final-test')
    return report


def compare_expanded_run(project, directory, output=None, extra_run=None):
    """Compare completed candidate and prior weights on identical validation cohorts."""
    import torch
    import yaml
    from ultralytics import YOLO
    from src.tracking.mlflow_utils import file_hash, read_completed_run

    lineage, _, _, _, _, _ = read_completed_run(directory)
    if 'expanded_data' not in lineage or lineage['smoke_test']:
        raise ValueError('Expected a completed expanded-data experiment')
    inputs = directory/'inputs'
    selection = json.loads((inputs/'selection.json').read_text())
    manifest = json.loads((inputs/'manifest.json').read_text())
    if manifest.get('protocol'):
        raise ValueError('Earlier models saw some repartitioned images; use the original shared validation cohort')
    cohorts = validation_cohorts(selection, manifest)
    output = output or directory/'comparison'
    output.mkdir(exist_ok=False)
    configs = {}
    for cohort,names in cohorts.items():
        paths = output/(cohort+'.txt')
        paths.write_text(''.join(str((inputs/'images/validation'/(name+'.png')).resolve())+'\n' for name in names),encoding='utf-8')
        config = output/(cohort+'.yaml')
        config.write_text(yaml.safe_dump({'path':str(inputs.resolve()),'train':'images/train','val':str(paths.resolve()),
                                         'names':{0:'helmet',1:'head',2:'person'}}),encoding='utf-8')
        configs[cohort] = config
    models = {'model_1':project/'artifacts/training/baseline',
              'model_2':project/'artifacts/training/fulltrain-v1',
              'model_3':project/'artifacts/training/assisted-fulltrain-v1','expanded_candidate':directory}
    if extra_run is not None:
        read_completed_run(extra_run)
        trained = set(json.loads((extra_run/'inputs/selection.json').read_text())['train'])
        if trained & set(cohorts['combined']):
            raise ValueError('Additional model trained on the comparison images')
        models['new_protocol_candidate'] = extra_run
    summary = {'status':'running','cohort_counts':{k:len(v) for k,v in cohorts.items()},
               'selection_sha256':file_hash(inputs/'selection.json'),'models':{},'production_approval':False,
               'limitations':['Validation labels are not fully independently reviewed.',
                              'CPU timing is a local inference benchmark, not container HTTP latency.',
                              'More epochs and data do not guarantee better accuracy or speed.']}
    torch.set_num_threads(2)
    for name,run in models.items():
        read_completed_run(run)
        weights = run/'fit/weights/best.pt'
        if file_hash(weights) != json.loads((run/'metrics.json').read_text())['checkpoint_sha256']:
            raise ValueError('Checkpoint changed: '+name)
        registration = json.loads((run/'mlflow.json').read_text())
        record = {'checkpoint_sha256':file_hash(weights),'validation':{},
                  'model_version':registration['model_version'], 'dataset_dvc_md5':registration['dataset_dvc_md5']}
        for cohort,config in configs.items():
            destination = output/name/cohort
            destination.mkdir(parents=True)
            result = evaluate(weights,config,destination,lineage['hardware']['device'],416,4)
            record['validation'][cohort] = {'metrics':result['metrics'],'per_class':result['per_class']}
        # Fixed validation examples only; same images, settings and CPU threads for every model.
        examples = cohorts['original'][:20]+cohorts['additional'][:20]
        model = YOLO(str(weights))
        options = {'device':'cpu','imgsz':416,'conf':.25,'iou':.7,'max_det':300,'verbose':False}
        for _ in range(2):
            model.predict(str(inputs/'images/validation'/(examples[0]+'.png')),**options)
        durations = []
        started = time.perf_counter()
        for image_id in examples:
            result = model.predict(str(inputs/'images/validation'/(image_id+'.png')),**options)[0]
            durations.append(float(result.speed['inference']))
        record['cpu_timing'] = {'images':len(examples),'warmups':2,'threads':2,
                                'inference_median_ms':float(np.median(durations)),
                                'inference_p95_ms':float(np.percentile(durations,95)),
                                'mean_predict_call_ms':(time.perf_counter()-started)*1000/len(examples)}
        summary['models'][name] = record
        (output/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    summary['status'] = 'completed'
    summary['candidate_minus_model1_map50_95'] = {
        cohort:summary['models']['expanded_candidate']['validation'][cohort]['metrics']['map50_95']-
               summary['models']['model_1']['validation'][cohort]['metrics']['map50_95'] for cohort in cohorts}
    (output/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    return summary


def validation_cohorts(selection, manifest):
    """Partition frozen validation; reject training/test IDs instead of evaluating them."""
    names = selection['validation']
    protected = set(selection['train']) | set(manifest['selection'].get('test',[]))
    if set(names) & protected or len(names) != len(set(names)) or names != manifest['selection']['validation']:
        raise ValueError('Comparison must use the frozen validation membership only')
    added = set(manifest['source_images'])
    cohorts = {'combined':names,'original':[n for n in names if n not in added],
               'additional':[n for n in names if n in added]}
    if not cohorts['original'] or not cohorts['additional']:
        raise ValueError('Both validation cohorts are required')
    return cohorts


def compare_research_labels(project, source, output, candidate=None):
    """Score published-label development cohorts without touching the consumed test."""
    import shutil
    import torch
    import yaml
    from src.tracking.mlflow_utils import file_hash, read_completed_run

    source = source.resolve()
    if source.parent != (project/'data').resolve() or not source.name.startswith('expanded-shel5k-'):
        raise ValueError('Expected an isolated SHEL5K candidate')
    manifest = json.loads((source/'manifest.json').read_text())
    selection = json.loads((source/'selection.json').read_text())
    cohorts = validation_cohorts(selection, manifest)
    shared = json.loads((project/'artifacts/training/expanded-css-v1-60e/inputs/selection.json').read_text())['validation']
    if not set(shared) <= set(cohorts['combined']):
        raise ValueError('The protected shared cohort changed')
    cohorts['shared196'] = shared
    models = {'model_5': (project/'artifacts/training/protocol-v2-30e', ['combined', 'original', 'additional']),
              'model_4': (project/'artifacts/training/expanded-css-v1-60e', ['shared196'])}
    if candidate is not None:
        if file_hash(candidate.resolve()/'inputs/manifest.json') != file_hash(source/'manifest.json'):
            raise ValueError('Candidate was trained on a different label version')
        models['candidate'] = (candidate.resolve(), list(cohorts))
    # Verify completed source runs and train/cohort isolation before writing outputs.
    for run, wanted in models.values():
        read_completed_run(run)
        trained = set(json.loads((run/'inputs/selection.json').read_text())['train'])
        if any(trained.intersection(cohorts[name]) for name in wanted):
            raise ValueError('Model trained on its requested evaluation cohort')
    output.mkdir(parents=True, exist_ok=False)
    inputs = output/'inputs'
    hashes = json.loads((source/'input-sha256.json').read_text())
    copied = {}
    for kind, suffix in (('images','.png'),('labels','.txt')):
        destination = inputs/kind/'validation'
        destination.mkdir(parents=True)
        for name in cohorts['combined']:
            relative = f'{kind}/validation/{name}{suffix}'
            target = destination/(name+suffix)
            shutil.copy2(source/relative, target)
            copied[relative] = file_hash(target)
            if copied[relative] != hashes[relative]:
                raise ValueError('Changed evaluation input: '+relative)
    (output/'input-sha256.json').write_text(json.dumps(copied,indent=2)+'\n')
    configs = {}
    for cohort, names in cohorts.items():
        paths = output/(cohort+'.txt')
        paths.write_text(''.join(str((inputs/'images/validation'/(n+'.png')).resolve())+'\n' for n in names))
        config = output/(cohort+'.yaml')
        config.write_text(yaml.safe_dump({'path':str(inputs.resolve()), 'train':'images/validation',
                                          'val':str(paths.resolve()), 'names':{0:'helmet',1:'head',2:'person'}}))
        configs[cohort] = config
    summary = {'status':'running', 'dataset_manifest_sha256':file_hash(source/'manifest.json'),
               'evaluation_code_sha256':file_hash(Path(__file__)),
               'cohort_ids':cohorts, 'models':{}, 'production_approval':False,
               'note':'Corrected publisher-label development comparison. No final-test evaluation.',
               'settings':{'image_size':416,'confidence':.001,'iou':.7,'max_det':300,'rect':False,
                           'half':False,'augment':False,'batch_size':4}}
    (output/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    torch.set_num_threads(2)
    for label,(run,wanted) in models.items():
        record = {'checkpoint_sha256':file_hash(run/'fit/weights/best.pt'),'validation':{}}
        for cohort in wanted:
            destination = output/label/cohort
            destination.mkdir(parents=True)
            result = evaluate(run/'fit/weights/best.pt', configs[cohort], destination, 'cpu', 416, 4)
            record['validation'][cohort] = {'metrics':result['metrics'],'per_class':result['per_class']}
            summary['models'][label] = record
            (output/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    if any(file_hash(inputs/name) != digest for name,digest in copied.items()):
        raise ValueError('Evaluation inputs changed during scoring')
    summary['status'] = 'completed'
    if candidate is not None:
        summary['research_selection'] = research_selection(summary['models'])
    (output/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def research_selection(models):
    """Predeclared development criteria; never grant production/stable approval."""
    checks, deltas = {}, {}
    for baseline, cohort in (('model_5','combined'), ('model_4','shared196')):
        old = models[baseline]['validation'][cohort]
        new = models['candidate']['validation'][cohort]
        delta = {'overall':new['metrics']['map50_95']-old['metrics']['map50_95']}
        old_classes = {r['Class']:r['mAP50-95'] for r in old['per_class']}
        new_classes = {r['Class']:r['mAP50-95'] for r in new['per_class']}
        delta.update({c:new_classes[c]-old_classes[c] for c in ('helmet','head','person')})
        deltas[baseline] = delta
        required_person_gain = .05 if baseline == 'model_5' else 0
        checks[baseline] = (delta['person'] >= required_person_gain-1e-12 and
                            delta['overall'] >= -1e-12 and
                            delta['helmet'] >= -.02-1e-12 and delta['head'] >= -.02-1e-12)
    return {'criteria_passed':checks, 'candidate_minus_baseline_ap50_95':deltas,
            'research_target_met':all(checks.values()), 'production_approval':False,
            'automatic_deployment':False}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compare a completed expanded-data experiment without retraining')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--compare-run',type=Path)
    mode.add_argument('--test-run',type=Path)
    mode.add_argument('--research-data',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--extra-run',type=Path)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[2]
    if args.research_data:
        if not args.output:
            parser.error('--research-data requires a new --output directory')
        result=compare_research_labels(project,args.research_data,args.output.resolve(),args.extra_run)
    elif args.test_run:
        if args.output or args.extra_run:
            parser.error('--test-run uses its fixed final-test directory and cannot compare older models')
        result=evaluate_protocol_test(project,args.test_run.resolve())
    else:
        result=compare_expanded_run(project,args.compare_run.resolve(),args.output.resolve() if args.output else None,
                                    args.extra_run.resolve() if args.extra_run else None)
    print(json.dumps(result,indent=2))
