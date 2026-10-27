"""Check CUDA, stage isolated training inputs, and train the approved YOLO baseline."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys

import yaml


def check_device(device):
    import torch
    import torchvision

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; fix the GPU environment or explicitly choose --device cpu")
    tensor = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
    product = tensor @ tensor.T
    if not torch.isfinite(product).all() or product[0, 0].item() != 14:
        raise RuntimeError("Device tensor test failed")
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 9, 9], [20, 20, 30, 30]],
                         dtype=torch.float32, device=device)
    selected = torchvision.ops.nms(boxes, torch.tensor([0.9, 0.8, 0.7], device=device), 0.5)
    if selected.cpu().tolist() != [0, 2]:
        raise RuntimeError("torchvision NMS device test failed")
    result = {"device": device, "torch": torch.__version__, "torchvision": torchvision.__version__,
              "cuda_runtime": torch.version.cuda, "tensor_test": True, "nms_test": True}
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        result.update(gpu=torch.cuda.get_device_name(device),
                      total_vram_bytes=torch.cuda.get_device_properties(device).total_memory)
    return result


def prepare_data(project, destination, smoke, reviewed_data=None, expanded_data=None):
    """Copies keep Ultralytics label caches and possible repairs outside DVC outputs."""
    manifest = json.loads((project / "data/splits.json").read_text())
    source = project / "data/processed"
    if reviewed_data is not None and expanded_data is not None:
        raise ValueError("Choose one candidate data source")
    if expanded_data is not None:
        source = expanded_data.resolve()
        if source.parent != (project / 'data').resolve() or not re.fullmatch(r'expanded-[A-Za-z0-9_-]+',source.name):
            raise ValueError("Expanded data must be an isolated data/expanded-* export")
        expanded = json.loads((source/'manifest.json').read_text())
        selected_export = json.loads((source/'selection.json').read_text())
        if selected_export != {s:expanded['selection'][s] for s in ('train','validation')}:
            raise ValueError("Expanded candidate selection disagrees with its manifest")
        all_ids = [name for ids in expanded['selection'].values() for name in ids]
        if len(all_ids) != len(set(all_ids)) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in all_ids):
            raise ValueError("Expanded split membership is not unique")
        expected_config = {'train':'images/train','val':'images/validation','names':{0:'helmet',1:'head',2:'person'}}
        if yaml.safe_load((source/'dataset.yaml').read_text()) != expected_config:
            raise ValueError("Expanded dataset configuration changed")
        input_hashes = json.loads((source/'input-sha256.json').read_text())
        expected = {f'{kind}/{split}/{name}{suffix}' for split,ids in expanded['selection'].items() for name in ids
                    for kind,suffix in (('images','.png'),('labels','.txt'))}
        if set(input_hashes) != expected:
            raise ValueError("Expanded input hashes do not cover its membership")
        for relative,digest in input_hashes.items():
            path = (source/relative).resolve()
            if not path.is_relative_to(source) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("Expanded input changed: " + relative)
        for split,ids in expanded['selection'].items():
            for kind,suffix in (('images','.png'),('labels','.txt')):
                if {p.stem for p in (source/kind/split).glob('*'+suffix)} != set(ids):
                    raise ValueError("Unexpected expanded input files")
        manifest = {**manifest,'splits':selected_export}
    if reviewed_data is not None:
        from scripts.review_annotations import check_review, cohort_selection

        source = reviewed_data.resolve()
        if source.parent != (project / "data").resolve() or not re.fullmatch(r"person-corrected-[A-Za-z0-9_-]+", source.name):
            raise ValueError("Reviewed data must be an isolated data/person-corrected-* export")
        review = json.loads((source / "review.json").read_text())
        if not check_review(project, review)["ready_for_export"]:
            raise ValueError("Reviewed candidate is incomplete")
        if review.get("training_proposal"):
            proposal_path = source / "training-proposal.json"
            if hashlib.sha256(proposal_path.read_bytes()).hexdigest() != review["training_proposal"]["draft_sha256"]:
                raise ValueError("Training proposal provenance changed")
        selected_export = json.loads((source / "selection.json").read_text())
        if selected_export != cohort_selection(project, review):
            raise ValueError("Candidate selection disagrees with the frozen review")
        manifest = {**manifest, "splits": selected_export}
        expected_config = {"train": "images/train", "val": "images/validation",
                           "names": {0: "helmet", 1: "head", 2: "person"}}
        if yaml.safe_load((source / "dataset.yaml").read_text()) != expected_config:
            raise ValueError("Candidate dataset configuration changed")
        for split, names in selected_export.items():
            for category, suffix in (("images", ".png"), ("labels", ".txt")):
                if {p.stem for p in (source / category / split).glob('*' + suffix)} != set(names):
                    raise ValueError("Candidate files disagree with selected membership")
    destination.mkdir(parents=True, exist_ok=False)
    selected = {}
    for split, count in (("train", 64), ("validation", 32)):
        names = list(manifest["splits"][split])
        if smoke:
            random.Random(manifest["seed"]).shuffle(names)
            covered, required = set(), []
            for name in names:
                classes = {int(row.split()[0]) for row in
                           (source / "labels" / split / f"{name}.txt").read_text().splitlines()}
                if classes - covered:
                    required.append(name)
                    covered.update(classes)
                if covered == {0, 1, 2}:
                    break
            names = (required + [n for n in names if n not in required])[:count]
        selected[split] = names
        for category, suffix in (("images", ".png"), ("labels", ".txt")):
            folder = destination / category / split
            folder.mkdir(parents=True)
            for name in names:
                shutil.copy2(source / category / split / f"{name}{suffix}", folder / f"{name}{suffix}")
    config = yaml.safe_load((source / "dataset.yaml").read_text())
    (destination / "dataset.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (destination / "selection.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
    if reviewed_data is not None:
        for name in ("review.json", "annotation-cohort.json"):
            shutil.copy2(source / name, destination / name)
        if review.get("training_proposal"):
            shutil.copy2(source / "training-proposal.json", destination / "training-proposal.json")
    if expanded_data is not None:
        for name in ('manifest.json','audit.json','README.dataset.txt','README.roboflow.txt'):
            shutil.copy2(source/name,destination/name)
    if reviewed_data is not None or expanded_data is not None:
        hashes = {p.relative_to(destination).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for category in ("images", "labels") for p in sorted((destination / category).rglob('*')) if p.is_file()}
        (destination / "input-sha256.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    return destination / "dataset.yaml", {s: len(n) for s, n in selected.items()}


def resume_run(project, output):
    """Resume intact local inputs/checkpoint while retaining the interrupted evidence."""
    output = output.resolve()
    if output.parent != (project/'artifacts/training').resolve() or (output/'metrics.json').exists() or (output/'mlflow.json').exists():
        raise ValueError('Resume requires an unfinished project training run')
    lineage = json.loads((output/'lineage.json').read_text())
    training = lineage['parameters']
    if training['deterministic']:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['YOLO_CONFIG_DIR'] = str(project/'artifacts/ultralytics-settings')
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    print(json.dumps(check_device(lineage['hardware']['device'])),flush=True)
    from ultralytics import YOLO, settings
    import torch
    from src.training.evaluate import evaluate
    from src.tracking.mlflow_utils import log_completed_run, file_hash
    settings.update({'sync':False,'mlflow':False})
    hashes = json.loads((output/'inputs/input-sha256.json').read_text())
    selected=json.loads((output/'inputs/selection.json').read_text())
    expected={f'{kind}/{split}/{name}{suffix}' for split,names in selected.items() for name in names
              for kind,suffix in (('images','.png'),('labels','.txt'))}
    if set(selected)!={'train','validation'} or set(hashes)!=expected or {s:len(v) for s,v in selected.items()}!=lineage['input_counts']:
        raise ValueError('Resume input coverage changed')
    for relative,digest in hashes.items():
        path=(output/'inputs'/relative).resolve()
        if not path.is_relative_to(output/'inputs') or file_hash(path)!=digest:
            raise ValueError('Resume input changed: '+relative)
    if file_hash(output/'inputs/selection.json') != lineage.get('selection_sha256'):
        raise ValueError('Resume selection changed; this resume path requires expanded-data lineage')
    last=output/'fit/weights/last.pt'
    checkpoint=torch.load(last,map_location='cpu',weights_only=False)  # Trusted project-created checkpoint only.
    with (output/'fit/results.csv').open(newline='') as stream:
        history=list(csv.DictReader(stream))
    completed=checkpoint['epoch']+1
    if completed != len(history) or not 0<completed<lineage['epochs_requested'] or checkpoint.get('optimizer') is None:
        raise ValueError('Resume checkpoint/epoch history is incomplete or inconsistent')
    args=checkpoint['train_args']
    if (Path(args['data']).resolve()!=output/'inputs/dataset.yaml' or args['epochs']!=lineage['epochs_requested']
            or args['batch']!=training['batch_size'] or args['imgsz']!=training['image_size']):
        raise ValueError('Resume checkpoint settings differ from recorded training')
    del checkpoint
    backup=output/'resumes'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup.mkdir(parents=True,exist_ok=False)
    for name in ('fit/weights/last.pt','fit/weights/best.pt','fit/results.csv','fit/args.yaml'):
        target=backup/'before'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(output/name,target)
    for name in ('src/training/train.py','src/training/evaluate.py','src/tracking/mlflow_utils.py'):
        target=backup/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(project/name,target)
    receipt={'started_at':datetime.now(timezone.utc).isoformat(),'completed_epochs_before':completed,
             'status':'started','note':'Process ended during the next epoch; cause unverified. Resume retains optimizer state; not a claim of bit-identical uninterrupted training.',
             'sha256':{p.relative_to(backup).as_posix():file_hash(p) for p in backup.rglob('*') if p.is_file()}}
    receipt_path=backup/'resume.json';receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    try:
        model=YOLO(str(last));model.train(resume=True)
        with (output/'fit/results.csv').open(newline='') as stream:
            history=list(csv.DictReader(stream))
        if len(history)!=lineage['epochs_requested'] or not all(math.isfinite(float(v)) for row in history for v in row.values()):
            raise ValueError('Resumed training did not finish with finite epoch history')
        report=evaluate(output/'fit/weights/best.pt',output/'inputs/dataset.yaml',output,
                        lineage['hardware']['device'],training['image_size'],training['batch_size'])
        report.update(smoke_test=lineage['smoke_test'],input_counts=lineage['input_counts'],epochs_completed=len(history),finite_training_losses=True)
        (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        receipt.update(status='completed',completed_at=datetime.now(timezone.utc).isoformat())
        receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    except Exception as error:
        receipt.update(status='failed',error=repr(error));receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
        raise
    # Registration may be retried separately; never mutate its bundled resume receipt.
    print(json.dumps(log_completed_run(project,output),indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="One epoch on 64 train/32 validation images")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--run-name", help="New run directory name; existing runs are never overwritten")
    parser.add_argument("--device", choices=("cuda:0", "cpu"), help="Explicit device override; no silent fallback")
    parser.add_argument("--reviewed-data", type=Path, help="DVC-tracked, completed person-corrected-* export")
    parser.add_argument("--expanded-data", type=Path, help="DVC-tracked expanded-* candidate with frozen provenance")
    parser.add_argument("--epochs", type=int, help="Recorded epoch override for a new named experiment")
    parser.add_argument('--resume-run',type=Path,help='Resume an interrupted expanded-data run with its original settings')
    args = parser.parse_args()
    if args.resume_run:
        if any((args.smoke,args.check_only,args.run_name,args.device,args.reviewed_data,args.expanded_data,args.epochs)):
            parser.error('--resume-run retains the recorded settings and cannot be combined with training overrides')
        return resume_run(Path(__file__).resolve().parents[2],args.resume_run)
    if args.run_name and not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        parser.error("--run-name allows only letters, digits, hyphens and underscores")
    if args.reviewed_data and (not args.run_name or args.smoke):
        parser.error("Reviewed-data experiments require a new --run-name and the full frozen cohort (no --smoke)")
    if args.expanded_data and (not args.run_name or args.reviewed_data):
        parser.error("Expanded-data experiments require a new --run-name and no --reviewed-data")
    if args.epochs is not None and (args.epochs < 1 or not args.run_name or args.smoke):
        parser.error("--epochs requires a positive count, a new --run-name, and no --smoke")
    project = Path(__file__).resolve().parents[2]
    params = yaml.safe_load((project / "params.yaml").read_text())
    training = dict(params["training"])
    if args.epochs is not None:
        training['epochs'] = args.epochs
    device = args.device or training["device"]
    # Configure cuBLAS before the hardware probe creates the first CUDA context.
    if training["deterministic"]:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    hardware = check_device(device)
    print(json.dumps(hardware, indent=2), flush=True)
    (project / "reports").mkdir(exist_ok=True)
    (project / "reports/gpu-check.json").write_text(json.dumps(hardware, indent=2) + "\n")
    if args.check_only:
        return
    # These are project-local settings, not the user's global Ultralytics settings.
    settings_directory = project / "artifacts/ultralytics-settings"
    settings_directory.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(settings_directory)
    from ultralytics import YOLO, settings
    from src.training.evaluate import evaluate

    settings.update({"sync": False, "mlflow": False})
    changed = subprocess.check_output([sys.executable, "-m", "dvc", "status", "--json"],
                                      cwd=project, text=True)
    if json.loads(changed):
        raise RuntimeError("DVC data stages have changed; run dvc repro before training")
    output = project / "artifacts/training" / (args.run_name or ("smoke" if args.smoke else "baseline"))
    if output.exists():
        raise FileExistsError(f"Run already exists: {output}; preserve it before starting another run")
    candidate = None
    if args.reviewed_data:
        candidate_path = args.reviewed_data.resolve()
        pointer = candidate_path.with_suffix('.dvc')
        metadata = yaml.safe_load(pointer.read_text())["outs"][0]
        if metadata["path"] != candidate_path.name or not metadata["md5"].endswith('.dir'):
            raise ValueError("Invalid candidate DVC pointer")
        candidate = {"path": candidate_path.relative_to(project).as_posix(), "dvc_md5": metadata["md5"],
                     "review_sha256": hashlib.sha256((candidate_path / "review.json").read_bytes()).hexdigest(),
                     "cohort_sha256": hashlib.sha256((candidate_path / "annotation-cohort.json").read_bytes()).hexdigest()}
        proposal = json.loads((candidate_path / "review.json").read_text()).get("training_proposal")
        if proposal:
            candidate.update(label_method=proposal["status"], training_proposal_sha256=proposal["draft_sha256"])
    expanded_candidate = None
    if args.expanded_data:
        expanded_path = args.expanded_data.resolve()
        expanded_pointer = expanded_path.with_suffix('.dvc')
        metadata = yaml.safe_load(expanded_pointer.read_text())['outs'][0]
        if metadata['path'] != expanded_path.name or not metadata['md5'].endswith('.dir'):
            raise ValueError("Invalid expanded DVC pointer")
        expanded_candidate = {'path':expanded_path.relative_to(project).as_posix(),'dvc_md5':metadata['md5'],
                              'manifest_sha256':hashlib.sha256((expanded_path/'manifest.json').read_bytes()).hexdigest()}
    print('Verifying dataset fingerprints and copying isolated training inputs; original data is preserved.',flush=True)
    data, counts = prepare_data(project, output / "inputs", args.smoke, args.reviewed_data, args.expanded_data)
    print(json.dumps({'staged_input_counts':counts}),flush=True)
    weights = project / "artifacts/pretrained" / training["model"]
    weights.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    lock = yaml.safe_load((project / "dvc.lock").read_text())
    # The checkout may be uncommitted; retain the actual source, not only a Git SHA.
    sources = sorted(project.glob("src/**/*.py")) + [project / p for p in
              ("params.yaml", "configs/config.yaml", "requirements-lock.txt", "dvc.yaml", "dvc.lock", "dataset.dvc")]
    if candidate:
        sources += [pointer, project / "docs/annotation-policy.md", project / "data/annotation-cohort.json",
                    project / "scripts/review_annotations.py"]
        if "training_proposal_sha256" in candidate:
            sources.append(project / "scripts/annotation_triage.py")
    if expanded_candidate:
        sources += [expanded_pointer,project/'scripts/audit_additional_data.py',project/'scripts/prepare_additional_data.py']
    source_hashes = {}
    for path in sources:
        relative = path.relative_to(project)
        copy = output / "source" / relative
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copy)
        source_hashes[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    lineage = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "smoke_test": args.smoke,
        "purpose": "hardware/pipeline test" if args.smoke else f"{training['epochs']}-epoch YOLOv8n training",
        "dataset_dvc_md5": yaml.safe_load((project / "dataset.dvc").read_text())["outs"][0]["md5"],
        "processed_dvc_md5": lock["stages"]["preprocess"]["outs"][0]["md5"],
        "splits_dvc_md5": lock["stages"]["split"]["outs"][0]["md5"],
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip(),
        "git_worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=project)),
        "file_sha256": source_hashes,
        "initial_weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "ultralytics": importlib.metadata.version("ultralytics"), "hardware": hardware,
        "parameters": training, "seed": params["data"]["seed"], "input_counts": counts,
        "epochs_requested": 1 if args.smoke else training["epochs"],
    }
    if candidate:
        lineage["reviewed_data"] = candidate
    if expanded_candidate:
        lineage['expanded_data'] = expanded_candidate
        lineage['base_data'] = {key:lineage[key] for key in ('dataset_dvc_md5','processed_dvc_md5','splits_dvc_md5')}
        lineage['dataset_dvc_md5'] = lineage['processed_dvc_md5'] = expanded_candidate['dvc_md5']
        # The expanded split identity is a SHA-256, explicitly named separately.
        lineage['splits_dvc_md5'] = None
        lineage['selection_sha256'] = hashlib.sha256((output/'inputs/selection.json').read_bytes()).hexdigest()
    (output / "lineage.json").write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")
    print(f"Starting {lineage['epochs_requested']} epochs on {device}; run: {output.name}",flush=True)
    model.train(data=str(data), epochs=lineage["epochs_requested"], imgsz=training["image_size"],
                batch=training["batch_size"], workers=training["workers"], device=device,
                cache=training["cache_images"], seed=params["data"]["seed"],
                amp=training["amp"], deterministic=training["deterministic"],
                optimizer=training["optimizer"], lr0=training["learning_rate"],
                weight_decay=training["weight_decay"], patience=training["epochs"],
                project=str(output), name="fit", exist_ok=False, plots=False, verbose=False)
    best = Path(model.trainer.best)
    with (Path(model.trainer.save_dir) / "results.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    losses = [float(value) for row in history for key, value in row.items() if key.strip().startswith("train/")]
    if not losses or not all(math.isfinite(value) for value in losses):
        raise RuntimeError("Training loss history is missing or non-finite")
    report = evaluate(best, data, output, device, training["image_size"], training["batch_size"])
    report.update(smoke_test=args.smoke, input_counts=counts, epochs_completed=len(history), finite_training_losses=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    if not args.smoke:
        from src.tracking.mlflow_utils import log_completed_run

        print(json.dumps(log_completed_run(project, output), indent=2), flush=True)
        if expanded_candidate and not json.loads((output/'inputs/manifest.json').read_text()).get('protocol'):
            from src.training.evaluate import compare_expanded_run
            print('Training and registration completed; comparing all models on the same validation images.',flush=True)
            print(json.dumps(compare_expanded_run(project,output),indent=2),flush=True)


if __name__ == "__main__":
    main()
