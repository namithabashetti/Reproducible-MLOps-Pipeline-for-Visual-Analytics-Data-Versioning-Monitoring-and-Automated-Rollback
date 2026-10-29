"""Persistent sequential training queue; existing runs and registry versions are immutable."""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.serving.deploy import write_receipt

QUEUE = ROOT/'artifacts/training/five-model-queue'
PLAN = ROOT/'configs/retraining.json'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


@contextmanager
def runner_lock(directory):
    """The OS releases this lock on process exit, including a crash."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'runner.lock').open('a+b') as handle:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b'0'); handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == 'nt':
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def validate_plan(project, plan, scan_labels=True):
    source = project/plan['dataset']
    pointer = yaml.safe_load(source.with_suffix('.dvc').read_text())['outs'][0]
    manifest = read(source/'manifest.json')
    if pointer['md5'] != plan['dataset_dvc_md5'] or manifest['seed'] != plan['seed']:
        raise ValueError('Frozen dataset or split seed changed')
    if manifest['classes'] != {'0':'helmet','1':'head','2':'person'}:
        raise ValueError('Incompatible class mapping')
    if manifest['protocol'] != 'grouped-70-15-15-v1':
        raise ValueError('Expected the grouped 70/15/15 protocol')
    if [(j['model_id'],j['epochs']) for j in plan['jobs']] != list(zip('12345',(30,30,30,60,70))):
        raise ValueError('The five requested epoch targets must be preserved')
    owner = {}
    for split, names in manifest['selection'].items():
        if split not in ('train','validation','test'):
            raise ValueError('Unknown partition')
        for name in names:
            if name in owner or Path(name).name != name or '/' in name or '\\' in name:
                raise ValueError('Split overlap or invalid image identifier')
            owner[name] = split
    for group in manifest['related_groups']:
        if any(n not in owner for n in group) or len({owner[n] for n in group}) != 1:
            raise ValueError('Related images cross partitions')
    counts = {s:len(v) for s,v in manifest['selection'].items()}
    if counts != {'train':5385,'validation':1154,'test':1154}:
        raise ValueError('Frozen split counts changed')
    boxes = {s:[0,0,0] for s in counts}
    if scan_labels:
        for name,split in owner.items():
            if not (source/'images'/split/(name+'.png')).is_file():
                raise ValueError('Missing image: '+name)
            for line in (source/'labels'/split/(name+'.txt')).read_text().splitlines():
                values = [float(v) for v in line.split()]
                if len(values)!=5 or values[0] not in (0,1,2) or not all(math.isfinite(v) for v in values):
                    raise ValueError('Invalid YOLO row: '+name)
                _,x,y,w,h = values
                if not (0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1
                        and x-w/2>=-1e-7 and y-h/2>=-1e-7 and x+w/2<=1+1e-7 and y+h/2<=1+1e-7):
                    raise ValueError('Out-of-bounds YOLO box: '+name)
                boxes[split][int(values[0])] += 1
        if any(min(v)==0 for v in boxes.values()):
            raise ValueError('A class is absent from a partition')
    historical=[]
    for job in plan['jobs']:
        directory=project/'artifacts/training'/job['historical_run']
        lineage=read(directory/'lineage.json'); registry=read(directory/'mlflow.json')
        if lineage['parameters']['model'] != 'yolov8n.pt':
            raise ValueError('Inspect the changed architecture before retraining')
        historical.append({'model_id':job['model_id'],'run':job['historical_run'],
            'parameters':lineage['parameters'],'input_counts':lineage['input_counts'],'registration':registry})
    return {'checked_at':now(),'dataset_dvc_md5':pointer['md5'],'manifest_sha256':sha(source/'manifest.json'),
            'seed':manifest['seed'],'counts':counts,'percentages':{k:100*v/sum(counts.values()) for k,v in counts.items()},
            'class_boxes':boxes,'classes':manifest['classes'],'related_groups_checked':len(manifest['related_groups']),
            'label_check':'Structural and publisher-hash checks; not proof that every visible object is labeled',
            'pristine_test':False,'test_policy':plan['test_policy'],'historical_models':historical}


def progress(job):
    directory=ROOT/'artifacts/training'/job['run']
    path=directory/'fit/results.csv'
    try:
        with path.open(newline='') as stream:
            rows=list(csv.DictReader(stream))
        job['epochs_completed']=int(rows[-1]['epoch']) if rows else 0
        job['last_epoch_metrics']=rows[-1] if rows else None
        job['last_progress_at']=datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat()
    except (OSError,ValueError,KeyError):
        job.setdefault('epochs_completed',0)
    job['best_checkpoint']=str(directory/'fit/weights/best.pt')
    job['last_checkpoint']=str(directory/'fit/weights/last.pt')
    registration=read(directory/'mlflow.json')
    if registration:
        job['registration']=registration


def matching_process(identity):
    if not identity:
        return None
    try:
        process=psutil.Process(identity['pid'])
        if abs(process.create_time()-identity['created'])>.01 or process.cmdline()!=identity['command']:
            return None
        return process if process.is_running() else None
    except psutil.Error:
        return None


def identity(process):
    return {'pid':process.pid,'created':process.create_time(),'command':process.cmdline()}


def completed(job, plan):
    from src.tracking.mlflow_utils import read_completed_run, local_store, verify_registration
    directory=ROOT/'artifacts/training'/job['run']
    lineage,*_=read_completed_run(directory)
    registration=read(directory/'mlflow.json')
    if (lineage['epochs_requested']!=job['epochs'] or lineage['dataset_dvc_md5']!=plan['dataset_dvc_md5']
            or lineage['seed']!=plan['seed'] or not registration):
        raise ValueError('Completed run does not match its queue assignment')
    client,_,registry=local_store(ROOT)
    verify_registration(client,registry['model_name'],registration['model_version'],directory)
    progress(job)
    job.update(status='training_completed',registration=registration,checkpoint_sha256=sha(directory/'fit/weights/best.pt'))


def evaluate_all(state, plan):
    """Freeze all five artifacts before any reused-test scores are generated."""
    from src.training.evaluate import evaluate
    from src.tracking.mlflow_utils import local_store
    source=ROOT/plan['dataset']; manifest=read(source/'manifest.json')
    hashes=read(source/'input-sha256.json')
    frozen=[{'model_id':j['model_id'],'run':j['run'],'checkpoint_sha256':j['checkpoint_sha256']} for j in state['jobs']]
    frozen_path=QUEUE/'finalized-checkpoints.json'
    if frozen_path.exists() and read(frozen_path)!=frozen:
        raise ValueError('Finalized checkpoint selection changed after test evaluation began')
    if not frozen_path.exists():
        write_receipt(frozen_path,frozen)
    client,_,_=local_store(ROOT)
    for job in state['jobs']:
        directory=ROOT/'artifacts/training'/job['run']
        for split in ('validation','test'):
            output=directory/'final-evaluation'/split
            if (output/'complete.json').exists():
                record=read(output/'complete.json')
                if record['checkpoint_sha256']!=job['checkpoint_sha256']:
                    raise ValueError('Completed evaluation identity changed')
                continue
            output.mkdir(parents=True,exist_ok=False)
            write_receipt(output/'plan.json',{'checkpoint_sha256':job['checkpoint_sha256'],
                'dataset_dvc_md5':plan['dataset_dvc_md5'],'split':split,'pristine_test':False,'test_policy':plan['test_policy']})
            for kind,suffix in (('images','.png'),('labels','.txt')):
                target=output/'inputs'/kind;target.mkdir(parents=True)
                for name in manifest['selection'][split]:
                    relative=f'{kind}/{split}/{name}{suffix}'
                    shutil.copy2(source/relative,target/(name+suffix))
                    if sha(target/(name+suffix))!=hashes[relative]:
                        raise ValueError('Evaluation input fingerprint changed')
            data=output/'inputs/dataset.yaml'
            data.write_text(yaml.safe_dump({'path':str((output/'inputs').resolve()),'train':'images','val':'images',
                                          'names':{0:'helmet',1:'head',2:'person'}}))
            lineage=read(directory/'lineage.json')
            result=evaluate(directory/'fit/weights/best.pt',data,output,'cuda:0',416,4,split,plots=True)
            if sha(directory/'fit/weights/best.pt')!=job['checkpoint_sha256']:
                raise ValueError('Checkpoint changed during scoring')
            result.update(pristine_test=False,test_policy=plan['test_policy'],model_id=job['model_id'],
                          registry_version=job['registration']['model_version'],dataset_dvc_md5=plan['dataset_dvc_md5'])
            write_receipt(output/'metrics.json',result)
            for metric,value in result['metrics'].items():
                client.log_metric(job['registration']['run_id'],f'retraining_{split}/{metric}',value)
            client.log_artifacts(job['registration']['run_id'],str(output/'evaluation'),f'retraining-{split}/plots')
            client.log_artifact(job['registration']['run_id'],str(output/'metrics.json'),f'retraining-{split}')
            write_receipt(output/'complete.json',result)
        job['status']='evaluated_reused_test'
        write_receipt(QUEUE/'status.json',state)


def run(adopt_pid=None):
    with runner_lock(QUEUE):
        plan=read(PLAN)
        state=read(QUEUE/'status.json')
        if state and state['plan_sha256']!=sha(PLAN):
            raise ValueError('Queue plan changed; preserve the existing job history')
        if state is None:
            audit=validate_plan(ROOT,plan)
            write_receipt(QUEUE/'dataset-audit.json',audit)
            state={'created_at':now(),'plan_sha256':sha(PLAN),'jobs':[{**j,'status':'queued','epochs_completed':0} for j in plan['jobs']]}
        state.update(runner=identity(psutil.Process()),status='running',updated_at=now())
        if adopt_pid:
            job=state['jobs'][0]
            process=psutil.Process(adopt_pid)
            if not any('continue-training.ps1' in c for c in process.cmdline()):
                raise ValueError('Only the inspected existing training supervisor may be adopted')
            lineage=read(ROOT/'artifacts/training'/job['run']/'lineage.json')
            if not lineage or lineage['dataset_dvc_md5']!=plan['dataset_dvc_md5'] or lineage['epochs_requested']!=30:
                raise ValueError('Existing training run differs from the queue plan')
            job.update(process=identity(process),status='running',adopted=True,
                stdout=str(ROOT/'reports/quality-upgrade-20261027/training-resumed.log'),
                stderr=str(ROOT/'reports/quality-upgrade-20261027/training-resumed.err'))
        write_receipt(QUEUE/'status.json',state)
        power=None
        try:
            if os.name=='nt':
                import ctypes
                power=ctypes.windll.kernel32.SetThreadExecutionState
                power(0x80000001)
            for job in state['jobs']:
                if job['status'] in ('training_completed','evaluated_reused_test'):
                    completed(job,plan)
                    continue
                if (QUEUE/'pause-request').exists():
                    state['status']='paused_between_models'
                    return
                process=matching_process(job.get('process'))
                directory=ROOT/'artifacts/training'/job['run']
                child=None
                if process is None:
                    if (directory/'mlflow.json').exists():
                        completed(job,plan)
                        continue
                    others=[p.pid for p in psutil.process_iter(['cmdline']) if p.info['cmdline'] and 'src.training.train' in p.info['cmdline']]
                    if others:
                        raise RuntimeError(f'Another training process is active: {others}; refusing duplicate GPU jobs')
                    if directory.exists():
                        if not (directory/'fit/weights/last.pt').exists():
                            raise RuntimeError('Interrupted run has no resumable checkpoint; preserve it for inspection: '+str(directory))
                        command=[sys.executable,'-u','-m','src.training.train','--resume-run',str(directory)]
                    else:
                        command=[sys.executable,'-u','-m','src.training.train','--expanded-data',plan['dataset'],
                                 '--run-name',job['run'],'--epochs',str(job['epochs'])]
                    attempt=datetime.now().strftime('%Y%m%d-%H%M%S')
                    out=QUEUE/f'model-{job["model_id"]}-{attempt}.stdout.log';err=out.with_name(out.name.replace('.stdout.','.stderr.'))
                    env={**os.environ,'YOLO_OFFLINE':'true','YOLO_AUTOINSTALL':'false','PYTHONUNBUFFERED':'1',
                         'MLFLOW_DISABLE_TELEMETRY':'true','DVC_NO_ANALYTICS':'true'}
                    with out.open('wb') as stdout,err.open('wb') as stderr:
                        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                    process=psutil.Process(child.pid)
                    job.update(process=identity(process),stdout=str(out),stderr=str(err),status='running',started_at=now())
                while process is not None:
                    progress(job);state['updated_at']=now();write_receipt(QUEUE/'status.json',state)
                    try:
                        exit_code=process.wait(timeout=20)
                        if child is not None:
                            exit_code=child.wait()
                        job['exit_code']=exit_code
                        break
                    except psutil.TimeoutExpired:
                        continue
                if job.get('exit_code') not in (0,None):
                    raise RuntimeError(f'Model {job["model_id"]} exited with {job["exit_code"]}; inspect stderr')
                completed(job,plan);job['completed_at']=now();write_receipt(QUEUE/'status.json',state)
            state['status']='evaluating_finalized_models';write_receipt(QUEUE/'status.json',state)
            evaluate_all(state,plan)
            state['status']='completed_research_evaluation'
        except Exception as error:
            state.update(status='failed',error=repr(error))
            if 'job' in locals() and job['status'] not in ('training_completed','evaluated_reused_test'):
                job.update(status='failed',error=repr(error))
            raise
        finally:
            state['updated_at']=now();write_receipt(QUEUE/'status.json',state)
            if power:
                power(0x80000000)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('run','status','pause','resume'))
    parser.add_argument('--adopt-pid',type=int)
    args=parser.parse_args()
    if args.action=='status':
        state=read(QUEUE/'status.json')
        if state:
            for job in state['jobs']: progress(job)
        print(json.dumps(state,indent=2))
    elif args.action=='pause':
        QUEUE.mkdir(parents=True,exist_ok=True);(QUEUE/'pause-request').touch()
        print('The active model will finish; subsequent models will wait.')
    else:
        if args.action=='resume':
            (QUEUE/'pause-request').unlink(missing_ok=True)
        run(args.adopt_pid)
