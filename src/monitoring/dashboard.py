"""Expose recorded evaluations and recovery evidence; never invent live accuracy."""

from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path

from prometheus_client import CollectorRegistry, Gauge, generate_latest


def read_json(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def history(project):
    rows = []
    for path in sorted((project/'reports').glob('live-demo-*/result.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:12]:
        result = read_json(path)
        if not result: continue
        sources = result.get('source_versions',[])
        rollback = result.get('rollback',{})
        rows.append({'report':path.parent.name, 'status':result.get('status','unknown'),
            'mode':result.get('mode','unknown'), 'controlled_demo':True,
            'previous_model':sources[1] if len(sources)==2 else None,
            'recovery_model':sources[0] if sources else None,
            'reason':', '.join(rollback.get('decision',{}).get('reasons',[])) or result.get('error','Unavailable'),
            'completed_at':result.get('completed_at'),
            'normal_model':result.get('original_api_restored',{}).get('model',{}).get('model_version')})
    return rows


def recorded_summary(project):
    """Show a completed new comparison only when its cohort and registry receipt match."""
    summary=read_json(project/'reports/faculty-summary.json')
    comparison_path=project/'reports/final-audit-20261026/comparison-with-protocol/comparison.json'
    comparison=read_json(comparison_path)
    original=read_json(project/'reports/final-audit-20261026/comparison/comparison.json')
    directory=project/'artifacts/training/protocol-v2-30e'
    registered=read_json(directory/'mlflow.json')
    lineage=read_json(directory/'lineage.json')
    if not all((summary,comparison,original,registered,lineage)):
        return summary
    candidate=comparison.get('models',{}).get('new_protocol_candidate',{})
    if (comparison.get('status')!='completed' or not registered.get('verified')
            or not comparison.get('selection_sha256')
            or comparison.get('selection_sha256')!=original.get('selection_sha256')
            or candidate.get('checkpoint_sha256')!=registered.get('checkpoint_sha256')
            or str(candidate.get('model_version'))!=str(registered.get('model_version'))
            or candidate.get('dataset_dvc_md5')!=registered.get('dataset_dvc_md5')):
        return summary
    records={str(row['model_version']):row for row in comparison['models'].values()}
    if str(registered['model_version']) in {str(model['version']) for model in summary['models']}:
        return summary
    models=[]
    for model in summary['models']:
        record=records.get(str(model['version']))
        if not record or record['checkpoint_sha256']!=model['checkpoint_sha256']:
            return summary
        updated=dict(model)
        if updated['name']=='Highest shared score': updated['name']='Expanded images'
        models.append(updated)
    models.append({'version':str(registered['model_version']),'name':'Fresh grouped split',
                   'dataset_version':registered['dataset_dvc_md5'],'dataset_name':'Grouped 70/15/15 protocol',
                   'epochs':lineage['epochs_requested'],'training_status':'Completed',
                   'acceptance':'Experimental; quality not approved','checkpoint_sha256':registered['checkpoint_sha256']})
    for model in models:
        record=records[str(model['version'])]
        metrics=dict(record['validation']['combined']['metrics'])
        precision,recall=metrics['precision'],metrics['recall']
        metrics['f1_macro_pr']=2*precision*recall/(precision+recall) if precision+recall else 0.
        model.update(metrics=metrics,per_class=record['validation']['combined']['per_class'],
                     latency_ms=record['cpu_timing']['inference_median_ms'])
    summary.update(models=models,comparison_sha256=hashlib.sha256(comparison_path.read_bytes()).hexdigest(),
        comparison_note='All shown models used the same 196 validation images and explicit evaluation settings. '
                        'Higher versions do not guarantee improvement. The new model remains experimental; deployment is unchanged. '
                        'F1* combines macro precision and recall. CPU timing is a local inference benchmark, not HTTP latency.')
    return summary


def project_status(project):
    summary = recorded_summary(project)
    directory = project/'artifacts/training/protocol-v2-30e'
    lineage = read_json(directory/'lineage.json')
    training = None
    if lineage:
        try:
            path = directory/'fit/results.csv'
            with path.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            training = {'run':directory.name, 'epochs_requested':lineage['epochs_requested'],
                        'epochs_recorded':int(rows[-1]['epoch']) if rows else 0,
                        'last_recorded_at':datetime.fromtimestamp(path.stat().st_mtime,timezone.utc).isoformat(),
                        'registered':bool(read_json(directory/'mlflow.json'))}
        except (OSError, ValueError, KeyError, IndexError):
            training = {'run':directory.name, 'epochs_recorded':None, 'epochs_requested':lineage['epochs_requested'],
                        'last_recorded_at':None, 'registered':False}
        comparison=read_json(project/'reports/final-audit-20261026/comparison-with-protocol/comparison.json')
        final_test=read_json(directory/'final-test/metrics.json')
        training['comparison_complete']=bool(comparison and comparison.get('status')=='completed')
        training['final_test_complete']=bool(final_test and final_test.get('evaluation_split')=='test'
                                           and final_test.get('production_approval') is False)
    return {'evidence':summary, 'history':history(project),
            'training':training,
            'production_approval':False,
            'note':'Recorded validation results, not live accuracy. No quality-approved stable model is assigned.'}


def evidence_metrics(project, now=None):
    now = now or datetime.now(timezone.utc)
    registry = CollectorRegistry()
    summary = recorded_summary(project)
    Gauge('va_evaluation_available','Recorded shared validation comparison is available',registry=registry).set(int(bool(summary and summary.get('models'))))
    scores = Gauge('va_validation_score','Last recorded shared validation metric; not live accuracy',
                   ['model_version','dataset_version','metric','cohort'],registry=registry)
    class_scores = Gauge('va_class_validation_score','Recorded per-class validation metric; not live accuracy',
                         ['model_version','class_name','metric','cohort'],registry=registry)
    if summary:
        for model in summary.get('models',[]):
            for metric,value in model.get('metrics',{}).items():
                if value is not None:
                    scores.labels(str(model['version']),model['dataset_version'],metric,summary['cohort']).set(value)
            for row in model.get('per_class',[]):
                for source,metric in (('Box-P','precision'),('Box-R','recall'),('Box-F1','f1'),('mAP50','map50'),('mAP50-95','map50_95')):
                    if row.get(source) is not None:
                        class_scores.labels(str(model['version']),row['Class'],metric,summary['cohort']).set(row[source])
    receipt = read_json(project/'artifacts/serving/deployment.json')
    drift = read_json(project/'reports/faculty-drift/summary.json')
    valid = False
    if drift and receipt and not (project/'artifacts/serving/deployment.lock').exists():
        try:
            valid = datetime.fromisoformat(drift['current_valid_until']) >= now and all(
                drift['identity'].get(k)==receipt['model'].get(k) for k in
                ('model_name','model_version','checkpoint_sha256','dataset_dvc_md5','inference'))
        except (ValueError,KeyError,TypeError): pass
    Gauge('va_drift_available','Fresh drift evidence for the current deployment',registry=registry).set(int(valid and drift['status']=='available'))
    drift_value = Gauge('va_drift_value','Latest measured PSI or KS; absent when stale or incompatible',
                        ['feature','metric'],registry=registry)
    if valid:
        for feature,record in drift['features'].items():
            if record['status']=='available':
                for metric in ('psi','ks_statistic','ks_pvalue'):
                    drift_value.labels(feature,metric).set(record[metric])
        if drift.get('drift_detected') is not None:
            Gauge('va_drift_detected','Drift warning only; never sufficient for rollback',registry=registry).set(int(drift['drift_detected']))
    events = history(project)
    Gauge('va_rollback_evidence_available','A saved demonstration result exists',registry=registry).set(int(bool(events)))
    if events:
        event=events[0]
        Gauge('va_last_rollback','Last saved controlled demo result, not a current health claim',
              ['previous_model','recovery_model','result','reason','mode','completed_at'],registry=registry).labels(
                  event['previous_model'] or 'unknown',event['recovery_model'] or 'unknown',event['status'],
                  event['reason'],event['mode'],event['completed_at'] or 'unknown').set(1)
    return generate_latest(registry)
