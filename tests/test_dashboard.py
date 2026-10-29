"""Missing and stale evidence must not become healthy status or fabricated accuracy."""
from datetime import datetime,timedelta,timezone
import json
from pathlib import Path
import tempfile
import unittest
from prometheus_client.parser import text_string_to_metric_families
from src.monitoring.dashboard import evidence_metrics,project_status,recorded_summary,history


class DashboardTests(unittest.TestCase):
    def test_current_research_run_does_not_inherit_old_completed_evaluations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for name in ('protocol-v2-30e','shel5k-v1-30e'):
                run=root/'artifacts/training'/name
                (run/'fit').mkdir(parents=True)
                (run/'lineage.json').write_text('{"epochs_requested":30}')
                (run/'fit/results.csv').write_text('epoch,train/loss\n1,0.9\n')
            old=root/'reports/final-audit-20261026/comparison-with-protocol'
            old.mkdir(parents=True)
            (old/'comparison.json').write_text('{"status":"completed"}')
            state=project_status(root)['training']
            self.assertEqual(state['run'],'shel5k-v1-30e')
            self.assertEqual(state['epochs_recorded'],1)
            self.assertFalse(state['comparison_complete'])
            self.assertFalse(state['final_test_required'])
            self.assertIn('kept separate',state['note'])

    def test_continuous_recovery_is_visible_without_a_finished_session(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);report=root/'reports/live-demo-fixture';report.mkdir(parents=True)
            (report/'session.json').write_text(json.dumps({'source_versions':['1','4']}))
            (report/'automatic-recovery.json').write_text(json.dumps({'status':'recovered',
                'completed_at':'2026-10-28T12:00:00+00:00','decision':{'reasons':['readiness']}}))
            rows=history(root)
            self.assertEqual(len(rows),1)
            self.assertEqual((rows[0]['previous_model'],rows[0]['recovery_model']),('4','1'))
            (report/'result.json').write_text(json.dumps({'mode':'automatic','status':'stopped'}))
            self.assertEqual(len(history(root)),1)
            (report/'result.json').write_text(json.dumps({'mode':'automatic','status':'failed',
                'original_api_restored':{'model':{'model_version':'4'}}}))
            row=history(root)[0]
            self.assertEqual(row['status'],'recovered; session failed')
            self.assertEqual(row['normal_model'],'4')

    def test_new_comparison_requires_matching_cohort_and_registration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            def save(relative, value):
                path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(value))
            base={'cohort':'fixed','models':[{'version':'4','name':'Highest shared score',
                  'checkpoint_sha256':'old','dataset_version':'old-data'}]}
            save('reports/faculty-summary.json',base)
            record=lambda version,sha,data: {'model_version':version,'checkpoint_sha256':sha,
                'dataset_dvc_md5':data,'validation':{'combined':{'metrics':{'precision':.8,'recall':.4},
                'per_class':[{'Class':'person','Box-R':.3}]}},'cpu_timing':{'inference_median_ms':35.}}
            comparison={'status':'completed','selection_sha256':'cohort-hash','models':{
                'expanded_candidate':record('4','old','old-data'),
                'new_protocol_candidate':record('7','new','new-data')}}
            relative='reports/final-audit-20261026/comparison-with-protocol/comparison.json'
            save(relative,comparison)
            save('reports/final-audit-20261026/comparison/comparison.json',{'selection_sha256':'cohort-hash'})
            save('artifacts/training/protocol-v2-30e/lineage.json',{'epochs_requested':30})
            receipt={'verified':True,'model_version':'7','checkpoint_sha256':'new','dataset_dvc_md5':'new-data'}
            save('artifacts/training/protocol-v2-30e/mlflow.json',receipt)
            result=recorded_summary(root)
            self.assertEqual([row['version'] for row in result['models']],['4','7'])
            candidate=result['models'][-1]
            self.assertAlmostEqual(candidate['metrics']['f1_macro_pr'],.5333333333)
            self.assertEqual(candidate['latency_ms'],35.)
            self.assertIn('Experimental',candidate['acceptance'])
            self.assertEqual(json.loads((root/'reports/faculty-summary.json').read_text()),base)
            for changed in ({'status':'running'},{'selection_sha256':'different'},{'selection_sha256':None}):
                save(relative,{**comparison,**changed})
                self.assertEqual(recorded_summary(root),base)
            save(relative,comparison)
            save('artifacts/training/protocol-v2-30e/mlflow.json',{**receipt,'checkpoint_sha256':'mismatch'})
            self.assertEqual(recorded_summary(root),base)

    def test_missing_evidence_and_actual_scores(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'reports').mkdir()
            empty=evidence_metrics(root).decode()
            self.assertIn('va_evaluation_available 0.0',empty)
            self.assertNotIn('va_validation_score{',empty)
            self.assertEqual(project_status(root)['history'],[])
            (root/'reports/faculty-summary.json').write_text(json.dumps({'cohort':'fixture','models':[
                {'version':'9','dataset_version':'hash','metrics':{'precision':.71,'map50_95':.31}}]}))
            samples=[s for f in text_string_to_metric_families(evidence_metrics(root).decode()) for s in f.samples]
            scores={s.labels['metric']:s.value for s in samples if s.name=='va_validation_score'}
            self.assertEqual(scores,{'precision':.71,'map50_95':.31})

    def test_stale_or_wrong_model_drift_is_not_exported_as_current(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'reports/faculty-drift').mkdir(parents=True)
            (root/'artifacts/serving').mkdir(parents=True)
            now=datetime.now(timezone.utc)
            identity={'model_name':'test','model_version':'4','checkpoint_sha256':'a','dataset_dvc_md5':'b','inference':{}}
            (root/'artifacts/serving/deployment.json').write_text(json.dumps({'model':identity}))
            report={'current_valid_until':(now+timedelta(minutes=1)).isoformat(),'identity':identity,
                    'status':'available','drift_detected':True,'features':{'brightness':{'status':'available',
                    'psi':.4,'ks_statistic':.2,'ks_pvalue':.001}}}
            path=root/'reports/faculty-drift/summary.json';path.write_text(json.dumps(report))
            self.assertIn('va_drift_detected 1.0',evidence_metrics(root,now).decode())
            lock=root/'artifacts/serving/deployment.lock';lock.write_text('active switch')
            self.assertNotIn('va_drift_value{',evidence_metrics(root,now).decode())
            lock.unlink()
            stale=evidence_metrics(root,now+timedelta(minutes=2)).decode()
            self.assertIn('va_drift_available 0.0',stale)
            self.assertNotIn('va_drift_value{',stale)
            report['identity']={**identity,'model_version':'3'};path.write_text(json.dumps(report))
            self.assertNotIn('va_drift_value{',evidence_metrics(root,now).decode())
