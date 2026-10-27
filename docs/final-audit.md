# Final project audit — 27 October 2026

The **college demonstration works**: real image inference, Docker deployment,
monitoring, data/model identities and automatic rollback are verified. The fresh
70/15/15 training experiment completed 30 epochs and registered as model 5; its
shared comparison and reserved final test also completed. Model 4 remains live
because model 5 scored lower on shared validation. This is **not full production or three-class
quality acceptance**. No main stable alias has been approved.

Use [faculty-demo.md](faculty-demo.md) for the presentation and restart commands.
This report supersedes older completion summaries. Historical reports are retained
as evidence of their own runs, not proof of current service availability.

## A. Findings and fixes

| Audit finding | Change / current result |
|---|---|
| Original split was 70/10/10 plus monitoring, not 70/15/15 | Created a separate grouped 70/15/15 protocol; preserved historical data and checkpoints |
| Expanded experiment omitted original images outside its selected cohorts | Fresh protocol includes all 5,000 original images plus 2,693 retained additional images |
| Reassigning old training images contaminates old-model test comparisons | New test restricted to fresh protocol training; old models remain on their shared validation cohort |
| Existing comparison omitted model 3 and mixed implicit evaluation settings | Re-evaluated all four models on the same 196 images with explicit identical settings |
| Drift collector confused training dataset identity with monitoring population identity | Recorded those identities separately while retaining model/settings consistency checks |
| Labeled audit rejected different training dataset versions | Verify registered training selection hashes and reject train overlap; compare a shared labeled batch |
| UI/Grafana lacked requested evidence and timing | Added measured timing, confidence/classes, recorded evaluation/class scores, dataset counts, training progress and recovery history |
| Demo restoration exception could omit its final report | Always record a failed result when normal API restoration fails |
| Presentation instructions were scattered and stale | Consolidated into faculty-demo.md, shortened README and preserved retired guides in the backup archive |
| Training process ended during epoch 12 | Validated inputs and checkpoint, backed up history/weights and resumed from completed epoch 11 on 27 October |

## B. Data versions and split evidence

| Protocol | Training | Validation | Test | Monitoring / interpretation |
|---|---:|---:|---:|---|
| Original 5,000 | 3,500 | 500 | 500 | 500 separate monitoring images |
| Expanded v1, model 4 | 6,020 | 196 | 77 | Historical cohort, preserved |
| Fresh grouped protocol | 5,385 | 1,154 | 1,154 | Monitoring images are validation replay subsets |

Fresh total: 7,693; percentages approximately 70/15/15. Related groups remain in
one split. Old protected test membership remains test, and validation/monitoring
membership remains validation. Previously trained-on images reassigned to test
are listed in the manifest. They disqualify models 1–4 from using this new test
as held-out evidence. Grouping is based on the recorded duplicate/variant audit;
it is not a guarantee that every visually similar scene has been discovered.

Fresh DVC identity: `b179e337e576fbc23586ed9dc42dce10.dir`.
Evidence: `data/expanded-protocol-v2/manifest.json`, `input-sha256.json`,
`data/expanded-protocol-v2.dvc`, and `reports/final-audit-20261026/repartition.log`.

Additional source: supplied Construction Site Safety Roboflow export v28,
2,801 images before exclusions, 108 excluded, 2,693 retained. It contains
augmentation variants, not 2,693 independently verified new scenes. Supplied
README states CC BY 4.0; upstream rights were not independently verified.
Original Kaggle URL/version remains unknown. Label completeness remains limited.

## C. Actual completed models

The following scores all use the same frozen 196 validation images (100 original,
96 additional), at 416 pixels, confidence 0.001, IoU 0.7, max detections 300,
`rect=False`, no half precision or test-time augmentation.

| Version | Training | Precision | Recall | F1* | AP50 | AP50–95 | CPU median |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | Original labels, 30 epochs | 91.16% | 46.40% | 61.50% | 53.91% | 32.44% | 27.1 ms |
| 2 | Revised labels, 30 epochs | 90.57% | 47.27% | 62.12% | 54.46% | 32.63% | 26.6 ms |
| 3 | Proposed person labels, 30 epochs | 64.83% | 59.50% | 62.05% | 55.17% | 33.90% | 27.2 ms |
| 4 | Expanded data, 60 epochs | 87.02% | 66.26% | 75.23% | 73.30% | 44.91% | 28.8 ms |
| 5 | Fresh grouped protocol, 30 epochs | 80.23% | 65.93% | 72.38% | 71.52% | 42.25% | 29.4 ms |

F1* is the harmonic mean of macro precision and recall, not mean per-class F1.
Per-class metrics are saved and displayed separately. CPU timing uses 40 fixed
images, two warmups and two threads; it is not HTTP latency. Timing varies with
host load. Earlier gains are small and no significance is established. Model 5
is 2.67 percentage points below model 4 in aggregate AP and also lower for person
AP (25.67% versus 30.16%). Model 4 remains the normal experimental detector;
model 5 is retained as a completed experiment, not promoted.

Evidence: `reports/final-audit-20261026/comparison-with-protocol/comparison.json`.
All five models' metrics and comparison artifact were logged to their existing
MLflow runs under `shared_validation_20261027` / `shared-validation-20261027`.
The dashboard combines the earlier four-model summary with this completed,
cohort- and registration-matched comparison. No registry aliases were changed.
Older 46.14% / 32.20% results used a different evaluation setup; preserve them as
historical measurements rather than mixing them into the current table.

Fresh run `artifacts/training/protocol-v2-30e` uses pretrained YOLOv8n and the
corrected protocol. It was interrupted during epoch 12; 11 completed epochs and
optimizer state were recovered. The resume keeps original settings and backs up
pre-resume checkpoints, CSV, args and actual resume source. It does not claim
bit-identical results to uninterrupted training. All 30 epochs completed and
model 5 is registered with verified evidence. It is experimental and has not
replaced the live model or received production quality approval.

Model 5's one-time reserved test scored **50.25% AP50–95**, 78.05% AP50,
88.69% precision and 70.50% recall on 1,154 images. Person AP50–95 is 41.71%
and recall is 49.77%; label limitations remain. This is a different cohort from
the shared 196-image comparison, so its higher aggregate score does not mean
model 5 beats model 4. Models 1–4 are ineligible for this reassigned test split.
The checkpoint and settings were frozen before scoring; no tuning followed.
Evidence: `artifacts/training/protocol-v2-30e/final-test/{frozen-plan,metrics,input-sha256}.json`,
with metrics and artifacts logged to model 5's actual run under `protocol_final_test`
and `protocol-final-test`. The evaluation process exited successfully.

## D. Faculty test images and inference

`demo-images` contains four clearly named examples, a guide, supplied source notice
and per-image source/hash mapping. DVC identity:
`cadfa27fbab27487805d865320b45b75.dir`.
The main UI's **Download faculty images** returns the same seven files; ZIP bytes
were checked against local files. A copy is `reports/faculty-test-images.zip`.

The upload interface was exercised with `test_normal_01.png`: normal model 4
returned 15 detections. During the confidence-1.0 candidate fault it returned zero.
A corrupt image was rejected with HTTP 422. The API returns measured inference
and processing times; the UI displays inference time with actual boxes.
Selected examples are illustrative validation images, not independent test scores.

## E. Automatic rollback evidence

| Real integration | Result | Evidence |
|---|---|---|
| Model 1 → source model 4 with explicit confidence-1.0 fault → model 1 | Passed; serving AP50–95 42.086% → 0% → 42.086% | `reports/live-demo-d949875bec7e/result.json` |
| Model 1 → source model 4 → actual stopped container → model 1 | Passed; readiness failure triggered rollback | `reports/live-demo-2fb9ea90983a/result.json` |

Both were started through the browser controls and used the normal deployment
procedure. Each verified three readiness checks, 20 real predictions, fresh
Prometheus monitoring, one durable recovery attempt and no retry swap-back.
Both then restored normal experimental model 4 and preserved empty main-registry
aliases. Quality mode also re-measured the same 250 labeled images after recovery.

This demonstrates the real mechanism under explicit controlled faults. It does
not claim natural deterioration, automatic ground truth from a webcam, or a
production-approved stable target. Drift alone and missing/stale evidence do
not trigger rollback. Thresholds remain initial calibration policies.

## F. UI and monitoring

The console includes the requested project title, camera/upload controls, named
examples, simple failure explanation, live model identity, inference timing,
actual model/class comparisons, dataset counts, recorded training progress and
recovery history. Advanced controls and detailed class scores are collapsible.
Outage tests explicitly state that AP scores do not apply.

Grafana has 20 panels: availability, readiness, identity, counts, request rate,
server errors, p95, RPM, mean and p99 response time, confidence by class, detected
objects by class, inference time, recorded model/class evaluations, last rollback,
drift availability, PSI/KS, request outcomes and all API errors.
Recorded quality is distinct from live accuracy. Stale drift is unavailable.
Latency/error ratios without observations are absent rather than fabricated zero.

Both Prometheus targets were confirmed up. All 20 panel queries returned real series during a bounded prediction replay; see `dashboard-active-20261027.json` in the audit directory. Fresh drift capture on 27 October used
250 reference and 250 current API predictions. It produced available evidence and
no detected drift for those batches; `reports/faculty-drift/summary.json` records
its actual expiry. Earlier drift evidence is preserved under the audit directory.
Do not treat a historical no-drift result as a permanent health statement.

## G. Verification and current runtime

- 111 local tests passed in 57.016 seconds, including resume artifact tampering
  and dashboard comparison/registration checks:
  `reports/final-audit-20261026/final-unit-tests-post-training-20261027.log`.
- Compose/Prometheus validation and active/idle PromQL regression passed.
- JavaScript syntax validation passed; browser upload, fault and recovery controls
  were exercised against the actual local service.
- CPU Docker image built and used for both successful integrations:
  `sha256:f2dc3db9948ac39fa7845915cfb376d66d63af02034b9bda23b5dca3d6b81d6a`.
- `scripts/start_demo.ps1` successfully restored the services after the overnight
  interruption; log `restart-20261027.log`. Normal checkpoint identity is model 4.
- CUDA tensor/NMS checks passed before resume. The charger is connected.
- Original model 1–4 checkpoint SHA-256 values are unchanged. DVC status for raw, expanded v1, fresh protocol and faculty images is clean (`{}`).
- Dependency consistency (`pip check`), source compilation and JavaScript syntax checks passed.
- Final-test isolation guards passed unit tests. The actual 1,154-image test also
  completed successfully; registry retrieval, data hashes and membership checks passed.

GitHub Actions remains the existing manual workflow on the dedicated Windows
runner and private repository. Older successful CI runs do not validate these
latest uncommitted changes. No new remote CI run or publication is claimed.

## H. Cleanup and preservation

`before-changes.zip`, `before-sha256.json` and `before.patch` retain the initial
source/configuration/docs. `vscode-demo.md` and `demo-handoff.md` were consolidated
and moved into `reports/final-audit-20261026/retired-guides`; links were updated.
README is now a current entry point rather than an accumulated session history.
Original datasets, completed checkpoints, registry artifacts and audit reports
were preserved. The supplied raw archive is explicitly Git-ignored; derived
snapshots use DVC. Legacy B2 scripts stay because historical recovery tests use
them. The active restore default is Google Drive.

## I. Completed evaluation and remaining delivery items

1. Completed: resumed 30-epoch run registered as experimental model 5, MLflow run
   `1227f13a5c404b198a329e70cb7b929d`, verified checkpoint
   `27bb78ba5241c1acc95d38e4601643c497fd32ed4e9540d70d324135a8118f6d`.
2. Completed: five-model shared validation and the new protocol's one-time
   1,154-image final test. The fixed test output refuses repeats/overwrites.
3. Reviewed: model 5's aggregate and person AP are lower on the shared cohort.
   Model 4 remains the live detector. All five measured results are displayed.
4. New local DVC snapshots have not been uploaded/restore-verified remotely, and
   current source changes have not had a new GitHub CI run.

The original waiting job was
`reports/final-audit-20261026/finish-evaluation.ps1`; its logs are `queued-evaluation.log`
and `.err`. It ended before evaluation started. After confirming successful
registration and that no evaluation process or output existed, comparison was
started again on 27 October and completed; see `comparison-recovery-20261027.log`.
The subsequent reserved test completed; see `final-test-20261027.log`. No model
deployment or approval changed. Do not rerun or overwrite these evaluation outputs.

## J. Acceptance boundary

The working college demo is available now at http://127.0.0.1:8767/.
The newer training experiment, shared comparison and reserved final test are complete.
Production/model-quality acceptance remains withheld because of label quality,
person-detection limitations, original-source provenance and missing stable
approval. These are explicit limitations, not silently waived requirements.
Google Drive remains the user-approved remote; no paid transfer was performed.
