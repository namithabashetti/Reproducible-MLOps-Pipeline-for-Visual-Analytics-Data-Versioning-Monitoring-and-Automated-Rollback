# Project execution report

Completed training/evaluation: **29 October 2026, 16:52 IST**. All five research runs completed and their registry artifacts, evaluation receipts and live inference identities were verified. This is college/research completion, not production acceptance.

## 1. Training status

All five are YOLOv8n experiments. The three 30-epoch jobs use the same seed, data and settings and produced identical validation metrics. Models 4 and 5 use longer training schedules; improvement is not guaranteed.

| Model | Target | Completed epochs | Saved job state | Registry version |
|---|---:|---:|---|---|
| 1 | 30 | 30 | Completed and evaluated | 6 |
| 2 | 30 | 30 | Completed and evaluated | 7 |
| 3 | 30 | 30 | Completed and evaluated | 8 |
| 4 | 60 | 60 | Completed and evaluated | 9 |
| 5 | 70 | 70 | Completed and evaluated | 10 |

Model 4 recovered from a pre-epoch input-copy interruption and a later interruption after 21 epochs. Its saved checkpoint resumed successfully to 60 epochs. Model 5 then finished 70 epochs, and the queue evaluated all frozen checkpoints. Interrupted inputs and logs were preserved. The queue supervisor has exited normally; it is not supposed to remain running after completion.

Historical supervisor PID: `23720`; launcher PID: `23544`. Queue status is `completed_research_evaluation`. Historical error fields describe the earlier interruption and are not evidence of a new failure. Final completion receipts, epochs, registrations and the terminal queue state take precedence.

All runs use local YOLOv8n initialization, seed 42, image size 416, batch 4, workers 0, AdamW, initial learning rate 0.001, weight decay 0.0005, deterministic training and AMP disabled. Exact lineage/source copies and loss rows live inside each run. Best weights are selected on validation.

### Model 1: `shel5k-v1-30e`

- Best checkpoint: `artifacts/training/shel5k-v1-30e/fit/weights/best.pt`.
- Last checkpoint: `artifacts/training/shel5k-v1-30e/fit/weights/last.pt`.
- Epoch losses and metrics: `artifacts/training/shel5k-v1-30e/fit/results.csv`.
- Last recorded stdout: `F:/major project implementation/reports/quality-upgrade-20261027/training-resumed.log`.
- Last recorded stderr: `F:/major project implementation/reports/quality-upgrade-20261027/training-resumed.err`.
- Historical training job PID: `11420` (completed).

### Model 2: `retrain-model2-30e`

- Best checkpoint: `artifacts/training/retrain-model2-30e/fit/weights/best.pt`.
- Last checkpoint: `artifacts/training/retrain-model2-30e/fit/weights/last.pt`.
- Epoch losses and metrics: `artifacts/training/retrain-model2-30e/fit/results.csv`.
- Last recorded stdout: `F:/major project implementation/artifacts/training/five-model-queue/model-2-20261028-211033.stdout.log`.
- Last recorded stderr: `F:/major project implementation/artifacts/training/five-model-queue/model-2-20261028-211033.stderr.log`.
- Historical training job PID: `17596` (completed).

### Model 3: `retrain-model3-30e`

- Best checkpoint: `artifacts/training/retrain-model3-30e/fit/weights/best.pt`.
- Last checkpoint: `artifacts/training/retrain-model3-30e/fit/weights/last.pt`.
- Epoch losses and metrics: `artifacts/training/retrain-model3-30e/fit/results.csv`.
- Last recorded stdout: `F:/major project implementation/artifacts/training/five-model-queue/model-3-20261028-231547.stdout.log`.
- Last recorded stderr: `F:/major project implementation/artifacts/training/five-model-queue/model-3-20261028-231547.stderr.log`.
- Historical training job PID: `11804` (completed).

### Model 4: `retrain-model4-60e`

- Best checkpoint: `artifacts/training/retrain-model4-60e/fit/weights/best.pt`.
- Last checkpoint: `artifacts/training/retrain-model4-60e/fit/weights/last.pt`.
- Epoch losses and metrics: `artifacts/training/retrain-model4-60e/fit/results.csv`.
- Last recorded stdout: `F:/major project implementation/artifacts/training/five-model-queue/model-4-20261029-094226.stdout.log`.
- Last recorded stderr: `F:/major project implementation/artifacts/training/five-model-queue/model-4-20261029-094226.stderr.log`.
- Historical training job PID: `8804` (completed).

### Model 5: `retrain-model5-70e`

- Best checkpoint: `artifacts/training/retrain-model5-70e/fit/weights/best.pt`.
- Last checkpoint: `artifacts/training/retrain-model5-70e/fit/weights/last.pt`.
- Epoch losses and metrics: `artifacts/training/retrain-model5-70e/fit/results.csv`.
- Last recorded stdout: `F:/major project implementation/artifacts/training/five-model-queue/model-5-20261029-114910.stdout.log`.
- Last recorded stderr: `F:/major project implementation/artifacts/training/five-model-queue/model-5-20261029-114910.stderr.log`.
- Historical training job PID: `16376` (completed).

## 2. Dataset verification

Dataset: `data/expanded-shel5k-v1`, DVC `ec398edfa361f9d9b4952019bb04411e.dir`.
Manifest SHA-256: `fca52b54f09b44e4454a2369a52140157c821bd8320fe1ab34fcd8ce9393f7e3`. Seed: **42**. Total: **7,693 images**.

| Partition | Images | Percentage | Helmet boxes | Head boxes | Person boxes |
|---|---:|---:|---:|---:|---:|
| train | 5385 | 69.99870% | 14801 | 5700 | 20713 |
| validation | 1154 | 15.00065% | 4048 | 1360 | 4358 |
| test | 1154 | 15.00065% | 3576 | 1398 | 4459 |

Integer counts explain the small rounding difference from exact 70/15/15. Class IDs are 0 helmet, 1 head, 2 person. All classes occur in every partition. The recorded 636 related-image groups stay in one partition; duplicate IDs and group overlap are rejected. Publisher hashes, image/label pairs and YOLO geometry were checked. Undiscovered near-duplicates and unknown sequence identities remain possible. Structural validation cannot prove every visible object was annotated.

Original data is unchanged. The original release matches Kaggle version-1 archive metadata and publisher image hashes; `data/source-provenance.json` records the evidence. SHEL5K v4 supplies corrected source labels with attribution. This verifies release identity, not the complete rights/history of every original photograph. Google Drive remains the approved remote. Restore verification checked 15,394 files; see the Drive receipt in this evidence folder.

## 3. Actual evaluation results

All scores below are proportions. Validation selected checkpoints; the reused test was scored only after all five checkpoints were frozen.

| Model | Val precision | Val recall | Val mAP50 | Val mAP50–95 | Test mAP50 | Test mAP50–95 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0.877488 | 0.765572 | 0.844072 | 0.538146 | 0.805385 | 0.520420 |
| 2 | 0.877488 | 0.765572 | 0.844072 | 0.538146 | 0.805385 | 0.520420 |
| 3 | 0.877488 | 0.765572 | 0.844072 | 0.538146 | 0.805385 | 0.520420 |
| 4 | 0.878171 | 0.780079 | 0.854900 | 0.548570 | 0.820369 | 0.534036 |
| 5 | 0.879794 | 0.790774 | 0.862009 | 0.555811 | 0.829618 | 0.539095 |

Model 5 is the strongest by validation mAP50–95. Its test mAP50–95 is 0.539095. The first three deterministic repeat runs have identical metrics. Different checkpoint file hashes include saved run metadata and do not guarantee different predictions.

| Model | Helmet test AP50–95 | Head test AP50–95 | Person test AP50–95 |
|---|---:|---:|---:|
| 1 | 0.52343 | 0.46268 | 0.57515 |
| 2 | 0.52343 | 0.46268 | 0.57515 |
| 3 | 0.52343 | 0.46268 | 0.57515 |
| 4 | 0.53516 | 0.47499 | 0.59195 |
| 5 | 0.53840 | 0.48439 | 0.59450 |

Each run contains complete per-class precision, recall, F1, AP, 4×4 confusion-matrix values, matrix plots and curves at `artifacts/training/<run>/final-evaluation/{validation,test}/`. The corresponding metrics and plot artifacts were verified in MLflow. Epoch losses and timing remain in each `fit/results.csv`; interruption/resume records explain any discontinuous elapsed-time counter.

**Test limitation:** Historical Model 5 already consumed these test images. These are reused held-out research results, not a pristine external acceptance test. Label completeness, unknown sequence similarity and upstream photo-rights history remain limitations.

## 4. Registry status

- Model 1: run `bc11eec4ee994b7fa965be3097c5d781`, registry version **6**, checkpoint SHA-256 `63be0621156f5fd970fed65506674420821809a7293b0dc49d59d547083ed7f6`.
- Model 2: run `90770211ee854237b0c752598f8d4bf7`, registry version **7**, checkpoint SHA-256 `267fe449ed5d927950e0bdb83cf3272b568a2c4d0702c4d623b0462c19b3f77e`.
- Model 3: run `7f6ddda9522b45458cd263168b9408b1`, registry version **8**, checkpoint SHA-256 `a3757214e107513b9326180bae2b328c83fdda82a6fabdafff8103cbe871f492`.
- Model 4: run `6971922fe0334b7ca1388f1aca56b360`, registry version **9**, checkpoint SHA-256 `14a5ca4b61f5560c5d9cf501129e478bd0c7b4fad3d597303b04953c8d5b4103`.
- Model 5: run `98536cca9d7b43ac93c46904297cc400`, registry version **10**, checkpoint SHA-256 `6cf98f8d2b01a473e55df8491fd3efba4661dbcb7f6c2d47ada2533688d8dabf`.

All five are experimental registrations. Historical model artifacts remain intact. The demonstration page has Model 5 / registry version 10 selected after live inference verification. This does not promote it to production or alter the rollback target.

## 5. Independent background execution

`scripts/train_all_models.py` runs sequentially in a detached Windows process. It records logs, process identity, errors, exit codes and checkpoints, locks against duplicate supervisors, and temporarily prevents idle sleep. It needs power and a running Windows session; it cannot survive shutdown as a live process. Resume validates completed artifacts and resumes an interrupted epoch checkpoint when available. Pre-epoch incomplete copies require inspection, as performed today.

From PowerShell in `F:\major project implementation`:

```powershell
.\.venv\Scripts\python.exe scripts/train_all_models.py status
```

Use the detached resume command in chapter 26 of `PROJECT_GUIDE.md` after an interruption. Do not run a second training command while this supervisor is alive. The pause command pauses between models. No remote weights, dataset download or token/chat connection is needed to advance training.

## 6. Demonstrate a selected model

1. Open http://127.0.0.1:8767/.
2. Choose **Existing trained models**, then **Model 1**, then **Load selected model**.
3. Check **Active: Model 1**, registry version 1 and **Verify the selected checkpoint**.
4. Click **Try a sample** or upload PNG/JPEG; expand prediction details to see the checkpoint used for that request.
5. Repeat with Models 2–5. To use newly trained models, choose **Retrained with corrected labels**; only completed valid registrations are enabled.

Switching verifies the actual registered weights, warm-up and checksum before replacing the selected demo model. Loading failure preserves the previous selection with an error. Each prediction carries its real model ID/version/checksum and measured inference time. Confidence is not accuracy. Demo selection does not change Docker serving, stable aliases or the rollback target. Physical camera capture and model switching were verified on 29 October: live 640×480 inference switched from retrained Model 5 / version 10 to Model 4 / version 9, resumed successfully, and stopped. Only model/dimension/status metadata was saved; no camera frames were saved.

Testing images are available from **Download 4 faculty images** and **Download 12 test images** on the main page. Their endpoints are `/faculty-pack` and `/sample-pack`. These are examples, not images guaranteed to cause a rollback.

## 7. Dashboard changes

The main page now focuses on model selection and detection in the requested order, with actual prediction details and simple limitation/privacy text. Section **06 / Supporting evidence**, Monitoring and Experiments links remain. `/pipeline` retains training, comparisons, testing images and recovery controls. The 4 MB PNG/JPEG restriction is enforced by the backend. Camera frames go to the local backend and are not saved by this page.

## 8. Monitoring and rollback verification

Existing Grafana panels are preserved; four demo-model panels were added, for 24 panels total. Docker deployment and demo-selection metrics remain distinct. MLflow runs and health endpoints remain available. The automatic watcher starts with the application; it does not need the rollback-start button. A single missed webcam detection or drift alone is not a rollback trigger.

The controlled-outage session `reports/live-demo-fbd9e2316693` recovered automatically after three readiness failures. Restoration passed three readiness checks, 20 image predictions and Prometheus identity verification. A later saved watcher receipt advanced 1,335 seconds after recovery while retaining the fallback. With no earlier approved target, further rollback is unavailable instead of cycling between models. This is operational recovery evidence, not proof of detection accuracy.

A later all-service interruption was handled by checking absent process ownership and exact saved Docker container/image identity, preserving the old lock, validating the original checkpoint and restoring the normal deployment. The old session remains recorded as interrupted then restored. See `interruption-recovery-20261029.json` and `automatic-recovery-verified.json` in the evidence folder. The normal stop/start scripts were then exercised successfully; training and MLflow were left running.

The latest controlled outage also passed a live 90-second continuation check: `reports/startup/rollback-live-verification-20261029.json`. The watcher independently restored existing Model 4 to existing Model 1 after persistent readiness failure. This verifies operational rollback. The stale-lock status bug was fixed so an interrupted watcher reports `running: false`; the main detection page now displays protection status. Telco Churn was stopped with explicit user approval to free ports 8000/3000. It must not run on those ports simultaneously with this project.

## 9. Cleanup

Only `docs/faculty-demo.md` was removed after consolidation. Its original copy and deletion manifest are in `reports/cleanup-20261029/`. The interrupted input-copy directory was renamed and preserved, not deleted. Dataset files, labels, DVC metadata, all checkpoints, MLflow records, active logs, monitoring, Docker and CI configuration remain.

## 10. Beginner guide

`F:\major project implementation\PROJECT_GUIDE.md` contains the 34 requested topics, actual Windows commands, model roles, dataset definitions, monitoring, rollback, resume instructions and a faculty demonstration sequence.

## 11. Checks performed

Final post-training suite: **131 tests passed in 48.750 seconds**, using `.\.venv\Scripts\python.exe -m unittest discover -s tests`. Log: `reports/startup/final-post-training-tests-20261029.log`. All five retrained models were then individually loaded and used for real sample predictions. Registry checkpoint hashes and MLflow evaluation artifacts matched; deployed checkpoint and main registry aliases remained unchanged. Evidence: `reports/startup/final-model-verification-20261029.json`. Earlier checks below remain historical evidence.

- Prior full local suite: **131 tests passed** (`final-selector-tests.log`).
- Latest selector tests: `python -m unittest discover -s tests -p test_demo_selection.py` — **4 passed**.
- Latest queue tests: `python -m unittest discover -s tests -p test_training_queue.py` — **3 passed**.
- An initial dotted-module test invocation failed to locate the non-package tests directory; corrected discovery commands above passed.
- `node --check` passed for both extracted dashboard scripts; `git diff --check` passed.
- Browser switching tested all five historical models and retrained Models 1–3; real prediction SHA values matched. Production receipts and aliases stayed unchanged (`browser-model-switches.json`, `selector-isolation-verified.json`).
- Main selector reload now matches the persisted selected demo generation/version. Model 3 / registry version 8 produced five detections in the latest real browser test.
- Both test-image ZIP downloads passed integrity checks with 4 and 12 images. Grafana returned 24 panels; MLflow, both health endpoints and the demo Prometheus metrics responded successfully (`restart-live-checks-20261029.json`).
- Controlled outage evidence and subsequent restored serving verified as described above.
- `scripts/stop_demo.ps1` and `scripts/start_demo.ps1` passed an actual stop/restart cycle; logs are in the evidence folder.

## 12. Remaining limitations and delivery status

- All requested epochs, final research evaluations, registry records, confusion matrices and five-model inference checks are complete.
- The test pool is reused; independent external acceptance would require new separately collected images.
- Publisher-label completeness and full upstream photo-rights history are not independently established. Research metrics do not establish production suitability.
- Physical webcam capture, live inference, switching and stopping passed. Metadata-only evidence: `reports/startup/camera-check-20261029.json`. Detection accuracy in a real deployment site still requires separate labeled evaluation.
- Current source delivery/CI status is recorded separately; earlier successful GitHub Actions runs cannot validate a different revision.
- No production promotion, automatic retraining or extra dataset transfer was performed in this finalization.
