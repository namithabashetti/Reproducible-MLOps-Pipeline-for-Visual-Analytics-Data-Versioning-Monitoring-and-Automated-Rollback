# Project guide

**Reproducible MLOps Pipeline for Visual Analytics: Data Versioning, Monitoring, and Automated Rollback**

This is the main beginner and demonstration guide for the existing Windows project.
Updated 29 October 2026. Training progress changes while the queue runs; the live
queue file and MLflow receipts take precedence over a written progress snapshot.
This is a college/research system. No production stable model has been approved.

All five requested retraining runs and research evaluations are complete. Retrained Models 1–5 correspond to registry versions 6–10. Model 5 had the highest validation mAP50–95 (55.58%); its reused-test score was 53.91%. See [the execution report](PROJECT_EXECUTION_REPORT.md) for all measured results.

All commands below use **PowerShell**, opened in VS Code's Terminal menu or as a
normal PowerShell window. Start with:

```powershell
Set-Location -LiteralPath 'F:\major project implementation'
```

The project already has Python 3.11.9, a local `.venv`, the dataset, pretrained
weights, Docker images and trained checkpoints. Do not download a second dataset
or replace the existing environment just to start the demonstration.

## 1. What the project does

Imagine a construction-site image containing two workers wearing helmets and
another worker with an uncovered head. The detector draws boxes around objects
it recognizes. The surrounding pipeline records which images, labels, code,
settings and weights produced that detector. It also runs a separate deployment
service, observes its health, and can restore a verified predecessor after a
qualified failure. Detection accuracy and service recovery are different claims:
a working recovery mechanism does not prove every worker was detected.

## 2. Helmet, head and person

The class mapping is fixed: **0 = helmet, 1 = head, 2 = person**. Head means an
uncovered head under the current mapping. Person means the visible person's body
box; a helmet box alone is not a person box. The corrected original-image labels
use SHEL5K helmet-object boxes. The supplied additional construction-site labels
retain their source convention, which may differ. Person detection and label
completeness remain limitations; this is not a certified safety system.

## 3. Images and YOLO labels

A dataset pairs images with answers used during learning. The original dataset
uses PNG images and Pascal VOC XML annotations. Derived training datasets use
PNG images and one text file per image. Each YOLO text row contains:

```text
class_id center_x center_y width height
```

The four coordinates are fractions of image width/height, between zero and one.
For example, `0 0.5 0.4 0.2 0.1` describes a helmet box centered halfway across
the image. Empty detections from a model are different from missing labels in a
dataset. Scripts verify geometry, classes, matching files, split membership and
publisher hashes; they cannot prove that every visible person was annotated.

The 5,000 original images match the published Kaggle release and SHEL5K image
hashes. SHEL5K v4 supplied corrected XML labels under CC BY 4.0. Attribution and
the annotation mapping are recorded in `data/source-provenance.json`, the derived
dataset's `annotation-provenance.json`, and `docs/research-quality-upgrade.md`.
No original image or annotation was edited in place.

## 4. Training, validation and testing

Training images update the weights. Validation images measure progress and select
the best saved checkpoint. Test images evaluate a checkpoint after those decisions
are frozen. Looking at test scores repeatedly and changing settings to improve
them turns the test into another development set.

The queue freezes all five finalized checkpoint hashes before its test phase.
Historical Model 5 already evaluated this test membership. New test reports are
therefore **reused held-out research results**, not a pristine external acceptance
test. The current five retraining jobs never train on this test membership. A truly
new independent final test requires additional, separately collected images.

## 5. The actual 70/15/15 split

For 1,000 images, a simple 70/15/15 example is 700 training, 150 validation and
150 testing images. This project has **7,693 retained images**:

| Partition | Images | Percentage |
|---|---:|---:|
| Training | 5,385 | 70.00% rounded |
| Validation | 1,154 | 15.00% rounded |
| Testing | 1,154 | 15.00% rounded |

Exact percentages are `count / 7693 × 100`; integer images cannot represent every
fraction exactly. The split seed is **42**. Related original images and known
additional-image variants are kept together. No recorded related group or image
ID crosses partitions. Source sequence IDs are not available for every web image,
so this does not prove that all unknown relationships were found.

The frozen corrected-label version is `data/expanded-shel5k-v1`, DVC identity
`ec398edfa361f9d9b4952019bb04411e.dir`. Its `manifest.json`, `selection.json` and
`input-sha256.json` record membership and identity. `configs/retraining.json`
binds every new job to this same dataset and seed.

| Partition | Helmet boxes | Head boxes | Person boxes |
|---|---:|---:|---:|
| Training | 14,801 | 5,700 | 20,713 |
| Validation | 4,048 | 1,360 | 4,358 |
| Testing | 3,576 | 1,398 | 4,459 |

## 6. Why there are five model names

All five historical models use **YOLOv8n**, not five different architectures.

| Historical label | Run directory under `artifacts/training` | What originally differed |
|---|---|---|
| Model 1 | `baseline` | Original labels; 3,500 train / 500 validation |
| Model 2 | `fulltrain-v1` | Revised labels; 3,500 train / 100 validation |
| Model 3 | `assisted-fulltrain-v1` | Proposed person labels; 3,500 train / 100 validation |
| Model 4 | `expanded-css-v1-60e` | Additional images; 6,020 train / 196 validation; 60 epochs |
| Model 5 | `protocol-v2-30e` | Grouped 70/15/15 protocol; 30 epochs |

The new queue retains five logical names while using the same corrected data for
all jobs. Model 1's new run is `shel5k-v1-30e`; the others are
`retrain-model2-30e`, `retrain-model3-30e`, `retrain-model4-60e`, and
`retrain-model5-70e`. First three runs have identical deterministic settings, so
identical metrics are expected rather than a defect. More epochs do not guarantee
better accuracy. Historical checkpoints and registry versions are preserved.

## 7. What the epoch targets mean

One epoch is approximately one pass over all training images. The requested
targets are **30, 30, 30, 60 and 70**, totaling 220 epochs. The laptop has a 4 GB
RTX 3050 GPU, so a persistent queue trains one model at a time. It does not shorten
the targets. Models 4 and 5 will take longer. The queue can outlive this chat, but
it cannot train while the laptop is shut down; keep AC power connected.

## 8. How weights change

Each new experiment starts from the existing local pretrained `yolov8n.pt`.
PyTorch compares predicted boxes/classes with labels, computes a loss, and
backpropagates gradients. AdamW uses those gradients to update weight tensors.
The approved settings remain image size 416, batch 4, AdamW, learning rate 0.001,
weight decay 0.0005, workers 0, AMP disabled and deterministic mode enabled.
`best.pt` is the validation-selected checkpoint. `last.pt` records the latest
training checkpoint; during an unfinished run it also supports optimizer resume.

## 9. Reading losses

`fit/results.csv` contains actual per-epoch box, classification and distribution
focal losses, validation losses, metrics and timing. Lower loss usually indicates
better fit to those labels, but is not a percentage accuracy. Training loss falling
while validation quality worsens can indicate overfitting. Compare equivalent
datasets and settings; do not compare loss values across different label policies
as if nothing else changed. The last completed CSV row is the recorded epoch,
not the partially running epoch printed in the progress log.

## 10. Precision, recall, mAP and confusion matrices

Precision asks how many reported detections were correct. Recall asks how many
labeled objects were found. IoU measures box overlap. AP combines precision and
recall over confidence cutoffs; mAP averages AP across classes. AP50 uses an IoU
threshold of 0.5. AP50–95 averages stricter thresholds from 0.5 to 0.95. A confusion
matrix shows class confusions, missed labels and background detections under its
evaluation rules. The queue's final evaluation requests real confusion plots and
stores the matrix with the metrics. It runs only after all checkpoints are frozen.

## 11. Why confidence is not accuracy

A box labeled 82% is that prediction's confidence score. It does not mean the
model is 82% accurate, nor that 82% of people were found. Accuracy claims require
ground-truth labels and a defined evaluation protocol. The UI displays real box
confidence and measured inference time. Recorded evaluation scores live on the
Pipeline evidence page and in MLflow, separate from live detections.

## 12. DVC and the approved storage replacement

The original PRD named GCS. The user subsequently approved **Google Drive** after
Backblaze transaction limits. This guide follows that approved configuration.
DVC pointer files identify dataset snapshots; image bytes stay outside Git. The
private Drive remote backs them up. Version identity and a verified restore are
stronger evidence than merely seeing a folder online.

```powershell
.\.venv\Scripts\python.exe -m dvc status
.\.venv\Scripts\python.exe -m dvc status data/expanded-shel5k-v1.dvc
```

These commands compare local files and their recorded dependencies; they do not
upload images. The corrected snapshot was restored from a separate Drive-only
cache and all **15,394 files** were verified. See
`reports/quality-upgrade-20261027/drive-delivery/result.json`.
For account reauthorization, the existing command is
`.\.venv\Scripts\python.exe scripts/gdrive_session.py auth`.
Complete Google sign-in yourself; never paste tokens or credentials into chat.
Remote pulls/pushes require that private account's secure local authentication.
Do not commit `.dvc/config.local`, OAuth files or environment secrets.

## 13. OpenCV preprocessing

OpenCV decodes each image, scales it down when needed, and pads it to 416×416
without distorting its aspect ratio. Padding uses value 114. Box coordinates are
transformed alongside the image and stored as normalized YOLO coordinates.
Derived PNGs are lossless. Original files remain in `dataset/`.

```powershell
.\.venv\Scripts\python.exe -m dvc repro
```

This reruns changed preparation stages and updates their lock records. Run it
only when preparation dependencies changed, **before** training or a presentation;
DVC temporarily removes outputs while rebuilding them. It is not a startup command.

## 14. YOLO and PyTorch detection

YOLO predicts boxes, classes and confidence in one model pass. In this project,
GPU training uses PyTorch; serving uses CPU inference so it can run alongside
training. The predictor uses the configured confidence threshold (currently 0.25),
IoU setting 0.70 and at most 300 detections. CPU inference duration varies with the
scene and other laptop work. It is not a hardware-independent speed benchmark.

## 15. MLflow experiments and registry

MLflow records completed experiments, parameters, metrics, source snapshots,
dataset identities and checkpoint hashes. Its registry gives each artifact a
version number. A logical name such as “retrained Model 1” is not the same thing
as registry version 1: the completed retrained Models 1 and 2 registered as
versions **6 and 7**, respectively. Later numbers must be read from their receipts,
not guessed. `mlflow.json` records each run ID and verified registry version.
Open http://127.0.0.1:5000/ to inspect the local database and artifacts.

## 16. FastAPI and the actual selected checkpoint

There are two intentionally separate inference paths:

- The main page sends image bytes to the local console's `POST /predict` on 8767.
  It uses the explicitly selected demo model in `src/serving/demo_selection.py`.
- The Docker deployment exposes `POST /predict` on 8000. The Pipeline evidence
  page proxies it through `/deployment/predict`; the rollback watcher observes it.

Both paths reuse the verified loader and image-processing functions. Demo selection
validates the registry bundle, checkpoint hash, classes and warm-up prediction
before replacing the in-memory selection. Failed loading keeps the previous model.
Predictions include the actual registry version, SHA-256, settings hash, selection
mode, image dimensions, boxes, classes, confidence and measured timing. The backend
enforces PNG/JPEG and the 4 MB limit; decoded pixel counts are also bounded.

## 17. Docker serving

Docker packages the deployment API and its dependencies. The selected immutable
image ID and verified model directory are recorded in a deployment receipt.
The model is mounted into the container; changing a dropdown label does not
change this mount. Demo selection in the console never runs a Docker deployment,
changes a stable alias or edits the saved rollback target. Bindings remain local.

## 18. GitHub Actions

The existing private repository's workflow performs tests and serving/monitoring
checks on its dedicated Windows runner. It is manually dispatched and defaults
to **no deployment**. Stable and experimental deployments have separate gates.
Do not cite an earlier green workflow run as proof that current uncommitted
changes passed CI. Current local test logs are in the current evidence folder;
remote CI status must be checked for the exact submitted commit.

## 19. Prometheus metrics

Prometheus scrapes the Docker API and console every ten seconds. The original
deployment metrics include request outcomes, latency, model identity and readiness.
The separate `va_demo_*` metrics describe the demonstration selector: requests,
errors, request time, inference time and the actual selected checkpoint. Counters
reset when their process restarts. A scrape being available does not prove model
accuracy. Open http://127.0.0.1:9090/ for raw queries.

## 20. Grafana monitoring

Open http://127.0.0.1:3000/d/visual-analytics-operations. Existing panels are retained.
The added demonstration panels distinguish the selected demo model and its traffic
from the Docker deployment's traffic. Empty graphs may mean no requests occurred
in the chosen time window; they must not be read as zero latency or perfect health.
Make predictions, wait for a scrape, and use a recent time window.

## 21. Evidently drift

The drift code compares brightness, sharpness and prediction-confidence
distributions using recorded statistical evidence such as PSI and KS. A changed
distribution may be a camera, lighting or scene change. It is not ground truth
that the detector is wrong. Old, expired or wrong-model evidence is unavailable,
not healthy. Drift alone never authorizes rollback in this project.

## 22. Candidate and stable roles

A candidate is an experimental artifact under evaluation. Stable means explicitly
approved for a deployment role; previous stable is the verified predecessor that
can recover that deployment. Finishing training or receiving a registry version
does not grant either approval. The main registry currently has no production
stable alias. The local recovery laboratory copies the registry and creates scoped
fixture aliases; these do not approve the original models for production.

## 23. Automatic rollback

The application starts the laboratory watcher without requiring a “Start rollback”
click. It deploys historical source Model 4 as its normal-settings candidate and
captures source Model 1 as the scoped predecessor. The copied registry calls that
candidate version 2; this does not overwrite historical trained Model 2.

The watcher checks current identity and fresh health evidence. Consecutive readiness
failures or a validated labeled AP50–95 drop of at least 0.05 can qualify. Labeled
replays run every five minutes against the same 250 monitoring images; their old
source labels remain imperfect. A single missed webcam object does not qualify.
Recovery uses the normal verified deployment procedure and checks three readiness
responses, twenty predictions and fresh monitoring identity. Only one attempt is
allowed per failed deployment. The fallback stays selected; there is no automatic
ping-pong to the failed candidate or invented earlier fallback.

Session evidence is under `reports/live-demo-<id>/`: `session.json`, `watch.json`,
scheduled audits and, after recovery, `automatic-recovery.json`. `result.json`
appears when a session stops or fails. Historical successful and failed tests are
retained. A past recovery followed by a watcher failure is not full session success.

## 24. How the demo selector works

Choose a model set, choose Model 1–5, then click **Load selected model**. “Existing
trained models” resolves the historical runs. “Retrained with corrected labels”
resolves the new queue runs only after successful registration. Unfinished models
are marked not ready; the backend also rejects selecting them. The actual loaded
model remains visible below the image. Choosing an option alone does not load it.
During a switch the page stops camera capture, invalidates old pending results,
waits for verified loading, and can resume the camera with the new model. Images
and frames go only to this laptop's loopback API and are not saved by this page.

## 25. Show Model 1 while Model 4 is selected

1. Open http://127.0.0.1:8767/.
2. Choose **Existing trained models**, then **Model 1**.
3. Click **Load selected model** and wait for “Loaded Model 1”.
4. Confirm registered version 1 and open **Verify the selected checkpoint**.
5. Click **Try a sample**, upload a supplied image, or start the camera.
6. Open **View prediction details** and point to the actual model version/hash.
7. Repeat with Models 2–5, using the same image to compare detections.

To demonstrate the newly corrected-label Model 1, select the retrained model set;
its registered version is 6. The JSON and model strip make this distinction explicit.
Use independently recorded validation scores to discuss quality, not a few chosen
example images or changing confidence values.

## 26. Start, pause and inspect the persistent queue

The requested queue completed on 29 October 2026: 30/30, 30/30, 30/30, 60/60 and 70/70 epochs, with final research evaluations. Do not resume a completed queue or start five GPU processes manually. The commands below explain inspection and recovery for future interrupted runs.
Inspect its state from any PowerShell terminal:

```powershell
.\.venv\Scripts\python.exe scripts/train_all_models.py status
```

This prints targets, completed epochs, status, process identities, log paths,
checkpoint paths and registrations. The authoritative file is
`artifacts/training/five-model-queue/status.json`. The supervisor uses an OS lock
to reject duplicate supervisors and checks for existing training processes before
launching another job. It holds a temporary system-awake request while active.

```powershell
.\.venv\Scripts\python.exe scripts/train_all_models.py pause
```

This requests a pause **between models**. It lets the active model finish; it does
not pretend to pause a GPU epoch safely in the middle. For a new detached start
or a restart after the supervisor has stopped:

```powershell
New-Item -ItemType Directory -Path 'reports/queue-launch' -Force | Out-Null
$stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
Start-Process -FilePath '.\.venv\Scripts\python.exe' -ArgumentList '-u','scripts/train_all_models.py','resume' -WorkingDirectory 'F:\major project implementation' -WindowStyle Hidden -RedirectStandardOutput "reports/queue-launch/$stamp.stdout.log" -RedirectStandardError "reports/queue-launch/$stamp.stderr.log"
```

The initiating terminal may close. The process keeps running and advances the
queue without this AI conversation. The queue reuses a still-running child by
verified PID/creation time/command. Do not run `resume` while the supervisor is
already active; its ownership lock will reject the duplicate.

## 27. Logs and interrupted training

Each job's `stdout` and `stderr` fields identify its separate logs. For example:

```powershell
$queue = Get-Content -LiteralPath 'artifacts/training/five-model-queue/status.json' -Raw | ConvertFrom-Json
$queue.jobs | Select-Object model_id,epochs,epochs_completed,status,stdout,stderr
```

The adopted Model 1 log remains
`reports/quality-upgrade-20261027/training-resumed.log`. Other job logs are inside
the queue directory. Preserve all logs and `best.pt`/`last.pt` files. If a job
fails, the queue records the error and stops rather than hiding it. On resume,
the trainer verifies input hashes, original settings, epoch history and optimizer
state before using `last.pt`. A missing/inconsistent checkpoint is a real blocker;
do not delete its evidence and call it a successful resume.

The existing direct resume command, used only when the queue is not running, is:

```powershell
.\.venv\Scripts\python.exe -m src.training.train --resume-run artifacts/training/retrain-model3-30e
```

Use the actual interrupted run directory. This foreground command needs its
terminal to remain open; the queue's detached launch is preferred. Completed
training with a registration-only failure should use the existing importer after
inspection, not repeat epochs:

```powershell
.\.venv\Scripts\python.exe -m src.tracking.mlflow_utils import --run-dir artifacts/training/retrain-model3-30e
```

## 28. Verify the model processing predictions

```powershell
Invoke-RestMethod 'http://127.0.0.1:8767/health'
Invoke-RestMethod 'http://127.0.0.1:8000/health'
```

The first response is the demo selection; the second is the separate Docker
deployment. Their versions may intentionally differ. Compare a prediction's
checkpoint SHA-256 with its run's `mlflow.json`, and verify its settings hash and
confidence threshold. The selector's persisted receipt is
`artifacts/serving/demo-selection.json`; the deployment receipt is
`artifacts/serving/deployment.json`. The selector never edits the latter.

## 29. Start and stop the local project

```powershell
.\scripts\start_demo.ps1
```

This checks Docker, starts monitoring and the saved deployment API, and starts
MLflow and the console if needed. It reuses an active automatic recovery session.
It does not launch training or perform remote data transfers. Console and tracking
processes are detached, so this startup terminal may close.

```powershell
.\scripts\stop_demo.ps1
```

This asks the recovery session to stop and restore its original deployment, waits
for the receipt, stops the verified console process, and stops the API and monitoring
containers. It intentionally leaves training and MLflow running. Do not force-remove
a deployment lock after a crash: inspect its owner and saved receipt first. Shutdown
or sleep can interrupt training despite closing no terminal; pause between models
and wait for the queue to acknowledge that state before shutting down.

## 30. Pages to open

| Page | URL | Purpose |
|---|---|---|
| Detection | http://127.0.0.1:8767/ | Model selector, real boxes and prediction identity |
| Pipeline evidence | http://127.0.0.1:8767/pipeline | Queue, evaluations, sample gallery and recovery controls |
| Grafana | http://127.0.0.1:3000/d/visual-analytics-operations | Deployment and separate demo traffic |
| MLflow | http://127.0.0.1:5000/ | Saved experiments and registry |
| Demo health | http://127.0.0.1:8767/health | Actual selected demo model |
| Deployment health | http://127.0.0.1:8000/health | Actual watched Docker model |

The main page retains **06 / Supporting evidence**, **Monitoring ↗** and
**Experiments ↗**. Test packs are available at `/faculty-pack` (four named examples)
and `/sample-pack` (twelve images), relative to the detection URL.

## 31. Common errors

- **Model not ready:** check its real training/registration status. Choose an
  available model set; do not expect an unfinished checkpoint to load.
- **Selection verification failed:** the previous model remains selected. Inspect
  the console stderr log and artifact hashes; never relabel its output as the new model.
- **No boxes:** this can be a true empty scene or a missed object. It does not
  independently prove failure. Try a known image and inspect recorded evaluations.
- **Camera unavailable:** check browser/Windows permissions or use an image upload.
- **Upload rejected:** use a valid PNG or JPEG no larger than 4 MB; changing the
  filename extension does not change the encoded image format.
- **DVC changed:** inspect the listed dependency or data changes. Reproduce before
  starting jobs; never bypass the fingerprint check to make training proceed.
- **Queue has failed:** read the specific job's stderr and saved error. Other models
  remain queued and old working checkpoints remain intact.
- **Lock or interrupted restoration:** preserve receipts and inspect the actual
  container/process. Do not blindly delete the lock or claim recovery passed.

## 32. Reproduce an experiment

Use the recorded Git/source snapshot, locked Python dependencies, initial-weight
hash, exact DVC snapshot, split seed, selection hash and lineage settings. Restore
the dataset through the approved private Drive authentication first if it is absent.
Run DVC status, then create a **new** run name so the old evidence is preserved:

```powershell
.\.venv\Scripts\python.exe -m src.training.train --expanded-data data/expanded-shel5k-v1 --run-name reproduction-model1-30e --epochs 30
```

Do this only with no active GPU queue. The trainer snapshots source files and
isolates input copies so Ultralytics caches cannot modify DVC data. Resume is
optimizer continuation, not a guarantee of bit-for-bit identity to an uninterrupted
run. Different hardware/software versions may also change results.

## 33. Complete college demonstration script

Start the project and open Detection. Say: “This page uses the model selected here;
the deployment service and recovery test are separate.” Load historical Model 1,
show its version/hash, and try a supplied image. Switch to Model 4 on the same image,
then to a completed retrained model. Explain that selected examples illustrate boxes,
while evaluation metrics measure a defined labeled dataset.

Open Pipeline evidence. Show the queue and distinguish running, completed,
registered and evaluated states. Explain the shared corrected dataset, exact epoch
targets, and reused-test limitation. Open MLflow to show actual run records. Open
Grafana and show both demo and deployment identities and request counters.

For recovery, wait for **Automatic protection is active** on Pipeline evidence.
Under Advanced controls, **Simulate service outage** stops the candidate container
only. The already-running watcher decides whether to recover; that button does not
choose or deploy the fallback. Wait for recovery verification, show the saved reason
and model change, and repeat a prediction against the deployment service. Confirm the
main demo selector remains unchanged. This is an explicit availability-failure test,
not a claim that normal images naturally broke model quality. Do not press it while
the session is still preparing or when the fallback has already been selected.

## 34. Final verification and limits

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe scripts/train_all_models.py status
.\.venv\Scripts\python.exe -m dvc status
```

Before presenting, verify the loaded demo identity, all supporting links, a real
sample prediction, separate Docker health, fresh monitoring and the current queue.
Before claiming all training finished, verify every requested epoch count, completed
validation, verified MLflow registration and final-evaluation receipts. A configured
or queued job is not a completed job. Final test/plot generation waits until all five
models finish. This requested queue completed all ten validation/test evaluation receipts on 29 October 2026.

Current implementation evidence is under `reports/quality-upgrade-20261027/`.
`browser-model-switches.json` records real browser predictions from all historical
models and available retrained models. `selector-isolation-verified.json` checks
that selection did not change aliases, deployment receipt or rollback target.
The exact final test logs and live recovery receipts must be consulted for their own
dates. Webcam hardware/permission behavior requires an actual local camera test;
do not claim a synthetic or image-upload test verified a physical camera.

Remaining research limits include incomplete source labels, imperfect person
detection, unknown upstream rights/history for some web images, undetected possible
scene relationships and absence of a pristine new external test. The original
public dataset release identity itself is now verified. None of the queue, UI or
recovery work grants production safety approval.
