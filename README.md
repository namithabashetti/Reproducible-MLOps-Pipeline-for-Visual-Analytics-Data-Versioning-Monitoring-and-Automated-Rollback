# Reproducible MLOps Pipeline for Visual Analytics

This college major project connects dataset versioning, object detection, model
tracking, local serving, monitoring, and recovery. Its central rule is that drift
is evidence for a health decision: drift alone never triggers rollback.

**Current status: Phase 11 rollback controls passed all 53 local and GitHub CI tests, live recovery-probe checks, monitoring validation, and the CPU image build. The model remains experimental, with no approved rollback target.**
The dataset is split into 3,500 train, 500 validation, 500 final test, and two
250-image monitoring pools. OpenCV produced 5,000 padded 416x416 PNGs and matching
YOLO labels. Independent checks verified every image and all 25,502 boxes, with
raw bytes/modification times unchanged and no candidate group crossing splits.
Evidence: `reports/phase3-validation.json`; all 39 current tests passed locally.
Google Drive remains the default DVC remote, with full raw-dataset recovery
verified in `reports/gdrive-restore.json`. Derived outputs are cached locally.
B2 is retained as a legacy remote with incomplete recovery evidence.
The CUDA smoke run and 30-epoch baseline completed with finite losses and metrics.
Best-checkpoint validation mAP50 is 61.94%, but person detection is weak
(1.94% per-class mAP50). This is an experimental baseline, not an approved stable
model. MLflow now records this baseline as `hard-hat-detector` version 1.
Local tracking, experimental CPU serving, and operational monitoring are verified.
Local drift reporting, health decisions, and bounded rollback controls are implemented. Neither `stable`
nor `previous_stable` has been assigned.

## Approved lifecycle

```text
Existing Kaggle-attributed dataset (exact provenance unverified)
  -> DVC + Google Drive
  -> OpenCV preprocessing
  -> YOLOv8n + PyTorch training and evaluation
  -> MLflow Tracking + Model Registry
  -> GitHub Actions
  -> Docker + FastAPI on this laptop
  -> Prometheus/Grafana operations + Evidently PSI/KS drift evidence
  -> deterministic Health Decision
       healthy   -> continue monitoring
       unhealthy -> previous approved stable MLflow model
                 -> same deployment procedure -> verify -> resume monitoring
```

Google Drive is the user-requested replacement for Backblaze B2 after B2 recovery
hit a daily transaction cap. B2 had previously replaced GCS in the original PRD.
FR-03 and amended AC-02 now use Drive. The original PRD remains unchanged; see
[the decision record](docs/decisions.md).

## Repository responsibilities

| Path | Purpose and pipeline position |
| --- | --- |
| `dataset/` | Original image/XML inputs, preserved unchanged; bytes tracked by DVC rather than Git. |
| `dataset.dvc` | DVC dataset identity; Git stores this pointer instead of raw data. |
| `.dvc/config` | B2 and Google Drive destination configuration; no credentials. |
| `src/data/validate.py` | Read-only image/annotation validation and content fingerprint. |
| `src/data/verify_restore.py` | Fresh-workspace/cache recovery and content comparison. |
| `src/data/split.py` | Validate raw inputs, group similar images, and create deterministic split manifests. |
| `src/data/preprocess.py` | OpenCV padding/downscaling and corresponding VOC-to-YOLO box conversion. |
| `src/training/train.py` | CUDA checks, isolated train/validation copies, source/data lineage, and YOLOv8n training. |
| `src/training/evaluate.py` | Checkpoint evaluation on validation data, global/per-class metrics, and checkpoint hash. |
| `src/tracking/mlflow_utils.py` | Validate/import completed training evidence, register an experimental version, verify retrieval, and launch local MLflow. |
| `src/serving/api.py` | Verify/load the staged checkpoint, warm up CPU inference, validate PNG/JPEG requests, and return boxes in original-image coordinates. |
| `src/serving/deploy.py` | Resolve registry aliases or an explicit experimental version, verify all evidence, stage bytes, run Docker, and verify readiness/prediction. |
| `Dockerfile` / `.dockerignore` | Build the CPU runtime from an allowlisted context; exclude datasets, weights, credentials and Git history. |
| `requirements-serving.txt` | Separate pinned Linux CPU serving dependency closure, checked by pip inside the image. |
| `compose.monitoring.yaml` | Local Prometheus/Grafana containers, pinned images, loopback ports, and persistent monitoring volumes. |
| `configs/monitoring/prometheus.yaml` | Ten-second scrapes of the API through its shared Docker network name. |
| `configs/monitoring/grafana/` | Provision the Prometheus datasource and seven-panel read-only operations dashboard. |
| `tests/test_serving.py` | API status/identity, image bounds, coordinate mapping, checkpoint integrity, and deployment approval gates. |
| `.github/workflows/ci.yml` | Manual Windows validation: isolated checkout, locked dependencies, source compilation, tests, and commit/lock evidence. |
| `dvc.yaml` / `dvc.lock` | Data-stage commands, dependencies, parameters, and resulting output identities. |
| `scripts/b2_session.py` | Masked credential entry for real DVC commands; secrets remain in process memory. |
| `scripts/b2_window.ps1` | Local Windows credential dialog with Paste buttons and character counts. |
| `scripts/gdrive_session.py` | Local Google browser sign-in, storage-space check, and DVC transfer/verification. |
| `tests/test_data_validation.py` | Real PNG/XML fixtures for invalid-data and fingerprint behavior. |
| `tests/test_restore.py` | Real DVC integration regression for recovery below a Git-ignored directory. |
| `tests/test_preprocessing.py` | Group isolation, repeatability, pixel/box geometry, and raw-data protection. |
| `tests/test_training.py` | Isolated smoke selections, held-out exclusion, and evaluation report validation. |
| `tests/test_tracking.py` | Real temporary SQLite/registry tests for retry recovery, artifact integrity, and stable-alias preservation. |
| `configs/config.yaml` | Non-secret service/storage settings and approved initial health policies. |
| `params.yaml` | Dataset split/preprocessing parameters in DVC lineage and recorded training parameters. |
| `requirements.txt` | Exact direct dependencies needed by the current implemented phase. |
| `docs/decisions.md` | Approved choices, provenance limitations, credential policy, and remaining gates. |
| `.gitignore` | Keeps raw/derived data, environments, credentials, and model binaries out of Git. |
| `Major_Project_PRD_v2_with_Literature_Gap_Analysis.docx` | Original requirements and acceptance criteria. |

There are no empty stage modules. Source files will be added when their pipeline
stage is implemented. Processed data and local MLflow artifacts now exist. Validation
and recovery evidence is written to the ignored `reports/` directory.

For viva explanation, configuration files have no executable processing:

| File | Inputs | Processing/output | Dependencies and consumer |
| --- | --- | --- | --- |
| `configs/config.yaml` | Approved service, storage, class, and policy decisions | Declarative settings; no pipeline action | PyYAML; future data, serving, monitoring, health, and rollback code |
| `params.yaml` | Approved split/training choices | Declarative experiment parameters | PyYAML; DVC data stages and training entry point |
| `requirements.txt` | Dependencies required by the completed phase | pip installs the pinned direct dependency into `.venv` | Python/pip; local environment setup |
| `.gitignore` | Paths unsuitable for Git | Git excludes matching untracked paths; no deletion | Git; repository hygiene, with DVC pointers outside ignored outputs |
| `docs/decisions.md` | PRD, approvals, and inspection evidence | Records rationale, limitations, and gates | Human review; all implementation phases |
| `README.md` | Current verified project state | Setup, architecture, and explanation guide | Human review; entry point to the lifecycle |

## Local environment

Use Windows 11 and Python 3.11.9. In PowerShell, from the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip check
```

Check `py -3.11 --version` before creating a clean environment: it must resolve to
3.11.9 for the recorded setup. Calling the environment's Python directly avoids
requiring a PowerShell execution-policy change. The environment does not inherit
global site-packages. pip may reuse cached distributions, but the global Python
installation is not modified.

`requirements.txt` lists direct dependencies through Phase 8: PyYAML, Pillow, DVC,
its S3 and Google Drive integrations, NumPy, OpenCV, PyTorch, torchvision, and
Ultralytics, MLflow, FastAPI, Uvicorn and the official Prometheus client. `requirements-lock.txt` pins their resolved
Windows/Python 3.11 dependencies. FastAPI/Uvicorn were already present through
MLflow; naming them as direct serving dependencies did not change this lock.
The Linux image uses `requirements-serving.txt`, CPU PyTorch 2.2.2/torchvision 0.17.2,
and the same Ultralytics 8.3.230. Prometheus client 0.26.0 is pinned in both environments.

Docker Desktop 4.94.0, Engine 29.8.2, Compose 5.5.1, and WSL 2 were detected during
preparation. The Linux engine is reachable, and the user verified `hello-world`.
The Phase 7 CPU serving image has been built and exercised on this Linux engine.

The project environment uses torch `2.2.2+cu121`, torchvision `0.17.2+cu121`, and
Ultralytics `8.3.230`. The global CPU-only installation is untouched. GPU tensor
operations and torchvision NMS passed on the 4 GB RTX 3050 A Laptop GPU;
evidence is in `reports/gpu-check.json`. The one-epoch smoke training/evaluation
also passed (`artifacts/training/smoke/metrics.json`). Settings are 416 pixels, batch 4,
zero loader workers, and no image RAM cache. CPU fallback must be explicitly
selected and recorded; it must not silently replace a failed GPU run.

## Dataset inspection and preparation policy

The supplied dataset contains 5,000 RGB PNG images and 5,000 matching Pascal
VOC-style XML files, totaling 1,320,154,239 bytes. All images decoded in the
discovery inspection; XML dimensions matched the images and all 25,502 boxes had
positive area and lay within image bounds. No exact byte/pixel duplicates or
duplicate boxes within annotations were found. These checks do not establish
annotation completeness or rule out related scenes.

| Class ID | Class | Boxes | Images containing class |
| --- | --- | ---: | ---: |
| 0 | helmet | 18,966 | 4,581 |
| 1 | head | 5,785 | 920 |
| 2 | person | 751 | 158 |

Image dimensions are 416x415 (2,461 images), 416x416 (2,192), 415x416 (324), or
415x415 (23). Average encoded image size is about 256 KiB. There are 4,524 boxes
with a side shorter than 16 pixels and 4,187 boxes touching coordinate zero.
Do not blindly subtract one from every coordinate during VOC conversion.

The original data has no split manifest, YOLO labels, provenance metadata, or
preprocessing code. Exact Kaggle version and upstream preprocessing are
**unverified**. The probable source is listed in the decision record, not asserted
as verified provenance. Do not download another copy for this implementation.

Phase 2 validates the raw inputs before any derived data is created. Phase 3
produces YOLO labels and OpenCV-prepared images in a separate directory.
These near-square inputs are padded to the configured size, preserving their
pixels and applying matching box transforms. Raw files are not modified.

The initial split is 3,500 training, 500 validation, 500 monitoring, and 500 final
test images, using seed 42. Related images stay together under the documented
similarity screening. Preserve class representation and
document any count changes needed for grouping. The monitoring pool is divided
into disjoint 250-image reference/current pools. Keep the final test set out of
model selection, threshold tuning, and demonstration calibration.

## Phase 3: split and preprocess

Run the two local data stages from the repository root:

```powershell
.\.venv\Scripts\python.exe -m dvc repro
```

`dvc.yaml` uses this laptop's `.venv\Scripts\python.exe` explicitly, so activating
the environment is unnecessary. Commands are Windows-specific; the future Linux
serving container is separate. DVC records raw input hashes, code, the dependency
lockfile, selected parameters, and output hashes in `dvc.lock`. It skips unchanged
stages. To regenerate the derived output, use `dvc repro --force preprocess`;
DVC replaces that tracked output, and the script refuses to overwrite a nonempty
directory when invoked directly. Raw data is always a dependency, never a stage output.
These commands perform no cloud transfer or training.

The split stage revalidates raw data and screens every image pair using a
63-bit low-frequency DCT perceptual hash, including horizontal mirrors. Pairs
within 10 differing bits form connected groups. The 151 candidate pairs were
visually reviewed on 2026-10-08 using eight contact sheets under `reports/`.
They include obvious watermark/brightness/border variants and some incidental
matches. All candidates are conservatively kept together: 135 groups, 278 images,
largest group four. This screen is not proof of complete scene independence;
source-scene metadata is absent and more distant variants may remain undetected.

Whole groups are assigned deterministically with seed 42, largest groups first,
balancing class presence within the exact split capacities. No images are removed.
The initial counts are retained, with these class-presence results:

| Split | Images | With helmet | With head | With person |
| --- | ---: | ---: | ---: | ---: |
| Training | 3,500 | 3,206 | 644 | 110 |
| Validation | 500 | 458 | 92 | 16 |
| Monitoring reference | 250 | 230 | 46 | 8 |
| Monitoring current | 250 | 228 | 46 | 8 |
| Final test | 500 | 459 | 92 | 16 |

`data/splits.json` stores every membership, candidate pair, group, class count,
seed, and the raw content fingerprint. It is DVC-managed rather than embedded
in source code. The test set remains outside model/threshold selection, and
monitoring reference/current groups do not overlap. Stratification uses labels
only to allocate class representation, before any model or threshold tuning.

OpenCV writes separate 416x416 PNG images and YOLO text labels to
`data/processed/images/<split>/` and `data/processed/labels/<split>/`. The supplied
415/416-pixel inputs need only zero or one padding pixel per axis. Larger future
images are downscaled with preserved aspect ratio; smaller images are not enlarged.
Padding is centered with an extra odd pixel on the bottom/right, using value 114.
No denoising, color manipulation, or offline augmentation is applied.

Box coordinates use the actual resized dimensions and padding offsets, then become
normalized `class x_center y_center width height` rows. The validated raw zero-origin
coordinates are preserved without blindly subtracting one. Empty annotations produce
empty label files. Class IDs remain helmet=0, head=1, person=2. PNGs retain uint8 pixels;
the future YOLO loader handles RGB conversion and tensor normalization. See the
[Ultralytics label format](https://docs.ultralytics.com/datasets/detect/).

Use the generated `data/processed/dataset.yaml` for later training. Its paths resolve
relative to the YAML and expose only training and validation. Final test and monitoring
data require explicit selection in their later phases. `preprocessing.json` beside
the YAML records transformation settings, versions, counts, and content fingerprints.
Derived artifacts are cached locally by DVC; they have not been uploaded to Drive.
They can be regenerated from the already verified raw snapshot and the pinned code.

The independent all-file audit passed: all 5,000 images preserve the original
pixels with correct padding, and all 25,502 labels map back to the source boxes
within 0.00001 pixel after rounding. Raw bytes and modification times are unchanged.
Evidence is in `reports/phase3-validation.json`; annotated spot checks are in
`reports/preprocessing-preview.jpg`. Repeated fixture preprocessing produced
identical bytes, and a second `dvc repro` skipped both unchanged stages.

| File/function | Inputs | Processing/output | Dependencies and next stage |
| --- | --- | --- | --- |
| `split.py:find_groups` | Raw PNG/XML | Perceptual candidate graph and whole groups | OpenCV/NumPy, annotation validator; split assignment |
| `split.py:assign_splits` | Groups, class presence, counts, seed | Deterministic complete/disjoint memberships | NumPy, standard library; preprocessing |
| `preprocess.py:preprocess_dataset` | Raw data, manifest, class IDs, size/pad | Validates lineage; writes padded PNGs, YOLO labels, YAML, report | OpenCV, validator; YOLO training |
| `dvc.yaml` / `dvc.lock` | Raw snapshot, code, parameters, dependency versions | Reproducible stage execution and content identities | DVC; experiment lineage |
| `tests/test_preprocessing.py` | Small real images/XML and grouped fixtures | Checks boundaries, color, boxes, isolation, raw preservation, repeatable bytes | unittest, OpenCV/NumPy; phase acceptance |

## Phase 4: train and evaluate

After `dvc repro` reports current data, run the hardware gate and smoke test:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.training.train --check-only
.\.venv\Scripts\python.exe -X utf8 -u -m src.training.train --smoke
# Run the baseline after the smoke test succeeds.
.\.venv\Scripts\python.exe -X utf8 -u -m src.training.train
# For a later experiment, choose an unused name and reviewed parameters.
.\.venv\Scripts\python.exe -X utf8 -u -m src.training.train --run-name experiment-02
```

The smoke run uses one epoch on 64 training and 32 validation images, selected
deterministically with all three classes represented. The baseline uses 30
epochs on all 3,500 training and 500 validation images. Both start from the
official pretrained YOLOv8n checkpoint, downloaded once into
`artifacts/pretrained/`. No additional dataset is downloaded. Keep AC power
connected and the laptop awake during training.

The training parameters in `params.yaml` use 416 pixels, batch 4, zero loader
workers, no image RAM cache, FP32 (AMP disabled), AdamW with initial learning rate
0.001 and weight decay 0.0005, seed 42, and deterministic mode. Other training
defaults, including online augmentation, come from the pinned Ultralytics version
and are saved in each run's `fit/args.yaml`. Deterministic settings do not promise
identical results on different hardware/software versions.

Each run copies only train/validation inputs into `artifacts/training/smoke/` or
`artifacts/training/baseline/`. Label caches or library repairs therefore cannot
modify DVC-managed data. Existing run directories are refused rather than
overwritten. Final-test and monitoring pools are never selected here. Use an
explicit `--device cpu` only when choosing and recording CPU execution.

`lineage.json` records raw/processed/split DVC hashes, the initial checkpoint
SHA-256, package/hardware details, parameters, Git commit, and dirty-checkout
status. The `source/` snapshot and file hashes retain the actual code and
configuration, including uncommitted changes. `inputs/selection.json` records
exact input memberships. `fit/results.csv` holds epoch metrics; `fit/weights/`
holds best/last checkpoints. After training, `metrics.json` records an independent
validation pass of `best.pt`, precision, recall, mAP50, mAP50-95, per-class results,
and the evaluated checkpoint hash. These are validation-selection metrics,
not an unbiased final-test estimate. Smoke metrics are only pipeline evidence.
Precision and recall are class means at Ultralytics' validation max-F1 operating
point; they do not establish a serving confidence threshold. mAP50 uses IoU 0.50,
while mAP50-95 averages IoU thresholds 0.50 through 0.95.

Ultralytics automatic MLflow logging stays disabled. The project explicitly logs
completed full runs after checkpoint evaluation, so parameters, source/DVC lineage,
and validation results remain connected. Smoke runs are not automatically registered.
If tracking fails, completed training artifacts remain intact; retry the import
command below rather than repeating training. Registration never implies stable approval.

The baseline completed on 2026-10-09 with 4,178.35 seconds (69.6 minutes) recorded
through epoch 30, followed by checkpoint evaluation. Validation of `best.pt` on
500 images produced these class-mean results:

| Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: |
| 60.94% | 56.79% | 61.94% | 39.43% |

| Class | Validation images containing class | Boxes | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| helmet | 458 | 1,896 | 92.52% | 88.06% | 93.86% | 61.22% |
| head | 92 | 633 | 90.31% | 82.31% | 90.01% | 56.14% |
| person | 16 | 118 | 0.00% | 0.00% | 1.94% | 0.93% |

Person detection needs targeted error review and improvement before stable
promotion. Its low representation is a known limitation, not a proven sole cause
of this result. No class was removed, and no final-test/monitoring images were
used to improve these numbers. The model remains explicitly `stable_model: false`.

Evidence: `artifacts/training/baseline/metrics.json`, `lineage.json`,
`fit/results.csv`, `reports/phase4-validation.json`, and
`reports/phase4-training-results.png`. All 30 epochs have finite values; source
snapshot and checkpoint hashes match; train/validation memberships match the DVC
manifest. DVC reports no changed data stages, both output hashes match Phase 3,
and raw file sizes/modification times are preserved. The project-local settings
directory is now created explicitly before Ultralytics import; generated settings
and fonts are retained under ignored `artifacts/ultralytics-settings/`.

| Function | Inputs | Processing/output | Dependencies and next stage |
| --- | --- | --- | --- |
| `train.check_device` | Explicit CUDA/CPU device | Tensor arithmetic and NMS probes, hardware/version report | PyTorch/torchvision; training gate |
| `train.prepare_data` | Processed data, split manifest, smoke/full choice | Independent working copies, selected memberships, training YAML | Standard library/PyYAML; YOLO loader |
| `train.main` | Parameters, DVC data, pretrained checkpoint | Source/lineage snapshot, training, loss checks, checkpoint evaluation | DVC/PyTorch/Ultralytics; Phase 5 tracking |
| `evaluate.evaluate` | Checkpoint, validation YAML, device/size/batch | Real precision/recall/mAP, per-class report, checkpoint identity | Ultralytics/NumPy; model review and tracking |

## Phase 5: local MLflow tracking and registry

Use the SQLite backend and local artifact directory already approved in
`configs/config.yaml`. The module resolves both paths against the repository root:

- Metadata/registry: `artifacts/mlflow/mlflow.db`.
- Model/evidence files: `artifacts/mlflow/artifacts/`.
- Dashboard: [local MLflow](http://127.0.0.1:5000), bound only to loopback with one worker.

Start the server in a local terminal when it is not already running:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.tracking.mlflow_utils serve
```

Leave that terminal open; Ctrl+C stops that foreground server. The server started
during Phase 5 runs in the background, with logs and its launcher PID under `reports/`.
It is not installed as an automatic Windows service. Tracking commands access the
local database directly and can run without the dashboard server. The HTTP API
and dashboard use the same database/artifacts. Preserve both storage locations;
the SQLite file alone does not contain model weights. For a manual backup, stop
writers/server and copy the entire `artifacts/mlflow/` directory.

Import or verify the completed baseline without retraining:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m src.tracking.mlflow_utils import
.\.venv\Scripts\python.exe -X utf8 -m src.tracking.mlflow_utils verify
# Select another completed run explicitly.
.\.venv\Scripts\python.exe -X utf8 -m src.tracking.mlflow_utils import --run-dir artifacts/training/experiment-02
```

The importer validates checkpoint/source hashes, DVC snapshot identities, complete
finite epoch history, validation metrics, and disjoint train/validation membership
before creating a record. It logs requested parameters, the effective device,
per-epoch loss/metric curves, best-checkpoint global/per-class validation metrics,
resolved YOLO arguments, exact memberships, source/configuration snapshots, and
the checkpoint. Original training dependencies stay in `source/requirements-lock.txt`;
the import environment and importer code are stored separately under `tracking/`.
No images, credentials, or final-test/monitoring results are copied into MLflow.

This is explicitly a **completed-run import**, not a claim of live MLflow logging
during Phase 4. MLflow run/metric timestamps describe the import. Epoch steps and
values come from the original `results.csv`; its cumulative time and the source
training start time are preserved as evidence. Failed or incomplete training
cannot be registered by this importer.

A SHA-256 of the evidence bundle identifies a serial retry. The importer reuses
its existing run and model version, including after interruption before writing
the local `mlflow.json` receipt. Run only one importer for a bundle at a time;
concurrent import locking is not implemented. Verification resolves the numeric
registry version, checks retrieved checkpoint/evidence bytes, and compares stored
metrics/history with the training files. Missing or altered evidence causes an error.

The registry stores the **native Ultralytics checkpoint** as `model/best.pt`, with
format metadata. This is not an MLflow PyFunc flavor or an MLflow serving endpoint.
The FastAPI deployment stage resolves a specific registry version, retrieves its
checkpoint, and load it with `ultralytics.YOLO`; no custom model-wrapper class is
needed. See the official [MLflow client API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html)
and [artifact-store architecture](https://mlflow.org/docs/latest/self-hosting/architecture/artifact-store/).

Verified baseline registration:

| Field | Value |
| --- | --- |
| Experiment / registered model | `hard-hat-detector` |
| Run | `473308c0c5df40d2b1095f6a5779441c` |
| Numeric version | `1` |
| Checkpoint URI | `models:/hard-hat-detector/1/best.pt` |
| Approval status | `experimental` |
| Stable aliases | Neither `stable` nor `previous_stable` exists |

Open the [baseline run](http://127.0.0.1:5000/#/experiments/1/runs/473308c0c5df40d2b1095f6a5779441c)
to inspect parameters, `validation/*` metrics, `epoch/*` curves, tags, and artifacts.
Dataset identity is in `dataset_dvc_md5`, with processed/split IDs and original
Git SHA/dirty status alongside it. The source snapshot captures uncommitted code.

The baseline's weak person detection remains a quality limitation. Registration
does not assign a stable alias, invent a second model, or provide a rollback target.
An explicitly reviewed model must later receive `stable`; before replacing it,
preserve the previously approved numeric version as `previous_stable`. Resolve
those explicit aliases through MLflow, never `latest` or `version - 1`. The
promotion/deployment/rollback procedure belongs to later phases.

Evidence is in `artifacts/training/baseline/mlflow.json` and
`reports/phase5-validation.json`. The real run was retrieved after server restart,
all recorded evidence/checkpoint hashes matched, and retrying preserved exactly
one run/version. A retrieved checkpoint loaded successfully for CPU prediction
on one validation image; this is a loading sanity check, not a new quality result.
All seventeen tests passed, including interrupted imports, stable-alias preservation
in a disposable registry, and rejection of corrupted artifacts. CUDA tensor/NMS
and both DVC storage-plugin imports still pass after installing MLflow.

The dependency-lock update triggered an actual DVC reproduction. Both split and
processed-output hashes remained identical, `dvc status --json` returned `{}`,
and all 10,000 raw files retained their recorded sizes and modification timestamps.
The baseline run and experimental version were also checked in the MLflow UI.

| Function | Inputs | Processing/output | Dependencies and next stage |
| --- | --- | --- | --- |
| `local_store` | Approved MLflow configuration, project root | Absolute local SQLite/artifact locations and client | PyYAML/MLflow; logging or dashboard |
| `read_completed_run` | Completed training directory | Validated evidence and bundle fingerprint | JSON/CSV/SHA-256; import gate |
| `log_completed_run` | Validated training evidence | Parameters, metrics, artifacts, experimental version, receipt; serial retry recovery | MLflow Tracking/Registry; model selection |
| `verify_registration` | Numeric model version and original evidence | Registry retrieval, hash and metric/history comparison | MLflow artifacts; later host-side deployment retrieval |
| `main` | `import`, `verify`, or `serve` command | Dispatches the corresponding lifecycle action | argparse/subprocess; local operator |

## Phase 6: local CI and runner setup

`.github/workflows/ci.yml` implements manual local CI and optional deployment:

```text
Manual dispatch on main -> runner/Python preflight -> checkout dispatched commit
  -> install locked dependencies -> compile sources -> isolated tests -> Docker build
  -> optional explicit deployment -> matching readiness/prediction probe -> summary
```

The user approved runner installation/registration on 2026-10-11. The official
Windows x64 runner 2.338.0 is checksum-verified and registered to this repository.
GitHub validation passed on 2026-10-12. Phase 7 adds real Docker build/deployment
steps. Dispatch `deployment=none` (default) to validate/build, `experimental` with
an explicit `model_version` to demonstrate an unapproved model, or `stable` to
resolve the reviewed alias. Stable deployment currently refuses because no such
alias exists. This workflow never changes registry aliases or approval tags.

The workflow accepts only manual `workflow_dispatch` runs from `main` in the
existing private repository. It has read-only repository permissions, a pinned
checkout action, no persisted checkout credentials, a 60-minute limit, and one
concurrency group. There are no push, pull-request, schedule, or training triggers.
The source commit is the one recorded by the dispatch, not a later moving branch
tip. Logs and a summary go to the private repository; data/model artifacts are
not uploaded. Tests create tiny temporary datasets and registries, so no Drive
OAuth, B2 keys, original dataset, or live MLflow server is needed.

### Approved local runner setup

A repository-scoped Windows x64 runner provides the execution path from GitHub
to this laptop. It receives jobs over outbound HTTPS and executes their commands
locally. Later it can reach Docker Desktop and the local MLflow registry without
opening inbound ports or adding a cloud server. See GitHub's
[runner reference](https://docs.github.com/en/actions/reference/runners/self-hosted-runners).

| Setting | Approved value |
| --- | --- |
| Repository | Existing private `varshithdoli/Reproducible-MLOps-Pipeline-for-Visual-Analytics-Data-Versioning-Monitoring-and-Automated-Rollback` only |
| Runner directory | `F:\visual-analytics-actions-runner` |
| Runner name | `visual-analytics-laptop` |
| Work directory | `_work` under the runner directory |
| Labels used by workflow | `self-hosted`, `Windows`, `X64` |
| Account/lifetime | Current Windows account, non-elevated; manually start/stop `run.cmd`; no Windows service or startup task |
| Python | Existing Python 3.11.9 launcher, new `venv` under the runner's temporary directory |
| Network | Outbound GitHub/package downloads; no inbound firewall changes |

The checkout action cleans its job directory. A pre-check rejects any workspace
outside `F:\visual-analytics-actions-runner\_work\` before checkout can execute.
Never point the runner at `F:\major project implementation`: that directory holds
the original dataset, working changes, environment, and real registry. The runner
will use its own clone and job-temporary environment; pip's local download cache may be reused.

This repository's long name produces a 238-character runner checkout path.
Windows long-path support is already enabled on this laptop; the workflow also
sets Git's `core.longpaths=true` through job-local environment variables. The
first GitHub attempt failed during Git initialization before this setting was
added. No machine-wide Git configuration or Windows registry setting was changed.
Python's virtual-environment executable also exceeded the Windows launch path
limit inside that checkout. CI therefore creates its environment under the shorter
`RUNNER_TEMP\venv` path; the runner cleans its temporary directory between jobs.
Dependency installation skips ahead-of-time bytecode compilation in this
temporary environment; the separate compile step still checks project sources.
The job summary uses Python's standard library for hashing and UTF-8 output,
so it does not depend on PowerShell module discovery in the runner session.
Installation gets at most three attempts, with a five-second delay, to tolerate
intermittent package-index TLS transport failures observed on this laptop. Each
failed attempt is logged; exhausted retries fail the job. TLS verification and
all package pins remain unchanged.

**Security implication:** repository workflow code runs with the Windows account's
permissions, including access to files, OAuth tokens, and Docker that the account
can access. A separate checkout is not a security sandbox. Read-only GitHub token
permissions do not limit local filesystem access. Only trusted people may write
workflow/code in this repository; review changes before manual dispatch and stop
the runner when unused. The workflow guards prevent accidental runs; someone who
can change the workflow can change those guards. GitHub discusses these risks in
its [secure-use guidance](https://docs.github.com/en/actions/reference/security/secure-use).

The setup procedure is:

1. In this repository, open **Settings -> Actions -> Runners -> New self-hosted
   runner**, select Windows/x64, and obtain the official download/checksum.
2. Download and verify the official runner into the proposed directory. Use a
   normal PowerShell terminal, not an elevated one.
3. Run its interactive `config.cmd` with this repository URL, the proposed name,
   and `_work` directory. Enter the one-hour registration token only in the local
   terminal. Decline Windows-service installation. Never paste the token in chat.
4. Start `run.cmd` manually and verify this repository lists the runner as idle.
5. Publish reviewed project sources/workflow to `main`, after checking Git
   exclusions and remote state. The initial publication was completed on
   2026-10-11; dataset bytes, model artifacts and credentials remain excluded.
6. Dispatch **Local validation** on `main`, verify all steps and commit/lock hashes
   in GitHub, then stop the runner with Ctrl+C when finished.

GitHub's [setup instructions](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/add-runners)
describe registration and token lifetime. Its current
[billing documentation](https://docs.github.com/en/billing/concepts/product-billing/github-actions)
lists self-hosted runner execution as free; this workflow uses no hosted runner,
Actions artifact upload, cloud storage transfer, or paid infrastructure. Laptop
power, disk space, and network usage still apply.

Registration used the existing Git Credential Manager sign-in to obtain a
short-lived repository registration token. It was passed to the official runner
through its supported process environment, not command arguments, source, or chat.
No Windows service was installed. For this validation the runner is started with
`--once`, so it exits after processing one job. Installation evidence is in
`reports/phase6-runner-installation.json`; runner credentials remain in the
separate runner directory and must never be committed or uploaded.

Initial local preparation checks used the existing pinned environment in a
separate source-only snapshot. Those checks alone did not prove GitHub execution.
The workflow passes actionlint and PowerShell parsing; workspace rejection,
native-command failure propagation, retry recovery/exhaustion, and summary output
were checked locally.

The real [successful GitHub run](https://github.com/varshithdoli/Reproducible-MLOps-Pipeline-for-Visual-Analytics-Data-Versioning-Monitoring-and-Automated-Rollback/actions/runs/38212488144)
validated commit `dac82c43d53ed9f08cb526a534f545082bda4468` on 2026-10-12.
It passed fresh dependency installation, `pip check`, source compilation,
**all 17 tests in 49.078 seconds**, and the final commit/lock summary. The job took
6 minutes 33 seconds and needed no installation retry in that successful run.
The one-job runner then exited; GitHub reports it offline and not busy. This is
expected for a manually operated runner, not a Windows service failure.

Evidence is in `reports/phase6-validation.json`, `reports/phase6-github-run.json`,
and `reports/phase6-github-final.log`. Earlier failed attempts and their fixes
remain documented. Tests use temporary fixtures without copying the raw dataset
or live registry. All 10,000 original files retain their recorded metadata and
DVC reports no changes. At the Phase 6 boundary, container build/deployment and
stable promotion remained separate gates. Phase 7 below completes the serving work.

Docker Desktop's Linux engine was unavailable during the Phase 6 check. This
does not prevent these CI tests; start Docker Desktop and verify the engine
before Phase 7 container work. Do not treat earlier Docker verification as proof
that the engine is currently running.

## Phase 7: local Docker and FastAPI serving

Start Docker Desktop before building. From the project directory in PowerShell:

```powershell
$docker = "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin\docker.exe"
& $docker build -t visual-analytics-api:phase7 .
.\.venv\Scripts\python.exe -m src.serving.deploy --version 1 --experimental --image visual-analytics-api:phase7
Invoke-RestMethod http://127.0.0.1:8000/health
```

`deploy.py` accesses the local SQLite registry directly, so the MLflow dashboard
need not run. It checks the finished run, registered source, bundle and checkpoint
hashes, then stages an immutable version/settings directory under
`artifacts/serving/`. Staging retries reuse matching bytes and refuse corruption.
The container receives only that directory, mounted read-only; it needs no Drive,
MLflow, dataset, or credential access. No weights or dataset enter the Docker build.

The deployed image is resolved to its exact Docker image ID. The container runs
as user 10001, with a read-only filesystem and writable temporary directory,
one Uvicorn worker, and two CPU threads. Only host loopback port 8000 is published.
`/health` becomes ready after SHA-256 verification, model/class validation and a
warm-up prediction. The host deployment also checks the returned model/settings
and sends a real synthetic-image prediction before reporting success. Readiness
means the service loaded; it is not a model-quality or overall-health decision.

Send **raw image bytes**, not JSON, a path, or multipart form data:

```powershell
curl.exe --fail-with-body -H "Content-Type: image/png" --data-binary "@dataset/images/hard_hat_workers101.png" http://127.0.0.1:8000/predict
& $docker logs visual-analytics-api
& $docker stop visual-analytics-api
```

The prediction response includes model name/version, checkpoint hash, experimental
status, oriented input dimensions, and `class_id`, `class_name`, `confidence`,
`xyxy` for each detection. Coordinates refer to the decoded, EXIF-oriented image.
The same OpenCV letterbox function used in preprocessing pads/downscales to 416
without upscaling; inverse transforms map boxes back to the input dimensions.
Confidence 0.25, IoU 0.70, max 300 detections, CPU FP32 and no inference augmentation
are recorded settings. These are starting inference defaults, not calibrated health
thresholds or accuracy claims. Requests accept static PNG/JPEG, at most 4 MiB encoded
and 4,194,304 decoded pixels. Invalid content returns 415/422, oversized images 413,
not-ready 503, and inference failures 500. An empty detection list is a valid result.
Phase 8 adds `/metrics` without changing the prediction contract.

`artifacts/serving/deployment.json` points to the last deployment attempt; per-container
receipts preserve its image ID, source revision/dirty state, model lineage, settings,
readiness and prediction result. Replacements briefly interrupt serving. This phase
does not automatically recover a failed deployment: the host reports failure and
leaves logs for inspection. Only a container labeled for this project can be replaced.
Do not run simultaneous manual deployments; CI already serializes its jobs.

Without `--experimental`, only numeric versions referenced by `stable` or
`previous_stable` may deploy; omitting the version resolves `stable` explicitly.
Neither alias currently exists. Do not assign either merely to make CI pass.
The same `deploy()` procedure is available for approved normal deployment and the
future rollback controller; automatic target selection/recovery belongs to Phase 11.

The approved self-hosted runner uses its isolated checkout for code/image builds
and `--store-project 'F:\major project implementation'` for this laptop's local
registry/staging directory. It never checks out over the original dataset or uses
the development virtual environment. Docker must already be running. Linux package
pins and the Python base-image digest are committed; Debian system packages come
from Bookworm repositories at build time, so byte-identical rebuilds are not claimed.
The final image ID identifies the actual deployment.

| File/function | Inputs and processing | Output / requirement / evidence |
| --- | --- | --- |
| `api.py`: `load_model`, `predict_image`, `predict` | Verified local checkpoint, frozen manifest, bounded image body; shared preprocessing and serialized CPU inference | Readiness/JSON detections; FR-12; API contract tests and real container requests |
| `deploy.py`: `stage_model`, `deploy` | Registry version/alias, local bundle, Docker image; verify hashes and deployment identity | Staged checkpoint and deployment receipt; FR-09/FR-11; real registry fixture and container checks |
| `Dockerfile`, `.dockerignore`, `requirements-serving.txt` | Allowlisted source plus Linux CPU dependencies | Non-root image without credentials/data; FR-11; build and pip consistency checks |
| `.github/workflows/ci.yml` | Explicit dispatch and repository commit | Tests/build/optional deployment; FR-10/AC-08; actual GitHub run evidence |

The app uses FastAPI's [lifespan loading](https://fastapi.tiangolo.com/advanced/events/),
Ultralytics' [prediction results](https://docs.ultralytics.com/modes/predict/), and
Docker's [build-context and base-image guidance](https://docs.docker.com/build/building/best-practices/).

Phase 7 verification passed on 2026-10-12 in
[GitHub run 38250748811](https://github.com/varshithdoli/Reproducible-MLOps-Pipeline-for-Visual-Analytics-Data-Versioning-Monitoring-and-Automated-Rollback/actions/runs/38250748811).
Commit `6cdbf6f2541becac00466e1507e263ffc5135db7` passed a fresh dependency install,
pip consistency, source compilation, all **22 tests in 84.781 seconds**, the real
Docker build, and deployment with matching readiness and a prediction probe.
The deployed receipt records a clean source checkout and image
`sha256:dadbbc1f347df05c25ade128314ab152fa75ec5351000158b781e28680582b90`.
The runner exited after the successful job; the local API remains running.

Separate local checks verified seven detections on validation image
`hard_hat_workers101`, input-size/type errors, concurrent predictions, restart
readiness, refused startup with corrupted weights, and refused default deployment
without a stable alias. All 46 Linux serving pins were checked inside the image.
Dataset metadata and DVC stages remain unchanged. Evidence is in
`reports/phase7-validation.json`, `reports/phase7-api-validation.json`, the GitHub
run receipt/log, and `artifacts/serving/deployment.json`. Model quality has not been
improved by serving it; stable promotion remains pending.

## Phase 8: Prometheus and Grafana operations

Build/deploy the current API through the existing procedure, then start monitoring:

```powershell
$docker = "$env:LOCALAPPDATA\Programs\DockerDesktop\resources\bin\docker.exe"
& $docker build -t visual-analytics-api:local .
.\.venv\Scripts\python.exe -m src.serving.deploy --version 1 --experimental
& $docker compose -f compose.monitoring.yaml up -d
```

Open [Grafana Operations](http://127.0.0.1:3000/d/visual-analytics-operations) or
[Prometheus](http://127.0.0.1:9090/query). The API metrics are at
[localhost:8000/metrics](http://127.0.0.1:8000/metrics). All published ports bind only
to 127.0.0.1. Grafana provides anonymous **Viewer** access for this local demonstration;
no initial admin user is created, sign-up/login UI are disabled, and dashboards and
datasources are provisioned from files. There are no passwords in the repository.
Anyone with access to this laptop can read/query these local operational metrics.
This configuration is not intended for public hosting. Edit dashboard JSON in Git,
not through Grafana. Do not enable anonymous Admin access.

`deploy.py` creates/reuses the project-labeled `visual-analytics` Docker network
and attaches each replacement API as `api`. Prometheus scrapes `api:8000/metrics`
every 10 seconds, with a 5-second timeout. Grafana queries `prometheus:9090` on
that network. Replacing the API keeps its scrape address; metrics carry model
version and container/deployment ID so different deployments are not aggregated
into one latency or error measurement. Network names owned by another project
are refused. The monitoring Compose file intentionally leaves model deployment
under the existing registry-verifying procedure.

| Metric | Meaning / consumer |
| --- | --- |
| `va_prediction_requests_total` | POST `/predict` requests labeled success, client_error or server_error; counts/rates and server-error fractions |
| `va_prediction_duration_seconds` | Classic histogram in seconds, including upload reading, decoding, inference queue and model execution up to response headers; approximate p95 |
| `va_model_ready` | Model loaded with successful warm-up; operational readiness, not quality or overall health |
| `va_model_info` | Model name/version, checkpoint SHA, inference-settings SHA, experimental/approved mode and deployment ID |
| Prometheus `up` | Whether the last scrape succeeded; API failure is visible even when the service cannot export its own gauge |

Health/metrics/other routes do not increment prediction traffic. Client input
errors and request cancellations are separated from server errors. The dashboard
uses `5xx / (success + 5xx)` and excludes client errors from latency too. Every
outcome is initialized at zero; no traffic produces an undefined error fraction
and p95, not a fabricated healthy measurement. Two-minute `rate()` windows handle
counter resets, but require enough scrape history. Version/deployment labels keep
replacement histories distinct. In-memory API counters reset on process restart.
Latency buckets are 0.025, 0.05, 0.1, 0.2, 0.5, 1, 2, 3, 5, 10, 30 and 60 seconds
plus infinity. This is a histogram estimate, not exact request-level p95 and not
a calibrated health threshold. Network delivery time is not measured.

The seven panels show scrape availability, readiness, loaded model identity,
prediction counts, request rate, server-error fraction, and approximate p95.
Readiness is masked when `up=0`, so a stale gauge cannot appear healthy during
an outage. No alert or rollback controller is enabled in this phase. There is
no confidence/accuracy proxy, drift computation, or use of final-test data.

Prometheus 3.15.0 and Grafana 13.2.3 images are pinned by digest. Named Docker
volumes preserve time series and Grafana state. Prometheus retains at most two
days / 1 GB of TSDB blocks; WAL/head storage and total container disk use may
exceed that block-retention limit. Grafana telemetry/update checks and automatic
plugin preinstallation are disabled. No cloud monitoring subscription is used.

```powershell
& $docker compose -f compose.monitoring.yaml logs --tail 30
& $docker compose -f compose.monitoring.yaml stop
# Later, with the API running:
& $docker compose -f compose.monitoring.yaml up -d
```

Stopping keeps the named volumes. Do not use `down -v` unless intentionally
discarding monitoring history. These containers have no automatic restart policy;
start Docker and the services after a laptop reboot. The API can be restarted
with `docker start visual-analytics-api` when its existing image/model are suitable.

Local Phase 8 checks passed: all 23 tests, exact counter deltas for ten real
validation-image predictions and two invalid images, all seven PromQL expressions
and Grafana datasource queries, read-only dashboard permissions, API outage and
recovery, and preserved historical data/dashboard after monitoring restart.
The 500 outcome is exercised by an isolated mocked-inference test, not by enabling
a production fault. No browser automation surface was available for a visual
screenshot review; the dashboard definition, permissions, and data queries were
verified through Grafana's API. Evidence: `reports/phase8-integration.json`.

Adding the client changed the Windows dependency lock, so both DVC stages were
actually rerun. Split and processed-output hashes stayed identical; only the lock
dependency hash changed. The original dataset remains unchanged. CI also validates
Compose and runs `promtool check config` before building.

[GitHub run 38518901297](https://github.com/varshithdoli/Reproducible-MLOps-Pipeline-for-Visual-Analytics-Data-Versioning-Monitoring-and-Automated-Rollback/actions/runs/38518901297)
passed on 2026-10-14 for commit `27ea7e337e41ab150db59a1e16de154e8deade1d`:
fresh dependencies, compilation, **23 tests in 104.347 seconds**, Compose/promtool
validation, image build and deployment with readiness/prediction checks. Prometheus
automatically collected the replacement container's deployment ID. The once-only
runner exited; API, Prometheus and Grafana remain running locally. The two table
panels also have explicit label-to-column transforms, verified after provisioning
through the dashboard definition and incoming dataframe labels. That dashboard-only
follow-up does not change the CI-tested API image. Full evidence and file lists:
`reports/phase8-validation.json`. Next is Phase 9, Evidently with PSI/KS drift evidence.

Implementation follows the official [Prometheus client metric types](https://prometheus.github.io/client_python/instrumenting/),
[histogram guidance](https://prometheus.github.io/client_python/instrumenting/histogram/),
and [Grafana provisioning](https://grafana.com/docs/grafana/latest/administration/provisioning/).

| Requirement | File / function | Validation |
| --- | --- | --- |
| FR-13: operational collection | `api.py`: `lifespan`, `measure_predictions`, `metrics`; `prometheus.yaml` | Outcome/latency/scrape-exclusion unit test; real counter deltas and target queries |
| FR-14: operational display | `compose.monitoring.yaml`, Grafana datasource/provider/dashboard files | Seven panel queries, provisioned read-only dashboard, restart persistence |
| Monitoring continuity | `deploy.py`: `deploy` attaches the stable Docker DNS name | Live scrape, stopped-API detection and restored readiness |

## Storage and credentials

The user requested Google Drive after B2 recovery hit a daily transaction cap.
Both destinations are configured in `.dvc/config`; Drive is the default following
successful full recovery verification. Google Drive is distinct from Google Cloud
Storage; no GCS bucket or AWS service is used.

Google client and user-token JSON are stored outside the repository under
`%LOCALAPPDATA%\visual-analytics-dvc\`. Transfer subprocesses receive OAuth
credentials through their environment. Never paste credentials into chat, commit
them, or put them in container images. See the Google Drive setup below.

The Drive migration targets USD 0 in additional spending, using existing account
capacity without buying storage or enabling paid API usage. This is project
policy, not an enforced provider billing cap. The account had ample available
capacity at preflight; this does not establish that its subscription is a free
plan. Changed versions and derived data consume additional space.
The earlier B2 budget was USD 0/month with a USD 1/month ceiling; no provider
billing settings were changed.

## Phase 2 commands and recovery evidence

From the project root, run local checks:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m src.data.validate --report reports/dataset-validation.json
.\.venv\Scripts\python.exe -m dvc status dataset.dvc
.\.venv\Scripts\python.exe -m src.data.verify_restore --local-cache
```

The validator fully decodes images, checks pairing, dimensions, class names and
finite/in-bounds boxes, reports exact pixel duplicates, and computes a SHA-256
manifest over relative paths and file contents. It fails on invalid input and
does not write inside the raw dataset. Empty XML annotations are counted as
background examples, not silently removed. Source provenance remains unverified.

DVC uses copy-based caching and `dvc add --no-relink dataset` to preserve raw
workspace files. `dataset.dvc` identifies the actual cached version. Commit its
pointer with code/configuration changes; do not run `dvc add` to hide unintended
raw changes. There is no second, invented dataset revision in this phase.

`--local-cache` restores from the existing local DVC cache into an isolated
workspace with its own empty cache. It compares the restored content fingerprint
against the validation report and writes `reports/local-restore.json`. This
proves local recovery only, not cloud access. Temporary restored copies are removed
after verification; the original dataset is never the cleanup target.

For Google Drive, run:

```powershell
# Upload and verify a new snapshot.
.\.venv\Scripts\python.exe scripts/gdrive_session.py push-verify
# After interrupted verification, resume without repeating the upload.
.\.venv\Scripts\python.exe scripts/gdrive_session.py verify
```

Recovery downloads persist under
`reports/gdrive-restore-cache/<dataset-md5>/<remote-config-hash>/`. This cache is
separate from the original local DVC cache and B2 downloads. Retries fetch missing
objects; the temporary checkout is removed after validation. Run one verification
at a time. A resumed success records `fresh_cache: false`, `isolated_cache: true`,
and `resumed: true`. The completed Drive report has these flags and `verified: true`.
All restored file contents must match before a report is written.

B2 remains available with its cloud objects and partial recovery cache preserved.
Its helper explicitly targets B2 even after the default changes. If B2 recovery
is deliberately revisited, use the replacement key through
`powershell.exe -NoProfile -STA -ExecutionPolicy Bypass -File scripts/b2_window.ps1`
or `.\.venv\Scripts\python.exe scripts/b2_session.py verify`. The exposed key was
revoked, as confirmed by the user. Never reuse it. B2 recovery remains unverified.
Its provider-side caps and alerts are unchanged.

Recovery verification requires `dataset.dvc` and its matching
`reports/dataset-validation.json`. No Kaggle redownload is involved. Bare DVC
commands do not automatically load the helper's OAuth token; the helper supplies
it through the subprocess environment.

| File/function | Inputs | Processing | Output | Dependencies / next stage |
| --- | --- | --- | --- | --- |
| `validate.py:read_annotation` | XML, actual dimensions, allowed classes | Validate pixel-box semantics | Class/box pairs | Standard library; preprocessing can reuse it |
| `validate.py:validate_dataset` | Raw directory and class names | Decode, validate, count, fingerprint | Validation report | Pillow; DVC and later preprocessing |
| `verify_restore.py:verify_restore` | DVC pointer/config and validation report | DVC pull into isolated cache (cloud retries resume), compare fingerprints | Local, Drive, or B2 recovery report | DVC, validator; reproducibility evidence |
| `b2_session.py:main` | Masked replacement credentials and action | Run real DVC commands in a temporary credential environment | DVC output and optional recovery evidence | Standard library, DVC; no custom storage API |
| `gdrive_session.py:main` | Locally stored OAuth client, browser sign-in, action | Supply OAuth credentials to real DVC commands | Upload and content-verified recovery report | PyDrive2, DVC; private Drive storage |

## Google Drive setup

Google Drive is supported by DVC and holds the current 1.32 GB snapshot within
this account's existing capacity. Google advertises up to 15 GB free shared with Gmail
and Photos; this is not unlimited storage. Changed dataset versions and derived
data also consume space. API throttling and OAuth expiration remain possible.
See [Google storage](https://support.google.com/drive/answer/2375123) and
[Drive API limits](https://developers.google.com/workspace/drive/api/guides/limits).

1. The supplied private Drive folder is configured as the default remote
   `gdrive`, at `gdrive://15s2R1cbsqHRl_34n0u5r1SdeivW2NAVl`, with `verify=true`.
   Keep its sharing Restricted. Authenticated folder-permission and capacity
   checks passed; the URL alone is not evidence of access.
2. In Google Cloud Console, create/select a project, enable **Google Drive API**,
   and create an **OAuth Desktop app** client for your own account. This is API
   identity setup, not a Google Cloud Storage bucket. Configure the OAuth consent
   screen and your account as a test user if the app is in Testing. Do not enable
   paid storage or billing to follow this procedure.
3. Download the client JSON outside the repository (for example, Downloads).
   Run `.\.venv\Scripts\python.exe scripts/gdrive_session.py auth` locally. A file
   picker selects that JSON; Google sign-in happens in your browser. DVC's shared
   OAuth app is currently affected by a blocking issue, so use your own client.
4. The script keeps the client and user token under
   `%LOCALAPPDATA%\visual-analytics-dvc\`, outside Git, and prints remaining account
   storage without account details. Transfer subprocesses receive credentials
   through their environment, never command arguments. Do not upload either JSON
   file or paste its contents into chat. DVC requests Drive access; sign in with
   the intended account and review Google's consent screen.
5. Run `.\.venv\Scripts\python.exe scripts/gdrive_session.py push-verify` once to
   upload and verify. After an interruption, use the same command with `verify`
   instead. Drive downloads persist separately under `reports/gdrive-restore-cache/`.
   `reports/gdrive-restore.json` is created only after all restored contents match.
6. Full Drive recovery passed on 2026-10-08. The default remote is now `gdrive`,
   `storage.provider` is `google_drive`, and this guide reflects the migration.
   B2 objects and both providers' recovery caches are preserved.

OAuth Testing refresh tokens expire after seven days for Drive scopes. A personal
app's publishing/verification settings must be resolved before unattended CI/CD;
sign-in may be needed again after token expiration or revocation. Consumer Drive
uses user OAuth: service accounts have no personal storage quota and are not the
solution for a free My Drive folder. See [DVC Drive setup](https://doc.dvc.org/user-guide/data-management/remote-storage/google-drive),
[Google OAuth expiration](https://developers.google.com/identity/protocols/oauth2),
and [service-account storage constraints](https://developers.google.com/workspace/drive/api/guides/about-shareddrives).

The intended deployment uses local model artifacts. Stage a verified candidate
and the previous stable model locally before deployment; serving and rollback
must not download the raw dataset. A storage outage can delay transfers/retraining,
but should not interrupt predictions from an already deployed local model. This
is a design requirement for later phases, not a claim that serving is implemented.

## Model tracking, serving, and recovery policy

Local MLflow will use a SQLite metadata file and a local artifact directory.
There is no separate database server. A run must link its code revision, DVC
dataset identity, split manifest, preprocessing/training parameters, evaluation
metrics, and artifact to its registered model version.

FastAPI will serve on localhost port 8000 with only health, prediction, and metrics
endpoints. Prometheus (9090), Grafana (3000), and MLflow (5000) also remain local.
Compose will describe these services when their phases are implemented.

Maintain explicit stable model references. Before candidate deployment, record the
approved rollback target and deployment settings; never infer it by subtracting
one from the current version. Resolve the target to a numeric registry version
and verify its artifact. Normal deployment and rollback must call the same
deployment procedure. The host-side health process must remain usable when the
prediction service fails. If recovery fails, report failure instead of repeatedly
switching models.

## Monitoring and initial health policies

Prometheus/Grafana cover request count, latency, server errors, and service health.
Evidently PSI/KS cover image brightness, image sharpness, and per-image maximum
detection confidence. Confidence comparisons exclude images with no predictions;
empty-prediction rate is reported separately. Bind prediction baselines to model
version and inference settings. Confidence change is not measured accuracy loss.

All thresholds in `configs/config.yaml` are approved **initial calibration
policies**, not scientifically validated constants. Configuration alone does not
implement these rules:

| Signal | Intended decision |
| --- | --- |
| PSI or KS drift evidence alone | Record evidence and continue if overall health is acceptable; never rollback solely for drift. |
| Persistent readiness failure | Unhealthy after startup grace and the configured consecutive failures. |
| Server-error rate | Unhealthy at the configured rate in two non-overlapping qualifying windows; client validation errors are excluded. |
| p95 latency | Unhealthy above the larger of the absolute floor and baseline multiplier, in two qualifying windows. |
| Labeled audit mAP50-95 | Unhealthy for the configured absolute drop from stable V1 on the same sufficiently large labeled batch. |
| Drift plus moderate server errors | Unhealthy when both evidence sources meet their configured criteria. |
| Missing/stale/insufficient evidence | Record it as unavailable; never invent a healthy measurement. |

KS uses the family alpha divided by three planned continuous comparisons, plus a
minimum statistic. PSI uses reference-derived bins with documented smoothing.
Exact feature definitions, bin edges, and signal freshness rules must be fixed
and recorded during the monitoring phase, before demonstration calibration.

Ordinary requests have no guaranteed ground truth. Offline labeled audits of the
monitoring pool provide separately reported quality measurements. Neither input
drift nor prediction confidence is a substitute for labeled evaluation.

The demonstration fault is off by default. When explicitly enabled for a candidate
deployment, every tenth prediction request will return a controlled server error.
These are real measured service failures, not evidence that model weights degraded.
Rollback restores the recorded healthy deployment settings as well as the stable
model, then checks readiness and successful predictions before claiming recovery.

## CI/CD and incremental delivery

Use the existing private GitHub repository and `main`. Actions is reported enabled.
GitHub Actions cannot deploy to this laptop without an execution mechanism. A
repository-scoped local runner was approved and registered on 2026-10-11.
GitHub validation has passed. Phase 7 extends the workflow to build the real CPU
image and optionally deploy a selected local registry model. Local serving checks
and the expanded GitHub run passed; see the Phase 7 evidence above.

Proceed through data/DVC, preprocessing, training, MLflow, CI/CD, serving,
operational monitoring, drift, health, rollback, and final validation incrementally.
CI/CD may be designed before serving, but successful build/deployment evidence
requires the actual serving implementation. After each phase, report files,
behavior, reasons, integration, checks, results, and the next phase. Stop and
report failures. Do not claim acceptance criteria from configuration alone.

| PRD requirements | Current evidence | Implementation status |
| --- | --- | --- |
| FR-01 | Supplied dataset, `validate_dataset`, inspection report | Exact Kaggle provenance unverified; no download performed |
| FR-02 / FR-04 | `dataset.dvc`, `verify_restore`, content comparison | Local tracking and full local/Drive recovery verified |
| FR-03 / amended AC-02 | `.dvc/config`, `gdrive_session.py`, `reports/gdrive-restore.json` | Real Drive upload and content-verified recovery passed |
| FR-05 / AC-03 | `split.py`, `preprocess.py`, `test_preprocessing.py`, full-dataset audit | Training-ready images/labels verified; DVC lineage recorded |
| FR-06 / FR-07 | CUDA smoke run, 30-epoch baseline, best-checkpoint metrics, source/data/checkpoint lineage | Training/evaluation complete; weak person class prevents treating this baseline as stable |
| FR-08 / tracking acceptance | Local SQLite run, parameters, epoch/global/per-class metrics, artifacts, DVC/source identities | Verified |
| FR-09 / registry acceptance | `hard-hat-detector` version 1, verified artifact retrieval | Experimental version verified; stable selection pending quality review |
| FR-10 / AC-08 | GitHub run 38560936350: 29 tests, monitoring config validation, Docker build, registry deployment, readiness and prediction | Verified for explicit experimental deployment; stable promotion is separate |
| FR-11 / FR-12 | `Dockerfile`, `api.py`, `deploy.py`, serving tests and real validation-image prediction | Local experimental serving verified |
| FR-13 / FR-14 | API metrics, real Prometheus scrapes, Grafana panel queries, outage/restart checks | Local operational monitoring verified |
| FR-15 / FR-16 | Real 250/250 API replay, Evidently HTML/JSON, PSI/KS summary and exclusion/freshness tests | Local drift evidence verified; continuous arbitrary-traffic capture is not implemented |
| FR-17 / FR-18 | Decision matrix, exact traffic windows, live readiness outage/recovery and drift-only exclusion | Implemented; real comparative labeled quality remains unavailable until a stable baseline is approved |
| FR-19 to FR-21 | Recorded approved predecessor, shared deployment procedure, recovery probes and fresh Prometheus samples | Implemented; a live switch to an approved stable model remains unverified because none exists |

`tests/test_data_validation.py` checks the validation contract using valid and
invalid PNG/XML fixtures. `tests/test_restore.py` tests fresh-cache recovery with
a tiny real DVC dataset, interrupted-download recovery, provider cache isolation,
explicit B2 routing, and credential handling. Synthetic
filesystem remotes exercise recovery behavior; they do not prove cloud access.
Full-dataset local and cloud recovery reports provide separate evidence for
FR-04 and amended AC-02. Their content comparisons are not mocked.
`tests/test_preprocessing.py` adds grouping/isolation, coordinate and pixel checks,
raw protection, stale-manifest rejection, and byte-identical repeated output.
`tests/test_training.py` checks isolated smoke inputs and metric serialization;
its mocked evaluation test does not establish model quality. `tests/test_tracking.py`
uses real temporary SQLite databases and artifact stores to test imports, retries,
corruption detection, and alias preservation. Its tiny checkpoint bytes are fixtures,
not trained models. The actual registered baseline has separate retrieval/inference
evidence in `reports/phase5-validation.json`. The serving tests add request/status,
coordinate, corruption and approval checks; the tracking tests also exercise
staging from a real temporary registry. Real container predictions are recorded
separately in `reports/phase7-api-validation.json`.

## Phase 9: local Evidently drift reports

`src/monitoring/drift.py` compares the three approved input/output features with
PSI and KS. It records sample availability, empty-prediction rates, immutable
baseline identities, exact PSI edges, thresholds, and observation timestamps.
The API now returns an inference-settings fingerprint with each prediction so
collection cannot silently mix settings. The fixed measurement and freshness
contract is in [the decision record](docs/decisions.md#phase-9-fixed-drift-measurement-contract).
These reports supply evidence for Phase 10; they contain no health or rollback action.

With the local API running, create new, uniquely named outputs:

```powershell
.\.venv\Scripts\python.exe -m src.monitoring.drift collect --pool reference --output artifacts/monitoring/demo/reference.json
.\.venv\Scripts\python.exe -m src.monitoring.drift collect --pool current --output artifacts/monitoring/demo/current.json
.\.venv\Scripts\python.exe -m src.monitoring.drift report --reference artifacts/monitoring/demo/reference.json --current artifacts/monitoring/demo/current.json --output reports/drift-demo
```

Collection sends only the reserved monitoring images to localhost and preserves
raw data. These commands replay a controlled batch, not continuous capture of
arbitrary production requests. A failed request aborts collection. Existing files
are never overwritten. Reuse a reference only with the same model, settings,
data/split and feature identities. Current observations expire one hour after
collection starts; a saved report is historical evidence after that deadline.
Do not repeatedly replay the same pool and call those independent samples.

Open `reports/drift-demo/evidently.html` for eligible feature charts and
`summary.json` for the complete availability/identity/statistics record. Evidently
also exports JSON. Empty predictions are excluded from confidence distributions
and reported separately. Missing, stale, overlapping, or incompatible populations
do not become a healthy measurement. Confidence changes do not measure accuracy.
The initial thresholds remain calibration policies, not established guarantees.

Files for this phase: `src/monitoring/drift.py`, `tests/test_drift.py`, the prediction
fingerprint in `src/serving/api.py` and its serving test, drift parameters in
`configs/config.yaml`, dependency manifests, reproduced `dvc.lock`, and these docs.
No monitoring UI service or new cloud dependency is deployed.

Phase 9 local execution on 2026-10-14 produced 250 reference and 250 current
observations from experimental version 1. Brightness/sharpness/confidence PSI
values were 0.048254 / 0.138301 / 0.078897; KS statistics were 0.080000 /
0.072000 / 0.080691 and p-values 0.401123 / 0.536945 / 0.368641. No configured
PSI/KS threshold was crossed. Reference confidence used 249 images (one empty
prediction); current confidence used all 250. These are historical controlled
batch results, not evidence of current service health or accuracy.

All 29 local tests passed, including six drift tests. Tests cover unchanged and
shifted distributions, constant reference tails, empty predictions, insufficient
samples, stale/overlapping/incompatible evidence, corrupt values, changed serving
settings, and real Evidently export with matching scores. Actual batch artifacts
are in `artifacts/monitoring/phase9/`; HTML, JSON and the complete summary are in
`reports/phase9-drift/`. HTML file generation and JSON contents were verified;
visual browser review has not been performed.

Phase 9 validation is complete. [GitHub run 38560936350](https://github.com/varshithdoli/Reproducible-MLOps-Pipeline-for-Visual-Analytics-Data-Versioning-Monitoring-and-Automated-Rollback/actions/runs/38560936350)
validated implementation commit `ee002e52350485d6a3a157967dde7dde1891dfd7`:
29 tests, dependency consistency, source compilation, monitoring configuration,
Docker build and experimental version-1 deployment with readiness/prediction checks.
Both DVC stages reproduced identical output hashes after the lock change; raw
content and modification times were preserved. Detailed evidence and the file
inventory are in `reports/phase9-validation.json`. The following documentation-only
commit records those results. Next is Phase 10: combine service, operational,
labeled-quality and drift evidence in the separate health decision layer.

## Phase 10: combined health decision

The host-side checker combines readiness, traffic errors, latency, labeled
mAP50-95 audit evidence and fresh drift reports. It records a decision and reasons;
rollback execution belongs to Phase 11. Run a bounded check with new output paths:

```powershell
.\.venv\Scripts\python.exe -m src.monitoring.audit --output artifacts/monitoring/audit-demo
.\.venv\Scripts\python.exe -m src.monitoring.health latency-baseline --output artifacts/monitoring/latency-demo.json
.\.venv\Scripts\python.exe -m src.monitoring.health check --drift reports/phase10-drift/summary.json --audit artifacts/monitoring/audit-demo/audit.json --latency-baseline artifacts/monitoring/latency-demo.json --output reports/health-demo.json
```

Latency calibration needs two completed two-minute windows with >=100 successful
predictions each, zero server errors, and p95 at most two seconds. Ordinary idle
traffic cannot supply that baseline. Client errors are excluded. A check takes
about 20 seconds for three readiness observations and reads the last two completed
traffic windows. It can report a real readiness failure even when prediction is
unavailable. Existing output files are preserved.

`healthy` requires complete fresh evidence with no breach. `unhealthy` requires a
confirmed configured breach. `unknown` identifies missing, stale, incompatible,
insufficient, or pending evidence. Drift alone cannot produce unhealthy. The model
currently lacks an approved stable reference, so the real labeled audit returns
unavailable and an otherwise normal deployment remains unknown for overall health.
The final-test pool is excluded. Full rules, timestamps, restart handling and
measurement definitions are in [the decision record](docs/decisions.md#phase-10-host-side-health-decisions).

Files: `src/monitoring/health.py` collects and evaluates evidence;
`src/monitoring/audit.py` prepares identical isolated labeled inputs for verified
checkpoints; `tests/test_health.py` covers decisions and measurement boundaries;
`configs/config.yaml` fixes window/freshness parameters. No package changes were
needed. Output includes the evidence used, deployment identity, reasons and a short
`valid_until`; a saved healthy report must not be reused after that deadline.

Phase 10 local validation passed 39 tests in 48.236 seconds. A real integration
used two qualifying windows with 110 and 250 successful predictions; three client
errors were excluded from the first denominator. The measured baseline p95 was
0.089151 seconds, making the initial latency limit two seconds. A new current-pool
replay produced fresh drift evidence against the preserved Phase 9 reference.
With those measured signals, the normal decision was unknown solely because no
approved stable quality baseline exists. Stopping the API produced three real
readiness failures and an unhealthy decision while the host checker kept running.
Restarting the same container restored readiness and invalidated old traffic
windows. No model switch or alias update occurred. The standalone CLI also passed
and correctly reported unknown with nonqualifying post-restart windows.

`reports/phase10-live/` preserves normal/outage/recovery evidence;
`reports/phase10-validation.json` records the phase checks and file inventory.
`reports/phase10-drift/` contains the distribution report collected for that test;
saved reports are historical evidence after their freshness deadlines. The full decision
matrix and available audit path use controlled fixtures; they do not establish a
real stable-versus-candidate quality comparison. No approved stable model was
invented. DVC status is unchanged, and all 10,000 original file sizes/modification
times match the preserved dataset metadata.

GitHub [run 38692757040](https://github.com/varshithdoli/Reproducible-MLOps-Pipeline-for-Visual-Analytics-Data-Versioning-Monitoring-and-Automated-Rollback/actions/runs/38692757040)
passed on October 15, 2026 for implementation commit
`9f641efec87ddfe50e3d495c91e9d77632b18f68`: locked dependencies, source compilation,
all 39 tests, monitoring configuration validation and the CPU Docker image build.
Deployment was `none`, so the existing experimental serving container was retained.
The first attempt ended during dependency installation after runner logs stopped
abruptly; no test failure was recorded. The same commit passed on retry without a
source change. The phase report preserves both attempts.

The API passed a real prediction after restart, Prometheus reported the API up,
and Grafana's database health check passed. Registry aliases remain empty.
Next is Phase 11: rollback controls. Automatic rollback is not implemented by
Phase 10, and a real quality comparison still requires an approved stable baseline.

## Phase 11: bounded rollback controls

`src/monitoring/rollback.py` performs one host-side health/recovery cycle. With a
valid recorded target and fresh unhealthy evidence, it automatically calls the
same `deploy()` procedure used by normal deployment. It does not assign registry
aliases, promote a model, train, or install a scheduler. Run it explicitly:

```powershell
.\.venv\Scripts\python.exe -m src.monitoring.rollback --output reports/rollback-check.json
```

An existing Phase 10 report can be supplied with `--decision`; it must still be
fresh when consumed. Without that argument the controller collects current
readiness and traffic evidence. Optional `--drift`, `--audit`, and
`--latency-baseline` paths supply the other signals. Use a new output path for
each call. The current project returns `unavailable` because no approved target
was recorded for its experimental deployment; it leaves the API unchanged.

Before normal deployment replaces an approved, ready predecessor, it verifies
that predecessor against MLflow and records its numeric version, complete model
manifest, immutable Docker image ID and loopback port. The target is written in a
deployment intent before removal and attached to the candidate receipt. An
experimental predecessor never becomes a rollback target. Stable promotion still
requires separate review; before moving `stable`, preserve its numeric version as
`previous_stable`. This phase never creates either alias automatically.

At rollback, the recorded version must still be referenced by `stable` or
`previous_stable`, and its artifact hashes must verify. Recovery restores the
recorded inference settings and image, even if candidate configuration changed.
The controller re-evaluates health evidence, checks model/container/process
identity and expiry, then records a durable attempt before replacement. Drift
alone, unknown health, stale evidence, missing targets and concurrent operations
cannot initiate a switch. Normal deployment and rollback use the same local lock.

Recovery requires three matching readiness responses spaced ten seconds apart,
20 successful predictions on distinct existing validation images, and fresh
Prometheus samples for the new container/model/settings identity. The final-test
pool is excluded. `recovered` means operational recovery; it does not certify
model quality. Failed or interrupted attempts remain recorded and are never
automatically retried. The restored deployment receives no automatic target for
another switch, preventing a loop between models.

Files: `src/monitoring/rollback.py` contains the controller and recovery probes;
`src/serving/deploy.py` adds predecessor capture, recorded-settings restoration,
the shared lock, and atomic receipt publication; `src/monitoring/health.py` reads
the deployed port from its receipt. `tests/test_rollback.py` exercises the decision,
switch and recovery boundaries; `tests/test_tracking.py` verifies restoration and
revoked approval against a disposable real MLflow registry. No dependencies,
health thresholds, dataset files or serving API code were changed.

Phase 11 local validation passed 53 tests in 36.690 seconds. Live checks exercised
three readiness probes, twenty validation-image predictions and fresh Prometheus
samples against the existing experimental service without switching it. The real
controller refused rollback because no approved target exists. All 10,000 original
file sizes/modification times and DVC status are unchanged; registry aliases remain
empty. Evidence is in `reports/phase11-validation.json`,
`reports/phase11-live-recovery-probes.json`, and `reports/phase11-no-approved-target.json`.
GitHub [run 38696131541](https://github.com/varshithdoli/Reproducible-MLOps-Pipeline-for-Visual-Analytics-Data-Versioning-Monitoring-and-Automated-Rollback/actions/runs/38696131541)
passed on October 15, 2026 for implementation commit
`ede7e883a01832567714ea29a74296cea89f4809`: all 53 tests passed in 169.897 seconds,
with locked dependency consistency, source compilation, monitoring configuration
validation and the CPU Docker image build also successful. Deployment was `none`.
A real approved-model rollback cannot be demonstrated with the current registry.
Next is Phase 12: end-to-end validation, with that acceptance gap kept explicit.
