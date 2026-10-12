# Reproducible MLOps Pipeline for Visual Analytics

This college major project connects dataset versioning, object detection, model
tracking, local serving, monitoring, and recovery. Its central rule is that drift
is evidence for a health decision: drift alone never triggers rollback.

**Current status: Phase 6 runner registered; first GitHub validation pending.**
The dataset is split into 3,500 train, 500 validation, 500 final test, and two
250-image monitoring pools. OpenCV produced 5,000 padded 416x416 PNGs and matching
YOLO labels. Independent checks verified every image and all 25,502 boxes, with
raw bytes/modification times unchanged and no candidate group crossing splits.
Evidence: `reports/phase3-validation.json`; the current suite has seventeen passing tests.
Google Drive remains the default DVC remote, with full raw-dataset recovery
verified in `reports/gdrive-restore.json`. Derived outputs are cached locally.
B2 is retained as a legacy remote with incomplete recovery evidence.
The CUDA smoke run and 30-epoch baseline completed with finite losses and metrics.
Best-checkpoint validation mAP50 is 61.94%, but person detection is weak
(1.94% per-class mAP50). This is an experimental baseline, not an approved stable
model. MLflow now records this baseline as `hard-hat-detector` version 1.
Local tracking is verified. Serving, deployment, monitoring, and rollback remain future work. Neither `stable`
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

`requirements.txt` lists direct dependencies through Phase 5: PyYAML, Pillow, DVC,
its S3 and Google Drive integrations, NumPy, OpenCV, PyTorch, torchvision, and
Ultralytics, and MLflow. `requirements-lock.txt` pins their resolved Windows/Python
3.11 dependencies. It is not a lock for the future Linux serving container.
MLflow is pinned to 3.16.1. Its server dependencies include FastAPI; that does not
implement the project's future prediction service. Monitoring packages are not installed yet.

Docker Desktop 4.94.0, Engine 29.8.2, Compose 5.5.1, and WSL 2 were detected during
preparation. The Linux engine is reachable, and the user verified `hello-world`.
A serving-image build remains a future validation task.

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
The later FastAPI stage can resolve a specific registry version, download its
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

`.github/workflows/ci.yml` prepares the real validation part of CI/CD:

```text
Manual dispatch on main -> runner/Python preflight -> checkout dispatched commit
  -> install locked dependencies -> compile sources -> isolated tests -> summary
```

The user approved runner installation/registration on 2026-10-11. The official
Windows x64 runner 2.338.0 is checksum-verified and registered to this repository;
the first GitHub validation is pending. Docker build/deployment will extend this workflow
after Phase 7 supplies the serving application, Dockerfile, and health checks.
Until then, FR-10 / AC-08 are not complete; a source compilation check is not a
container build. Stable model approval also remains pending.

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
5. Publish the reviewed project sources/workflow to `main`, after checking Git
   exclusions and remote state. The read-only remote query on 2026-10-09 advertised
   no HEAD/main/master refs; no publication has been performed in this phase.
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

Local preparation checks use the existing pinned environment in a separate
source-only snapshot. They do not establish a clean dependency installation,
runner registration, GitHub workflow execution, Docker build, or deployment.
Evidence is in `reports/phase6-validation.json` and
`reports/phase6-local-validation.log`. The workflow passes actionlint and native
PowerShell parsing; the workspace rejection and native-command failure path were
exercised. The existing seventeen tests run without copying the raw dataset or
the real registry into that snapshot.

Docker Desktop's Linux engine was unavailable during the Phase 6 check. This
does not prevent these CI tests; start Docker Desktop and verify the engine
before Phase 7 container work. Do not treat earlier Docker verification as proof
that the engine is currently running.

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
GitHub execution and Docker build/deployment remain unverified until their actual
checks complete.

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
| FR-10 / AC-08 | Manual local-runner validation workflow and registered runner | First GitHub execution pending; build/deployment await serving |
| FR-11 / FR-12 | Approved Docker/FastAPI local serving design | Pending Phase 7 |
| FR-13 to FR-18 | Approved monitoring and calibration policies | Pending |
| FR-19 to FR-21 | Explicit stable-target and recovery policy | Pending |

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
evidence in `reports/phase5-validation.json`. All seventeen tests passed.
