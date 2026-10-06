# Reproducible MLOps Pipeline for Visual Analytics

This college major project connects dataset versioning, object detection, model
tracking, local serving, monitoring, and recovery. Its central rule is that drift
is evidence for a health decision: drift alone never triggers rollback.

**Current status: Phase 2 dataset validation and DVC setup.** Full raw-data validation
and local recovery into an isolated workspace/cache passed. B2 transfer requires the replacement credentials entered
through the local prompt. Data conversion, training, serving, monitoring, and
rollback are not implemented. Infrastructure availability is not pipeline completion.

## Approved lifecycle

```text
Existing Kaggle-attributed dataset (exact provenance unverified)
  -> DVC + Backblaze B2
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

Backblaze B2 deliberately replaces GCS in the original PRD. FR-03 and AC-02 will
be demonstrated using B2. The original PRD remains unchanged; see
[the decision record](docs/decisions.md).

## Repository responsibilities

| Path | Purpose and pipeline position |
| --- | --- |
| `dataset/` | Original image/XML inputs, preserved unchanged; bytes tracked by DVC rather than Git. |
| `dataset.dvc` | DVC dataset identity; Git stores this pointer instead of raw data. |
| `.dvc/config` | Authoritative B2 remote configuration; no credentials. |
| `src/data/validate.py` | Read-only image/annotation validation and content fingerprint. |
| `src/data/verify_restore.py` | Fresh-workspace/cache recovery and content comparison. |
| `scripts/b2_session.py` | Masked credential entry for real DVC commands; secrets remain in process memory. |
| `tests/test_data_validation.py` | Real PNG/XML fixtures for invalid-data and fingerprint behavior. |
| `tests/test_restore.py` | Real DVC integration regression for recovery below a Git-ignored directory. |
| `configs/config.yaml` | Non-secret service/storage settings and approved initial health policies. |
| `params.yaml` | Dataset split and training parameters that will participate in DVC lineage. |
| `requirements.txt` | Exact direct dependencies needed by the current implemented phase. |
| `docs/decisions.md` | Approved choices, provenance limitations, credential policy, and remaining gates. |
| `.gitignore` | Keeps raw/derived data, environments, credentials, and model binaries out of Git. |
| `Major_Project_PRD_v2_with_Literature_Gap_Analysis.docx` | Original requirements and acceptance criteria. |

There are no empty stage modules. Source files will be added when their pipeline
stage is implemented. Paths such as `data/processed` and `artifacts/mlflow` in
configuration describe future outputs; those outputs do not exist yet. Validation
and recovery evidence is written to the ignored `reports/` directory.

For viva explanation, configuration files have no executable processing:

| File | Inputs | Processing/output | Dependencies and consumer |
| --- | --- | --- | --- |
| `configs/config.yaml` | Approved service, storage, class, and policy decisions | Declarative settings; no pipeline action | PyYAML; future data, serving, monitoring, health, and rollback code |
| `params.yaml` | Approved split/training choices | Declarative experiment parameters | PyYAML; future DVC preprocessing/training stages |
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

`requirements.txt` lists direct dependencies through Phase 2: PyYAML, Pillow, DVC,
and its S3 integration. `requirements-lock.txt` pins their resolved Windows/Python
3.11 dependencies. It is not a lock for the future Linux serving container.
No training or monitoring packages are installed yet.

Docker Desktop 4.94.0, Engine 29.8.2, Compose 5.5.1, and WSL 2 were detected during
preparation. The Linux engine is reachable, and the user verified `hello-world`.
A serving-image build remains a future validation task.

The global PyTorch installation is `2.2.2+cpu`: it cannot train on the NVIDIA GPU.
The project environment will receive a tested CUDA-enabled build in the training
phase. GPU tensor operations, torchvision NMS, and a short YOLO training run must
pass before GPU training is claimed. Initial settings are 416 pixels, batch 4,
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
will produce YOLO labels and OpenCV-prepared images in a separate directory.
Preserve detail by padding these near-square images to the configured input size,
with matching box transforms. Do not overwrite, rename, or delete raw files.

The initial split is 3,500 training, 500 validation, 500 monitoring, and 500 final
test images, using seed 42. Scene/grouping review precedes final manifests; related
scenes must stay together where identifiable. Preserve class representation and
document any count changes needed for grouping. The monitoring pool is divided
into disjoint 250-image reference/current pools. Keep the final test set out of
model selection, threshold tuning, and demonstration calibration.

## Storage and credentials

DVC will use `s3://varshith-visual-analytics-dvc/visual-analytics/dvc` with the B2
endpoint in `.dvc/config`. S3 is the access protocol; no AWS service or
GCS resource is used. The bucket is reported private with encryption enabled.
The DVC remote is configured. Authenticated transfer is verified only when the
local B2 procedure below succeeds; configuration alone does not prove access.

Use only a replacement, bucket-restricted B2 application key. The previous exposed
key is ineligible for reuse. Its revocation and replacement were user-confirmed. Never paste
credentials into chat, commit them, or put them in Docker images.

During Phase 2, provide the replacement key to the DVC process through
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, or an explicitly documented local
credential profile outside the repository. A `.env` file is not automatically
loaded by DVC. Do not enter literal secrets in shell commands that retain history.

The storage planning target is USD 0/month, with a USD 1/month ceiling. These are
project policies, not remotely enforced Backblaze billing caps. Account caps and
alerts must be configured separately. Estimate storage/egress before transfers;
do not begin work expected to exceed the ceiling. Phase 1 performs no transfers.

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
proves local recovery only, not B2 access. Temporary restored copies are removed
after verification; the original dataset is never the cleanup target.

For B2, run this yourself in a local terminal (never enter keys in chat):

```powershell
.\.venv\Scripts\python.exe scripts/b2_session.py push-verify
```

It prompts for the replacement key ID and application key with input hidden, passes
them only to child-process environments, executes `dvc push`, then pulls into a
fresh workspace/cache and validates every restored file. Credentials are not saved
to `.env`, DVC config, or command history. Noninteractive input is rejected. Expect
about 1.32 GB upload and 1.32 GB download for the first round trip. It writes
`reports/b2-restore.json` only after successful verification. Failure exits nonzero;
never claim cloud recovery from a local-cache report.

The same prompt supports `status` (remote comparison), `push`, or `verify` when
only one operation needs repeating. Restore requires `dataset.dvc` and the latest
matching `reports/dataset-validation.json`. On a fresh checkout, first obtain the
raw files with `dvc pull`, then generate the validation report before the separate
fresh-cache demonstration. No Kaggle redownload is involved.

| File/function | Inputs | Processing | Output | Dependencies / next stage |
| --- | --- | --- | --- | --- |
| `validate.py:read_annotation` | XML, actual dimensions, allowed classes | Validate pixel-box semantics | Class/box pairs | Standard library; preprocessing can reuse it |
| `validate.py:validate_dataset` | Raw directory and class names | Decode, validate, count, fingerprint | Validation report | Pillow; DVC and later preprocessing |
| `verify_restore.py:verify_restore` | DVC pointer/config and validation report | DVC pull into fresh cache, compare fingerprints | Local or B2 recovery report | DVC, validator; reproducibility evidence |
| `b2_session.py:main` | Masked replacement credentials and action | Run real DVC commands in a temporary credential environment | DVC output and optional recovery evidence | Standard library, DVC; no custom storage API |

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
repository-scoped local runner remains subject to separate setup/security approval;
none is installed or registered by this project. No workflow or fake deployment
has been created in Phase 1.

Proceed through data/DVC, preprocessing, training, MLflow, CI/CD, serving,
operational monitoring, drift, health, rollback, and final validation incrementally.
CI/CD may be designed before serving, but successful build/deployment evidence
requires the actual serving implementation. After each phase, report files,
behavior, reasons, integration, checks, results, and the next phase. Stop and
report failures. Do not claim acceptance criteria from configuration alone.

| PRD requirements | Current evidence | Implementation status |
| --- | --- | --- |
| FR-01 | Supplied dataset, `validate_dataset`, inspection report | Exact Kaggle provenance unverified; no download performed |
| FR-02 / FR-04 | `dataset.dvc`, `verify_restore`, content comparison | Local tracking/recovery; B2 evidence pending local credential entry |
| FR-03 / amended AC-02 | `.dvc/config`, `b2_session.py`, `verify_restore` | Real B2 push/pull procedure; completion requires a successful B2 report |
| FR-05 to FR-07 | Approved preprocessing/training configuration | Pending |
| FR-08 to FR-12 | Approved local tracking/serving design | Pending |
| FR-13 to FR-18 | Approved monitoring and calibration policies | Pending |
| FR-19 to FR-21 | Explicit stable-target and recovery policy | Pending |

`tests/test_data_validation.py` checks the validation contract using valid and
invalid PNG/XML fixtures. `tests/test_restore.py` tests fresh-cache recovery with
a tiny real DVC dataset and checks that local recovery never creates B2 evidence.
The local restore command is the full-dataset integration check for
FR-04; the B2 round-trip command is the cloud integration check for amended AC-02.
Neither is replaced by mocked storage results.
