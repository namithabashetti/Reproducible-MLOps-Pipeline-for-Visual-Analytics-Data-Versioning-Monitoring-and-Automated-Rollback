# Approved project decisions

These records describe the user's approved implementation choices. They do not
claim that the pipeline stages have been implemented or accepted.

## Storage amendment to the PRD

**Original PRD storage:** DVC + Google Cloud Storage.

**Current approved storage:** DVC + Google Drive (verified 2026-10-08).

**Earlier approved storage:** DVC + Backblaze B2.

**Reason:** The user requested Google Drive after B2 daily transaction caps
interrupted recovery. DVC remains the dataset version-control tool.

**Impact:** FR-03, AC-02, and the storage references in architecture, setup,
traceability, and demonstration evidence now use Google Drive. Preserve the original PRD and
describe this amendment explicitly instead of claiming an unchanged implementation.

Use DVC's Google Drive integration with the user's private folder and local OAuth.
There is no custom storage API, AWS account dependency, or GCS bucket. The current path is:

```text
Existing Kaggle-attributed dataset -> local raw files -> DVC -> Google Drive
```

The preserved legacy B2 bucket is `varshith-visual-analytics-dvc` in region `us-east-005`.
Its privacy and encryption are user-confirmed, not independently verified through
the provider API. The prefix `visual-analytics/dvc` keeps this project's DVC objects
together. The remote URL, endpoint, and region now live in `.dvc/config`.
Authenticated transfer requires the replacement key entered locally.

The previous application key was exposed. It must not be used by this project.
The user confirmed replacement-key creation and revocation of the exposed key
during Phase 2. Use the replacement only. Do not inspect or print secret values
as part of status checks.

The earlier B2 planning budget was target USD 0/month, ceiling USD 1/month.
Drive uses a USD 0 incremental budget within existing account capacity, following
the user's free-storage request. No additional subscription or billing enrollment
was purchased. These are project policies, not configured provider-side caps.
Review available capacity and usage limits before new uploads.

## Google Drive migration verified

On 2026-10-06 the user requested a free storage alternative and implementation of
Google Drive if suitable. DVC Drive support and provider-specific resumable recovery
are prepared. On 2026-10-07 the user supplied folder ID
`15s2R1cbsqHRl_34n0u5r1SdeivW2NAVl`, now configured as the `gdrive` remote with
download verification enabled. Its free capacity, access permissions, OAuth
connection, upload, and full recovery were initially pending.
Google Drive API was observed Enabled in the existing Google Cloud project
`project-8b2f95b5-9279-4486-ab1`. The OAuth app identity form is prepared as
`Visual Analytics DVC` with External testing audience. The console subsequently
confirmed OAuth configuration creation after the user's action. A Desktop client
named `Visual Analytics DVC Desktop` was created and downloaded by the user. The
signed-in owner was added as the app's sole test user. On 2026-10-08, authenticated
checks confirmed the supplied folder is writable, not trashed, and not shared
publicly or domain-wide, with sufficient available account storage. Client and
user-token JSON are stored outside the repository in the user's local application
data directory. Upload and full recovery verification passed on 2026-10-08.
All 10,000 files and 1,320,154,239 bytes matched the original path/content fingerprint.
The download resumed from a retained Drive-only cache, without seeding from the
original dataset/local cache or B2 cache. Evidence: `reports/gdrive-restore.json`.
Drive is now the default; B2 objects and both providers' recovery caches are preserved.
Cache namespaces were moved to match the default-remote configuration change,
without discarding downloaded objects. No billing or payment action was taken.
This is a second explicit PRD storage amendment; Google Drive is distinct from GCS.

Use consumer-account OAuth with an owned Desktop app; no paid Workspace/service
account, paid storage plan, or billing enrollment is part of this migration.
Keep Google client/token JSON outside the repository. Available storage is
shared with the account's other Google services, and API/authentication limits
remain. Existing capacity does not establish that the account uses a free subscription.
OAuth Testing refresh tokens for Drive expire after seven days; resolve that setup
before unattended CI/CD. Cloud storage serves versioned data transfers, not live predictions.

The Drive plugin requires PyDrive2's compatible TLS dependency range. The project
virtual environment now resolves PyDrive2 1.21.3, pyOpenSSL 24.2.1, cryptography
43.0.3, and asyncssh 2.23.1; these compatibility pins are recorded in the lockfile.
Global Python packages are untouched. Future dependency updates must check both
storage plugins rather than keeping incompatible newer TLS packages by force.

## Dataset identity and provenance

Use the supplied dataset in `dataset/`, without another download or in-place edits.
The file inventory contains only PNG images and XML annotations; no source URL,
download version, license file, or preprocessing history was supplied.

The filenames, task, and classes are consistent with
[Kaggle Safety Helmet Detection](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection).
This is a probable source, not verified provenance. Do not invent a download date,
upstream dataset version, or license claim. Local DVC hashes will establish the
identity of the files actually used even while upstream provenance is unverified.

The first local DVC snapshot contains 10,000 files and 1,320,154,239 bytes:

- DVC directory MD5: `cd17bcaf3c75b4bf1ba13f802ecc2055.dir`
- Independent path/content manifest SHA-256: `15b3504dac4555eafbfcfb752ae642a323da03217311c4685f36acaa2c21d401`

Full validation and an isolated local-cache restoration passed with matching
fingerprints. This is evidence of local recovery, not a claim of B2 transfer.
Reports are regenerated under `reports/`; Git records the DVC pointer and this
snapshot identity. Raw file contents are never committed to Git.

On 2026-10-06, the user's terminal reported `10001 files pushed` (10,000 dataset
objects and the directory manifest). B2 sent a Class B daily-cap notification
during recovery. Full B2 content verification remains pending; an upload log is
not a recovery success report. Before the original temporary cache was removed,
4,262 completed objects were copied from that B2-only restore cache into the
retained recovery cache, verifying every object's MD5 against its cache filename.
No original dataset/cache objects were used to seed recovery. Verification now
retains downloaded objects across attempts, isolated by snapshot and remote
configuration, and labels resumed evidence explicitly. Account caps were not changed.

Preserve `helmet=0`, `head=1`, and `person=2`. Keep all three classes and report
per-class results because person annotations are much less frequent. Scene/group
screening precedes split creation. The final test set stays outside tuning; monitoring
reference and current images are disjoint. Any needed split-count adjustment must
be documented rather than silently breaking group separation.

## Phase 3: grouping, splits, and preprocessing

On 2026-10-08, an all-pairs OpenCV DCT perceptual-hash screen (63 bits excluding
DC, normal and horizontally mirrored views, maximum distance 10) found 151
candidate pairs. All eight candidate contact sheets were visually inspected.
Many pairs are recognizable source-image variants with different watermarks,
brightness, or borders; some are incidental matches. The conservative policy
keeps every candidate connection inside one split rather than asserting each
pair is a confirmed shared scene. Connected components produce 135 groups of
278 images, maximum size four. All remaining images are singleton groups.

Source-scene metadata is unavailable. This screening cannot prove full source
independence or detect every crop/viewpoint change. Visual inspection also found
watermarks and reflected/warped border content already present in the supplied
raw files. These are preserved; the pipeline does not invent repairs or silently
remove inputs. Structural label validity does not establish annotation completeness.

Seed 42 and deterministic whole-group allocation preserve the approved counts:
3,500 train, 500 validation, 500 final test, 250 monitoring reference, and 250
monitoring current. Class-presence balancing retains all three classes in every
split. Person-containing images number 110/16/16/8/8 in that order. No candidate
pair crosses any split, including the two monitoring pools. Label counts are used
for stratification before model selection; the final test set stays outside tuning.

OpenCV 4.11.0.86 and its required array dependency NumPy 1.26.4 were installed only
in `.venv` and pinned. The existing 415/416-pixel images are padded to 416x416
without resizing their content; generic downscaling is supported without upscaling.
Use padding 114, centered with an odd extra pixel on the bottom/right. Preserve
zero-origin pixel-box boundaries and convert the transformed boxes to normalized
YOLO xywh. Write lossless uint8 PNGs; leave RGB/tensor normalization to YOLO's loader.
No offline augmentation or denoising is introduced.

Two real DVC stages connect the raw snapshot to `data/splits.json` and
`data/processed`. The lockfile records raw, code, dependency, parameter, and output
identities. The processed output contains 5,000 PNGs, 5,000 YOLO labels, a portable
training YAML, and a preprocessing report (1,270,105,708 bytes total). The training
YAML deliberately exposes only train/validation; test and monitoring require
explicit later selection. The DVC commands target this Windows project environment.

- Split manifest DVC MD5: `5829128ce0486fa652bf8c88081a60a0`
- Processed directory DVC MD5: `d6f5cfdbcae97ff33e9aef81c1df5b75.dir`
- Processed PNG/label manifest SHA-256: `25a92d0c07b880d45be6db7bf5a46e705fb675d8cdaa22207b25ab8fe3c26ed6`

Derived outputs are cached locally and can be regenerated from raw data and the
pinned pipeline. They have not been uploaded to Drive. No model training or GPU
environment changes occurred in Phase 3.

All eleven tests passed. An independent Pillow/XML audit checked every processed
image pixel and all 25,502 YOLO boxes against the source, verified both content
fingerprints, and confirmed that raw sizes/modification times were preserved.
No candidate pair crosses a split. Evidence is in `reports/phase3-validation.json`.
Repeated fixture preprocessing produced identical bytes; repeated DVC execution
skipped unchanged stages. Phase 3 is complete. These checks establish conversion
correctness, not annotation completeness or model quality.

## Local execution and dependencies

All training and services run on the Windows laptop. Use Python 3.11.9 in `.venv`,
without inheriting global packages. Existing compatible versions may be reused in
the isolated environment, but global packages are not upgraded or uninstalled.

The initial model remains YOLOv8n. Ultralytics 8.3.229 and OpenCV 4.11.0.86 were
the approved starting versions. Resolving training dependencies exposed a real
conflict: Ultralytics 8.3.229 caps requests at 2.32.5, while the installed Google
API dependency requires requests >=2.33.0. Its psutil cap also excludes the
working environment. The next patch, Ultralytics 8.3.230, removes these caps;
see the [official release](https://github.com/ultralytics/ultralytics/releases/tag/v8.3.230).
Use that patch without changing the model architecture or downgrading Drive's
working dependencies. OpenCV stays 4.11.0.86. All resolved versions are locked.

On 2026-10-09, torch 2.2.2+cu121 / torchvision 0.17.2+cu121 were installed only
in the project environment. CUDA tensor and torchvision NMS checks passed on
the RTX 3050 A Laptop GPU (4 GB), driver 592.82. The global CPU-only installation
is untouched. The real YOLO smoke run subsequently passed one epoch plus
checkpoint validation with finite losses and metrics. Driver compatibility
alone is not treated as training evidence. The user connected
the charger and Windows confirmed AC power before the training gate.

Phase 2 installed DVC 3.67.1, dvc-s3 3.3.0, dvc-gdrive 3.0.1, and Pillow 10.2.0 alongside PyYAML
6.0.3. `requirements-lock.txt` records this Windows/Python 3.11 environment.
MLflow 3.16.1 was installed and verified in Phase 5. Phase 9 selects Evidently
0.7.23 for local reports. Its resolved dependencies are pinned in the Windows
lock; the CPU serving environment does not need Evidently. Add dependencies only
in the relevant phase.

MLflow uses local SQLite metadata and local artifacts. SQLite is an embedded
metadata file, not a new database service. Compose is approved for the real local
services, with host ports bound to localhost. No frontend or unrelated API is in
scope. Local model artifacts and metadata require preservation for recovery.

## Phase 4 training procedure

Training uses the approved 416-pixel, batch-4 YOLOv8n baseline for 30 epochs,
with AdamW (initial learning rate 0.001, weight decay 0.0005), seed 42, zero
loader workers, and no image RAM cache. FP32 avoids an additional AMP-check
model download and provides a straightforward initial GPU compatibility gate.
cuBLAS configuration is set before the first CUDA operation. Ultralytics'
remaining defaults, including online augmentation, are recorded in `fit/args.yaml`;
the pinned version and hardware remain part of the reproducibility record.

The hardware gate checks real CUDA tensor arithmetic and torchvision NMS. The
smoke run trains one epoch on deterministic 64/32 train/validation subsets with
all classes present, then validates the checkpoint. Its metrics are not baseline
quality evidence. The full baseline uses all 3,500/500 train/validation inputs.
Both start independently from the official pretrained YOLOv8n weights.

Training copies only selected inputs into ignored run directories so library
caches cannot alter the DVC-managed dataset. Final-test and monitoring pools
are excluded. Existing run directories are preserved, not overwritten. Lineage
includes raw/split/processed DVC identities, checkpoint hashes, package/hardware
details, parameters, Git SHA/dirty status, and snapshots of the actual source
and configuration. Uncommitted work is not represented as a clean Git revision.

Evaluate the best checkpoint on validation, recording precision, recall,
mAP50, mAP50-95, and per-class results. Validation supports selection and does
not replace the untouched final test. Check all epoch losses and reported
aggregate metrics for finite values. Local checkpoints are not approved stable
models. MLflow tracking/registry integration is the following phase.

The 30-epoch baseline and independent best-checkpoint validation completed on
2026-10-09. Recorded epoch time totals 4,178.35 seconds (69.6 minutes). All epoch
values are finite. Validation precision/recall/mAP50/mAP50-95 are
0.60943743 / 0.56788898 / 0.61936066 / 0.39427597. Per-class mAP50 is helmet
0.93860, head 0.90010, person 0.01939. Person precision and recall are both zero
at the reported max-F1 operating point; its evaluation covers only 16 images and
118 boxes. Low class representation is a limitation, not an established sole
cause. This baseline requires targeted error analysis/improvement before stable
promotion. Phase 4 execution is complete; production suitability is not claimed.

The best checkpoint SHA-256 is
`5f725cb9f769a68ddb8f9518a979c7999b20e152d9e406e7df34f923df43fe4f`.
`reports/phase4-validation.json` confirms checkpoint/initial-weight/source-snapshot
hashes, exact train/validation membership, finite epoch values, unchanged DVC
stages and preserved raw sizes/modification times. Preprocessing was reproduced
after dependency-lock changes, with the exact same split/processed hashes as
Phase 3. No data transfer was needed. All thirteen tests passed; CUDA tensor/NMS,
smoke training and full baseline evaluation provide separate real execution
evidence beyond the mocked metric-reporting unit test.

Ultralytics initially used a workspace fallback for its local settings because
the requested parent directory did not exist. After the run, settings/fonts were
preserved under ignored `artifacts/ultralytics-settings/`, and location resolution
was verified. The entry point now creates that parent before import. Run source
snapshots preserve the pre-fix code actually executed; this housekeeping change
does not alter training parameters or the completed results. Global user settings
were not used. The next phase should record this baseline as experimental in
MLflow, without automatically marking it stable.

## Phase 5: local experiment tracking and registry

On 2026-10-09, MLflow 3.16.1 was installed in the project environment. Its
Databricks SDK dependency requires Protobuf <7, so Protobuf changed from 7.36.2
to 6.33.6. The constrained dependency resolution preserved the existing GPU,
NumPy/OpenCV, DVC and Drive/TLS packages. The SDK is an MLflow dependency; no
Databricks account, remote tracking service, or new project platform was added.
All resolved packages are pinned, `pip check` passes, and CUDA tensor/NMS plus
both DVC storage-plugin imports still work. Training retains its original
dependency snapshot; the later tracking environment is recorded separately.

Use the approved `artifacts/mlflow/mlflow.db` SQLite store and
`artifacts/mlflow/artifacts/` local artifact store. The dashboard binds only
127.0.0.1:5000 with one worker. The Python importer uses the same SQLite store
directly; HTTP access was separately verified. Usage telemetry is disabled.
Neither cloud uploads nor billing changes are part of this phase. Preserve the
database and artifacts together; no automatic backup/service installation is claimed.

`src/tracking/mlflow_utils.py` validates a completed training bundle and logs its
parameters, epoch curves, global/per-class validation metrics, source/configuration
snapshot, dataset identities, input memberships and native YOLO checkpoint. It
records the original Git SHA and dirty status without claiming a clean commit.
The Phase 4 baseline is imported after training; MLflow timestamps describe the
import, while original training start/time/steps remain in its evidence. Future
full runs invoke the same importer after successful evaluation. Smoke runs do
not automatically create registry versions. Named run directories allow later
experiments without overwriting the baseline.

The SHA-256 evidence fingerprint identifies retries. Interrupted logging or
registration can resume serially using the existing run/version; changed evidence
is a different bundle. Concurrent import locking is deliberately not implemented:
run one importer per bundle at a time. Damaged input evidence is rejected before
creating a tracking record. Retrieval compares registry checkpoint bytes, archived
evidence, metrics, and epoch history with the originals. MLflow 3.16's numeric
model resolver consults SDK defaults, so verification explicitly binds both
tracking and registry URIs to the intended store.

The registered model uses native `best.pt` plus format metadata, not a fabricated
PyFunc signature or extra wrapper class. The approved FastAPI stage will load
retrieved weights with Ultralytics; MLflow's generic model-serving endpoint is
not used. Registry retrieval through `models:/hard-hat-detector/1/best.pt` and
real CPU inference were tested. This inference check uses a validation image and
does not supply a new quality estimate.

- Experiment / model: `hard-hat-detector`.
- Run ID: `473308c0c5df40d2b1095f6a5779441c`.
- Numeric model version: `1`.
- Approval status: `experimental`.
- Checkpoint SHA-256: `5f725cb9f769a68ddb8f9518a979c7999b20e152d9e406e7df34f923df43fe4f`.
- Evidence fingerprint: `9000d9c427efec8fbe7f13f228ae042c14e125de9cd91bb3b3c2f4f22f3eb38f`.

Exactly one real run and one model version exist after retrying. Both survive
server restart. Neither `stable` nor `previous_stable` is assigned: person-class
performance remains weak, and stable approval has not occurred. Registration is
not approval; the project has no valid rollback target yet. The explicit alias
policy remains unchanged, and temporary-registry tests confirm that importing
a new experimental version preserves any existing stable alias. No second model
version or fake stable model was invented for the demonstration.

All seventeen tests passed, including four real SQLite/registry tests for
idempotent imports, interrupted logging/registration, corruption rejection,
Windows paths with spaces, and alias preservation. The real baseline's retrieval,
HTTP checks, restart persistence, and CPU load/predict evidence are recorded in
`reports/phase5-validation.json`; `artifacts/training/baseline/mlflow.json` links
the training directory to MLflow. Phase 5 tracking/registration is verified;
stable selection remains an explicit outstanding quality/approval gate.

After the dependency-lock change, DVC reproduced the split and preprocessing
stages. Their output hashes remained identical and the final DVC status was clean.
The metadata fingerprint of all 10,000 original dataset files also remained
unchanged. These checks are recorded in the Phase 5 validation report. The local
dashboard displayed the completed baseline and experimental version 1 without
stable aliases.

## Phase 6: CI preparation and runner proposal

The PRD requires actual GitHub Actions build/test/deployment execution. The user's
phase order places CI/CD before Docker/FastAPI implementation, so this increment
prepares checkout, locked dependency installation, compilation, and isolated
tests. Container build and deployment must be added with the real serving code
in Phase 7. No placeholder Dockerfile, successful no-op deployment, or automatic
model promotion is introduced. FR-10 / AC-08 remain pending end-to-end evidence.

The proposed repository-scoped Windows runner lives outside the development
checkout at `F:\visual-analytics-actions-runner`, uses `_work` for disposable
clones, and runs manually as the current non-elevated user without a service or
startup task. Registration has not occurred; the user's explicit prior instruction
requires separate setup/security approval. A separate checkout prevents accidental
checkout cleanup of the raw dataset and live registry but does not isolate the
Windows account's files or credentials from trusted workflow code.

`.github/workflows/ci.yml` is manual-only on private `main`, with read-only contents
permission, immutable checkout-action SHA, credentials persistence disabled, a
single concurrency group, native PowerShell failure propagation, and a pre-check
of the runner work path before checkout. Python must be exactly 3.11.9. Each job
gets a separate virtual environment and the existing full Windows dependency lock.
Tests use temporary fixtures; cloud credentials, dataset pulls, real MLflow
registration, and training are not CI steps. Logs/summary record the dispatched
commit and dependency-lock hash. The workflow is not a substitute for reviewing
repository writers and changes before running code on the laptop.

The local dry run copies only source/scripts/tests and uses the current pinned
environment. actionlint 1.7.12 was checksum-verified and used from ignored reports
storage, not added to application dependencies. Native PowerShell parsing,
workspace refusal, and failure propagation were checked. The verification report
explicitly distinguishes this preparation from a clean install or GitHub run.
The runner proposal, setup steps, permissions and cost references are in README.

On 2026-10-09, read-only Git remote discovery advertised no HEAD/main/master refs;
no push was made. Docker Desktop's Linux-engine pipe was unavailable. Start and
recheck Docker before serving implementation, not as a prerequisite for source
tests. No local runner directory/process was found at the proposed location.

### Runner approval and registration — 2026-10-11

The user explicitly approved the documented runner setup. Runner 2.338.0 was
downloaded from the official actions/runner release and verified against its
published SHA-256. GitHub confirmed this repository is private, the existing
credential has administrative access, and no runner was previously registered.
`visual-analytics-laptop` is now registered using the separate approved directory,
standard Windows/x64 labels, and `_work` checkout folder. It is not a Windows
service. Registration used the existing Git Credential Manager sign-in and a
short-lived token passed only through the runner process environment. Token values
were not printed or stored by the setup script. The runner is manually launched
with `--once` for the first validation job and exits after that job.

The first real GitHub job passed preflight but failed checkout because Git could
not create `.git/hooks` under the 238-character workspace path. Windows long-path
support was already enabled. The workflow now sets `core.longpaths=true` through
job-scoped Git environment configuration; Git initialization succeeded in the
same directory with this setting. No global Git or Windows setting was modified.
The failed run remains recorded rather than being presented as successful CI.
The second attempt passed checkout but could not launch the virtual-environment
Python executable under that same long directory. CI now builds its fresh Python
environment in `RUNNER_TEMP\venv`, outside the long repository path. This is a
job-temporary environment, not reuse of the development virtual environment.
Run `38143054792` then installed all locked dependencies successfully, passed
`pip check`, compiled sources, and passed all 17 tests in 59.295 seconds. Its
summary step failed because `Get-FileHash` was unavailable in the runner's
PowerShell session. The summary now uses the already-required Python standard
library. It was verified from the actual long checkout and with an empty
PowerShell module path. The next run also skips dependency bytecode precompilation
because its environment is temporary; project compilation remains a separate step.
On 2026-10-12, two attempts of run `38211578040` hit intermittent TLS record
errors fetching different PyPI indexes (`pydantic-core`, then `wrapt`). The same
pins had installed successfully in the earlier run. A verified HTTPS request
also confirmed the pinned pydantic-core release was available. The install step
now permits at most three attempts, logging failures and waiting five seconds
between attempts. Exhaustion fails the job; no TLS validation bypass or package
version change was introduced.

### GitHub CI validation passed — 2026-10-12

Run `38212488144` succeeded for commit
`dac82c43d53ed9f08cb526a534f545082bda4468`. The real repository-scoped Windows
runner checked out that commit, installed dependencies into a fresh temporary
environment, passed `pip check`, compiled the sources, passed all 17 tests in
49.078 seconds, and wrote the validation summary. The complete job took 6 minutes
33 seconds; installation succeeded on its first attempt in this run. The runner
exited after the job and GitHub confirmed it offline/not busy. No Windows service
or startup task was installed.

The original dataset metadata remains unchanged for all 10,000 files and DVC
status is clean. No training, dataset transfer, model promotion or deployment
was performed by CI. `reports/phase6-validation.json` records both the successful
run and previous failures; the full successful log is retained locally.
FR-10 / AC-08 have real CI validation evidence, but their complete build/deployment
acceptance remains dependent on Phase 7's Docker/FastAPI implementation. This
phase does not establish an approved stable model or a usable rollback target.

## Health decisions and rollback

PSI/KS are evidence sources, not rollback commands. The approved policy combines
operational, readiness, labeled quality when available, and drift evidence using
transparent rules. Initial thresholds remain calibration values. Persist the
model identity, observed signals, policy version, and matched rule for each
decision. Do not claim labeled quality from unlabeled traffic.

Stable means explicitly approved after evaluation, not the most recently created
registry entry. Before candidate deployment, preserve the previous stable target
and its known-good deployment settings. Use the same deployment procedure for
normal deployment and rollback, with numeric model identity and recovery checks.
The demo-only server-error fault is disabled by default and must not survive the
restoration of the known-good deployment.

## Phase 7 serving implementation — 2026-10-12

Docker Desktop was started and its Linux engine verified with `hello-world`.
The service uses a Python 3.11.9 Bookworm base pinned by digest, a separate Linux
CPU dependency closure, FastAPI 0.143.0, Uvicorn 0.54.0 and the existing YOLO/PyTorch
versions. Pip consistency passed inside the image. There is no CUDA, MLflow or DVC
dependency in the image. The Windows lock and data stages are unchanged.

The host retrieves all registered evidence and verifies its bundle/checkpoint
hashes before staging a numeric model version. Stable/previous_stable aliases
are the only approved deployment references; experimental use requires an explicit
version and flag. No real aliases/tags were changed. Local version 1 is still an
experimental model with weak person-class performance. The container receives only
read-only staged files and uses CPU inference, preserving the training letterbox
behavior and recording its settings. No cloud storage is involved in serving.

The API accepts raw PNG/JPEG bytes, avoiding an unnecessary multipart dependency.
The recorded starting settings are confidence 0.25, IoU 0.70, 300 maximum boxes,
two CPU threads, 4 MiB encoded-body and 4,194,304 decoded-pixel limits. These bound
local work and freeze prediction behavior; they are not health calibration results.
Readiness follows verified loading and warm-up; deployment additionally sends a
prediction probe. Image/coordinate/identity checks and seven detections on an
existing validation image passed. No final-test or monitoring images were used.

Image builds use an allowlisted context, with no datasets, credentials or models.
Containers run as a non-root user, with a read-only root filesystem and a temporary
write area, published only on 127.0.0.1:8000. Deployment receipts preserve exact
image IDs, code revisions, model lineage and settings. Debian OS dependencies are
resolved at build time; a byte-identical rebuild is not claimed. Replacement has
brief downtime. Automatic recovery, metrics and drift remain their later phases.

The approved repository runner now builds this image after tests and can invoke
the same deployment procedure with explicit dispatch settings. Default dispatch
only validates/builds. Local model access is an explicit host path; there is no
claim of GitHub-hosted access to this laptop.

GitHub run `38250748811` passed for commit
`6cdbf6f2541becac00466e1507e263ffc5135db7`: fresh locked installation, pip consistency,
compilation, all 22 tests in 84.781 seconds, CPU image build, and experimental
version 1 deployment with matching readiness and a real prediction probe. The
deployment receipt records a clean source tree and image ID
`sha256:dadbbc1f347df05c25ade128314ab152fa75ec5351000158b781e28680582b90`.
The once-only runner exited after success. The container remains available locally.
Registry aliases are still empty. Original data metadata and DVC stages remain
unchanged. `reports/phase7-validation.json` links the local and CI evidence.

## Phase 8 operational monitoring — 2026-10-14

The approved Prometheus/Grafana phase uses the official Python client 0.26.0,
Prometheus 3.15.0 and Grafana 13.2.3. Container images are pinned by digest.
No additional monitoring platform, cloud subscription or frontend was introduced.
The API exports prediction counters and latency histograms by outcome, readiness,
and model/inference identity. Counters/histograms identify model version and
deployment ID. Client errors/cancellations are separate; health/metric requests
do not become prediction traffic. The p95 dashboard is a histogram estimate,
not an exact request-level statistic or an approved decision threshold.

The registry-verifying deployment path now attaches the API to a project-owned
Docker network with the stable DNS name `api`. Monitoring uses Compose separately
from model deployment. All three ports remain loopback-only. Grafana is provisioned
as a local anonymous Viewer with no initial admin user, signup or login UI;
datasource/dashboard editing remains in Git. This is a deliberate local-review
configuration, not a public-hosting configuration. Credentials were not created.

Prometheus uses persistent local storage with two-day/1-GB block retention;
the WAL/head and total container storage are additional. Grafana data is also
persistent. Automatic plugin installation, telemetry and update checks are disabled.
The services have no auto-restart policy or Windows startup installation.

Local verification passed 23 tests and exercised ten real validation-image
predictions plus two invalid inputs. All seven dashboard queries worked through
both Prometheus and Grafana. Stopping the API produced `up=0` and unavailable
readiness; restarting restored collection. Restarting monitoring preserved
historical samples and the provisioned dashboard. Browser automation was not
connected, so visual screenshot inspection remains unperformed; dashboard
definitions, permissions and data queries were checked through Grafana's API.

The new client changed the Windows lockfile. Both DVC stages were rerun, and
their output hashes stayed identical; only dependency hashes in dvc.lock changed.
Raw inputs were preserved. No retraining, model promotion, drift computation,
overall health classification or automated rollback was performed. The isolated
500-metric test uses mocked inference; the production demo fault remains off.
GitHub run `38518901297` passed for commit
`27ea7e337e41ab150db59a1e16de154e8deade1d`: fresh dependencies, compilation,
23 tests in 104.347 seconds, Compose/promtool checks, image build, and verified
experimental deployment. Prometheus picked up the replacement API automatically.
The runner exited after the job; the three local services remain running.
The dashboard-only follow-up adds explicit label-to-column transforms to its
two tables; their provisioned definitions and incoming labels were checked via
Grafana's API. It does not modify the CI-tested serving image. Evidence is in
`reports/phase8-validation.json` and the associated integration/GitHub reports.

## Delivery gates

The private GitHub repository already exists; use it and preserve remote changes.
The user approved Phase 1 foundation after Phase 0 review. Subsequent phases are
delivered incrementally with their own validation reports.

The self-hosted GitHub Actions runner received separate setup/security approval
on 2026-10-11 and has been registered. Do not imply that GitHub-hosted jobs can
reach local services without an approved mechanism. No cloud deployment is planned.

Phase 2 Google Drive recovery passed. Incomplete legacy B2 recovery is no longer
a delivery gate. Phase 3 grouping and preprocessing are implemented with the
documented scene-independence limitation. CUDA checks, smoke training, and the
full Phase 4 baseline/evaluation passed. Phase 5 local tracking, registration,
and retrieval are verified. Person detection remains a model-quality limitation
before stable promotion. Phase 7 local container build and experimental deployment
and the expanded GitHub build/deployment are verified. Phase 8 operational
monitoring, persistence, outage/recovery and updated CI also passed. Still deferred: stable
model approval, later dependency resolution, and measured threshold calibration.
These do not prevent the repository foundation from being completed.

## Phase 9: fixed drift measurement contract

`src/monitoring/drift.py` collects controlled replays of the reserved 250-image
reference/current pools through the real localhost prediction API. It reads raw
images without writing to the dataset. These are sampled demonstration observations,
not a claim of continuous production traffic capture. Labels and the final-test
pool are not used. HTTP failures abort collection instead of becoming empty detections.

Before comparison/calibration, the following definitions were fixed:

- Brightness: mean OpenCV uint8 grayscale intensity (0..255) on the EXIF-oriented
  decoded image at original resolution, before serving resize/padding.
- Sharpness: natural `log1p` of the population variance of OpenCV's float64
  Laplacian (`ksize=1`, default reflected border) on that grayscale image.
- Output: maximum retained detection confidence after configured confidence/NMS
  filtering; an empty prediction is null and excluded only from this comparison.
  Empty-image counts and rates remain separately visible for both populations.
- PSI: ten nominal reference quantile intervals with duplicate edges collapsed,
  explicit infinite tail bins, and `nextafter(reference_max, +inf)` to retain the
  maximum in a finite bin (including constant references). Add 0.5 to each bin
  count in both populations, then normalize and sum `(current-reference) *
  ln(current/reference)`. Record every actual edge. Current data never chooses bins.
- KS: SciPy two-sided `ks_2samp(method="auto")`; require both p < 0.01/3 and
  statistic >= 0.10. Ties and related images limit the ideal independent continuous
  sample interpretation; these initial calibration thresholds do not prove accuracy.
- Require 200 non-null samples in each population for each feature. A positive
  available feature supplies drift evidence; otherwise missing features leave
  overall evidence unknown instead of silently returning no drift.
- Current population's oldest observation must be at most 3,600 seconds old and
  its collection span at most 1,800 seconds. Reference must finish before current
  starts. The immutable reference remains reusable while its model/checkpoint,
  inference settings, raw/split identities, feature definition and OpenCV version
  match. It is a fixed baseline, not a live health measurement. `current_valid_until`
  must be checked again by a future health consumer; historical reports stay historical.

These freshness/binning choices are initial demonstration settings, fixed before
viewing these pools' results; the previously approved PSI/KS thresholds are unchanged.
Both IDs and image-byte hashes are checked for cross-population overlap. Each
successful API response includes its inference-settings SHA-256; collection checks
that and the model/version/checkpoint to prevent mixing prediction baselines.
Invalid/non-finite/duplicate observations are rejected. Stale, insufficient, or
incompatible evidence is explicitly unavailable. Empty predictions are never zero
confidence, and confidence changes are never labeled accuracy degradation.

Evidently uses registered custom PSI/KS statistical tests to apply this exact
contract, rather than its automatic test selection/default bins. Reports export
local HTML and JSON plus a machine-readable summary with observation hashes,
identities, windows, package versions, thresholds, counts, scores, and reasons.
Unavailable columns are omitted from Evidently charts and listed in the summary;
an entirely unavailable comparison has no statistical chart. Existing output
paths are preserved. No Evidently cloud service, scheduler, additional endpoint,
health classifier, or rollback action is introduced.

References: [Evidently drift customization](https://docs.evidentlyai.com/metrics/customize_data_drift),
[Evidently 0.7.23](https://pypi.org/project/evidently/0.7.23/).

The dependency resolver added Evidently and its required packages without changing
any existing pinned package version; `pip check` passed. Both DVC stages were
actually reproduced after the lock change. Raw content SHA-256 remained
`15b3504dac4555eafbfcfb752ae642a323da03217311c4685f36acaa2c21d401`,
and processed content SHA-256 remained
`25a92d0c07b880d45be6db7bf5a46e705fb675d8cdaa22207b25ab8fe3c26ed6`.
Split and processed DVC output hashes are unchanged; only dependency-lock metadata
changed in `dvc.lock`. Raw file sizes/modification times remain unchanged.

Phase 9 finished with 29 passing local tests (90.002 seconds) and 29 passing
GitHub tests (188.266 seconds). GitHub run `38560936350` validated commit
`ee002e52350485d6a3a157967dde7dde1891dfd7`, built the CPU image and deployed
experimental registry version 1 with real readiness and prediction probes.
`reports/phase9-validation.json` records the full evidence. The separate health
decision and rollback phases remain unimplemented; registry stable aliases remain
unassigned. Reopening the session later found Docker stopped; the same existing
containers were restarted for the handoff. This does not refresh historical drift
observations or enable automatic service startup.

## Phase 10: host-side health decisions

`src/monitoring/health.py` combines direct readiness probes, exact Prometheus
counter/histogram differences, a measured latency baseline, labeled audit results,
and Phase 9 drift summaries. It emits `healthy`, `unhealthy`, or `unknown` with
per-signal reasons, deployment/model identity and a short validity deadline. It
never changes a model, registry alias or container. Readiness failures can be
evaluated while the API is down because the checker runs on the Windows host.

The implementation fixes the following initial calibration choices before the
live checks. Existing approved numeric breach thresholds remain unchanged:

- Readiness: three direct probes spaced at least ten seconds apart. Require three
  consecutive failures after the 60-second startup grace to declare a service
  breach. A successful latest probe resets that failure sequence. The actual
  Docker container start time identifies restarts; changed starts discard earlier
  probes. Unknown runtime start makes failure evidence unavailable. A stopped or
  unavailable Docker daemon does not crash the checker or imply healthy service.
- Operational evidence: the two latest completed UTC-aligned 120-second windows.
  For each nominal boundary use the latest real scrape at/before that boundary,
  within 15 seconds. Adjacent windows share a boundary sample, never an increment.
  Require continuous successful scrapes with gaps <=15 seconds, matching counter
  and histogram timestamps, no decreasing counters or changed creation times,
  and data from the current container process. Missing/ambiguous series, restarts,
  scrape gaps, and fewer than 100 eligible predictions per window are unavailable.
- Eligible traffic is success + server error. Client errors are excluded from
  both error-rate and latency denominators. Exact differences avoid extrapolated
  `increase` counts. Cumulative histogram differences provide an interpolated
  p95; an overflowing +Inf quantile is unavailable, not a fabricated finite value.
- Error >=5% in both qualifying windows or p95 above `max(2 seconds, 2*baseline)`
  in both windows is unhealthy. A single critical breach window remains pending
  (`unknown`) until sufficient persistence is observed.
- The latency baseline captures the larger p95 of two qualifying, error-free
  windows at or below the two-second floor. It records CPU/host and inference settings,
  source model and window counts. It expires after 24 hours and must match the
  current host/CPU and inference settings. This is operational calibration and
  does not approve experimental model quality or create a stable alias.
- Labeled audit: require an actual registered `stable` baseline and the deployed
  candidate, both on the same >=200-image labeled monitoring batch. Reject stale
  (>1 hour), incompatible, non-finite, or mismatched-batch evidence. An absolute
  mAP50-95 drop >=0.05 is unhealthy (1e-12 arithmetic comparison tolerance).
- Fresh positive drift plus >=2% server errors in both qualifying windows is
  unhealthy only when the drift current-population interval overlaps the latest
  operational window. Recheck drift identity, split, policy and expiry. Positive
  partial drift can support this combined breach; incomplete drift alone leaves
  evidence unknown. Drift by itself never makes the deployment unhealthy.
- Any confirmed breach wins over missing independent signals. Otherwise every
  required signal must be available to declare healthy. Missing stable quality
  evidence therefore remains unknown even when serving and latency look normal.
  Malformed optional evidence is unavailable; it cannot erase an independently
  confirmed readiness failure. Repeated checks of the same windows do not create
  additional persistence observations.

Decision validity is bounded by the latest probe's 15-second freshness and the
shortest applicable evidence expiry. Historical JSON is not a current health
measurement. A future rollback consumer must check identity and freshness again.
The CLI is a bounded check, not an installed background service or scheduler.

`src/monitoring/audit.py` resolves verified MLflow checkpoints through the existing
staging procedure. With an approved stable alias, it first checks DVC status, then
copies only monitoring-current images/labels to a new audit directory. Both models
use identical isolated inputs and explicit CPU AP-evaluation settings: 416 pixels,
batch 4, confidence .001, IoU .7, max 300 detections, no augmentation/half precision.
The low confidence is for offline AP evaluation, not a serving configuration change.
Copied-input hashes protect labels and pixels from evaluator cache writes or repairs.
The report records both checkpoint identities, batch hash, settings and real
mAP50-95 values. When stable is absent, it writes an unavailable result without
inventing a reference or changing aliases. Current v1 remains experimental.

No new package, API endpoint, cloud transfer, training run, or rollback mechanism
is introduced. The new configuration values are in `health`, outside the DVC
stage parameter dependencies. See [Prometheus HTTP range vectors](https://prometheus.io/docs/prometheus/latest/querying/api/)
for the raw-sample API used by the checker.

Phase 10 local validation passed 39 tests. The real two-window test measured
110/250 eligible requests, excluded three client errors, and captured baseline
p95 0.089151 seconds. Normal operation returned unknown only for missing stable
quality evidence; a real stopped API returned unhealthy after three failures.
Restart restored readiness and made pre-restart windows unavailable. No model
switch occurred. An approved stable comparison is still unavailable; fixtures
exercise its implementation and input isolation without claiming real quality
results. Evidence: `reports/phase10-validation.json` and `reports/phase10-live/`.
DVC status and original file metadata are unchanged; no dependency or serving API
code change required rebuilding data or replacing the serving image. GitHub
run `38692757040` passed on October 15, 2026 against implementation commit
`9f641efec87ddfe50e3d495c91e9d77632b18f68`, including all 39 tests, locked dependency
consistency, source compilation, monitoring configuration validation and the CPU
image build. Deployment was `none`; the existing experimental API was retained.
Run `38603116145` had ended during dependency installation after the runner logs
stopped abruptly. The exact interruption cause is unknown; the unchanged commit
passed on retry. Both attempts are preserved in the local phase report.

Post-restart service checks passed for a real API prediction, Prometheus target
availability and Grafana database health. No registry alias was added. Saved live
test decisions are historical evidence after expiry, not current health claims.
Phase 10 is complete; Phase 11 will add rollback controls, with stable selection
and real comparative quality still subject to the existing approval requirements.

## Phase 11: recorded-target rollback and bounded recovery

FR-19 selects the approved predecessor captured before a candidate replaces it,
not a guessed numeric neighbor or the registry's latest version. Normal deployment
verifies a ready predecessor's model against the local MLflow evidence bundle and
records the model manifest, immutable Docker image ID, numeric version and actual
loopback port before removal. It only captures an approved deployment whose
version still belongs to `stable` or `previous_stable`. Readiness is an operational
check; the existing registry approval remains the quality gate.

FR-20 uses the existing `deploy()` implementation for both paths. Artifact hashes
and alias approval are checked again before rollback; recorded inference/class
settings are restored rather than regenerated from candidate configuration. The
original image must be available locally. No remote storage or registry upload
is needed. Restored port information is also read by the host health collector.

The bounded `src.monitoring.rollback` CLI accepts a fresh Phase 10 decision or
collects evidence itself. It compares the current policy and model/container ID,
checks Docker's process start time, and re-evaluates the evidence before acting.
A quality-triggered decision must still refer to the current stable baseline.
The final removal boundary checks deployment/process identity and decision expiry
again. Unknown health and drift alone do not authorize rollback.

Normal deployment and rollback serialize through an exclusive local lock. Before
replacement, rollback flushes an exclusive attempt file keyed by failed container
ID. A failed, completed or interrupted attempt prevents another attempt for that
deployment. No retry loop, inferred cooldown, or automatic alias mutation is
introduced. An interrupted process leaves its lock/attempt evidence for manual
inspection; these records must not be automatically deleted to resume a switch.
Receipt pointers are published with atomic file replacement. A rollback-created
deployment has no automatic further rollback target, preventing oscillation.

FR-21 requires three matching readiness responses ten seconds apart, twenty
successful predictions from distinct validation images and fresh raw Prometheus
samples for `up`, the restored model identity and its readiness. Raw range samples
are used so an instant-query evaluation timestamp cannot disguise an old scrape.
Prediction responses must match the restored model and inference hash. Missing
or stale monitoring fails recovery confirmation, even if serving has restarted.
The attempt retains partial probe evidence and the restored deployment receipt;
it does not swap back or retry. Operational `recovered` is not overall model-quality
`healthy`, and old drift/audit/latency reports are not automatically relabeled.

No continuous service, demonstration fault, new package, training run, storage
transfer or stable promotion is introduced. The real registry remains experimental
and lacks an approved target. Controlled fixtures test switching and failure
paths; the live project must refuse rollback until a reviewed stable predecessor
has actually been recorded. Real approved-model recovery remains an acceptance
gap for Phase 12, rather than being simulated by approving the weak baseline.

Local validation passed 53 tests, including a revoked target alias during evidence
collection and a changed stable quality baseline. The available registry/artifact
restoration path uses a real disposable MLflow registry; Docker switching is
controlled in fixtures. Live checks against the existing experimental service
passed three readiness probes, twenty real predictions and fresh monitoring
verification without replacing it. The live controller reported no approved target
and made no switch. Dataset metadata, DVC status and registry aliases are unchanged.
GitHub run `38696131541` passed on October 15, 2026 for implementation commit
`ede7e883a01832567714ea29a74296cea89f4809`: all 53 tests passed in 169.897 seconds,
with locked dependency consistency, compilation, monitoring configuration validation
and the CPU image build successful. Deployment was `none`; the run did not replace
the experimental API. The implementation is validated, with the real approved-model
switch explicitly unverified until a reviewed target exists.

## Phase 12: repeatable acceptance audit

One project-specific command, `scripts/validate_pipeline.py`, composes the existing
dataset, registry, serving, drift and health functions. It writes to a new directory
under `reports/` and does not deploy, train or transfer data remotely. DVC and the
original content are verified again; remote recovery and the full preprocessing
audit are identified as historical evidence with matching current identities.

The live path measures two qualifying traffic windows, exports a new current-pool
Evidently report against the preserved reference, evaluates current health, checks
that normal-operation evidence cannot authorize rollback, and queries every
provisioned Grafana panel. Deployment receipt, Docker process start, registry aliases
and original dataset metadata must stay unchanged throughout the check. The
reported state is `completed_with_acceptance_gaps` when checks complete but
provenance, stable quality or actual approved rollback evidence is missing.

Phase 12 validation does not silently approve version 1 or invent a second model.
The clean-commit live run (`d4acdecff2e16c0deb0e067eda3e5e1849d77fe0`) completed on
October 15, 2026. Content and metadata of all 10,000 dataset files matched; DVC
was clean. Training/registry/deployment identities matched. The two traffic
windows contained exactly 110/250 eligible predictions with three excluded client
errors. The measured baseline p95 was 0.069444 seconds. Fresh PSI/KS reported no
drift; all seven Grafana panel queries passed. Overall health remained unknown
solely for missing approved stable quality evidence, and the rollback gate refused
that decision. Dataset, deployment receipt/process and registry aliases stayed
unchanged during the run.

Acceptance is deliberately incomplete: AC-02 through AC-12 are verified in the
documented local scope; AC-01 and AC-13 are partial; AC-14 and AC-15 are blocked.
The user confirmed that the original Kaggle URL/version is unknown. No inferred
source or version is substituted. The other gaps require a reviewed stable model,
real comparative quality evidence, a real switch to its recorded predecessor,
and fresh post-switch monitoring. Historical remote recovery and fixture tests
retain their stated scope. The final-test pool is still unevaluated.

The final audit is in `reports/phase12-live/validation.json`; its reproducible
command and acceptance table are in the README. GitHub run `38738939200` passed
on October 15, 2026 against the same implementation commit as the live run. All
53 regression tests passed in 140.484 seconds, along with dependency consistency,
compilation, monitoring configuration validation and the CPU image build.
Deployment was `none`. Final container checks showed the existing API healthy
and Prometheus/Grafana running. The validation phase is complete with the stated
acceptance gaps; full project acceptance remains incomplete.

## Person-class follow-up: annotation completeness before retraining

On October 16, 2026, `scripts/analyze_person.py` ran the existing baseline on all
500 validation images at confidence floor .001 and recorded diagnostic IoU50
matches across fixed cutoffs. It reads train/validation label statistics, uses CPU
inference, and excludes final-test and monitoring pools. The recorded Ultralytics
AP remains the model-selection metric; these counts diagnose failure modes.

The dataset supplies 472 person training boxes in 110 images, versus 13,304 helmet
and 3,902 head boxes. Validation supplies 118 person boxes in 16 images. At .25,
zero person predictions matched a supplied label; at .01, 35 matched and 1,149
were unmatched. All six highest-confidence person-negative validation images have
visible people but no person objects in their raw XML. Selected training examples
show analogous omissions. This establishes concrete annotation-completeness issues,
not their dataset-wide rate or a proven causal effect on model training.
The original annotation guidelines are unknown. The completeness review assumes
a consistent visible-person detection task; it does not establish the original
annotator's intent or authorize automatic relabeling.

Unmatched predictions are false positives against the supplied annotations; they
must not all be described as actual false alarms. Model scores remain weak, but
person precision cannot be interpreted as visual correctness without consistent
ground truth. No score threshold was changed to conceal the issue. No detection
cap was hit, and only three labeled validation person boxes have minimum side
below 16 pixels, so those observations do not support treating cap/size changes as
the primary correction.

The initial settings fallback was corrected by creating the report-local settings
directory before import. The task-generated workspace settings were moved into
the first report for audit. A clean-commit rerun produced byte-identical prediction
observations. Raw file metadata, DVC state, checkpoint bytes and registry aliases
were preserved. Five diagnostic matching tests and the 58-test local suite passed.
The next intervention is a documented annotation policy and a separate reviewed
derived-label version, followed by an isolated training candidate. Existing raw
data, split membership and experimental-model approval status remain protected.

GitHub run `38860292983` passed on October 16, 2026 against implementation commit
`c14683d5f47dd9014ec15b9d6d20e406a2c1f197`. All 58 tests passed in 195.586 seconds;
dependency consistency, compilation, monitoring configuration validation and the
CPU Docker build also passed. Deployment was `none`. Final serving/monitoring
checks passed for the unchanged experimental deployment. The diagnostic summary,
visual review, preservation checks, draft annotation policy and CI evidence are
recorded in `reports/person-analysis-validation.json` and
`reports/person-analysis-v2/`. No label correction or stable approval was applied.

## Traceable correction review and completion scope

The request to finish the remaining work authorizes progressing the annotation
correction workflow. `docs/annotation-policy.md` now defines a visible-person
review with source hashes, fixed splits and explicit unresolved-fragment handling.
The review set covers every training/validation image; raw files and held-out
pools remain protected. `scripts/review_annotations.py` only exports a separate
candidate after every record has reviewed status and no open issues. It does not
alter training inputs, DVC stages, registry aliases or the running deployment.

The first three corrected boxes were inspected on raw images and overlays by
Codex. Nine other known omissions have unresolved fragment issues; 3,988 records
remain pending. This is a partial review, not a completed corrected dataset.
All 64 local tests passed (six new annotation-workflow tests). Raw metadata and
DVC state remain unchanged. The previous successful CI evidence still applies
to the preceding implementation, not these new source files.

A scope question was presented because the existing model has not passed
three-class quality review: continue full annotation correction and retraining,
or explicitly accept the existing model as a limited demonstration baseline.
The latter would support an operational MLOps demonstration, not establish full
three-class model acceptance. No response or stable approval is inferred from
elapsed time. Quality comparison, actual rollback and post-rollback monitoring
remain unverified pending the required evidence and approval.

## Full three-class requirement retained; correction schema 2

The user explicitly selected "Keep the full three-class quality requirement".
The limited-demonstration route is not approved. Continue the annotation audit,
then isolated retraining and same-label comparative evaluation before stable
approval and real rollback validation. Full model acceptance is not implied by
the working operational pipeline or passing automated tests.

Systematic visual review exposed a missing uncovered-head label in
`hard_hat_workers1003`. Schema 2 therefore records explicit helmet, head and person
box lists; it preserves correct source boxes while allowing reviewed corrections
to any approved class. The exporter retains source identities and creates only
derived outputs. It no longer copies redundant source XML files into the export.
Three new tests verify head correction, all-class/schema validation, and policy
hash consistency after Windows Git newline conversion. Policy text is hashed
with normalized line endings; source image/XML bytes remain exact. All 67 local
tests passed in 36.103 seconds.

Mirror-boundary diagnosis also exposed four earlier reviewed boxes that crossed
copies or used an inaccurate boundary. Those decisions were reopened and their
history retained, with no export or training having occurred. The review now
contains four reviewed images, 24 requiring adjudication and 3,972 pending.
The schema-1 snapshot, source/policy hashes and review journal remain available
under `reports/person-review-v1/`. The four accepted records were visually
rechecked under schema 2; this remains assistant review, not independent expert
validation or a completed annotation audit.

Person-box suggestions from the already available pretrained checkpoint remain
in a separate report file with their model/settings/image identities. They can
reduce drawing work but do not set reviewed status or enter training labels.
No new weights, dataset, service or external annotation platform was introduced.

The suggestion pass completed for all 4,000 training/validation images without
changing review status. A local DVC snapshot of six audit files records the
current incomplete review, including its journal and the suggestion generator:
`data/annotation-review.dvc`, MD5 `917cc6afc440a0f40fe31f6dd38b9b96.dir`.
This snapshot is not a corrected training dataset and no remote upload occurred.

## Continued annotation adjudication: 14 reviewed images

Enlarged raw images and corrected all-class overlays resolved nine earlier
ambiguous records and one newly inspected record. Person boxes now separate
copies at the visually checked mirror boundaries. Several helmet boxes were
tightened and omitted cropped helmets added; helmet-only fragments were not
automatically treated as people. Each decision retains its prior record in the
review journal. Seven other newly inspected crowded or fragmented scenes remain
unresolved, with specific instance/class questions recorded rather than guessed.

The review contains 14 reviewed, 22 requiring adjudication and 3,964 pending
images. These are assistant visual reviews, not independent expert validation.
The source/policy/split/coordinate checks pass, and an actual attempted export
still refuses the incomplete review without creating a destination. Raw dataset
metadata and the baseline checkpoint match their previous hashes. Source code
did not change; the prior 67-test local result is retained, not represented as
a new test or CI run.

The six-file local DVC review snapshot is now
`8dee5ca31b8114ecd6c0ece67eb379f9.dir`. The earlier snapshot remains in local
DVC history. No remote data transfer, corrected dataset export, training, stable
promotion, final-test prediction or actual rollback occurred. Full three-class
quality and the remaining acceptance evidence are still required. Detailed
checks are recorded in `reports/annotation-adjudication-validation.json`.

## Local script-driven annotation workflow

The user requested an alternative to image-by-image review through chat to
reduce time and token usage, then authorized implementing the local script
workflow. `scripts.annotation_triage` reuses the existing suggestion file, checks
all 4,000 source and suggestion identities, and ranks heuristic warnings without
accepting any label. Its localhost editor supports drawing/editing all three
classes, adding individual person suggestions, and saving explicit decisions
with before/after journal entries. It uses the Python standard library and the
existing image/validation dependencies; no hosted labeling service, new model,
external package, data transfer or GPU run was introduced.

The scan found flags in 3,996 images; 3,486 have strong person suggestions with
no matching current person label, and 3,906 have head/helmet boxes without person
coverage. These are triage signals, not confirmed errors or measured accuracy.
The large counts mean there is no defensible automatic shortcut to a tiny
review subset. Reviewed counts remain 14/4,000, with 22 recorded unresolved and
3,964 pending. The frozen policy and complete-review export gate are unchanged.
The user has not approved replacing explicit review with sampled acceptance.

Shared record validation now serves both the exporter and editor. Seven tests
exercise the new data/provenance and local HTTP paths, including rejected stale,
invalid, held-out and unattested writes. All 74 local tests passed. Browser
verification confirmed queue loading and the missing-attribution save guard;
no production annotation was accepted during UI testing. The latest CI remains
the earlier 58-test run. The review snapshot is unchanged because this step
changes tooling, not annotation decisions.


## Annotation batch 006: 18 reviewed images

On 2026-10-17 the user selected continued Codex visual review in batches.
Eight pending training images were inspected against enlarged raw images.
Corrected all-class overlays were checked before accepting workers1031,
workers1035, workers1036 and workers1040 (all IDs have the `hard_hat_` prefix).
The corrections separate visible mirrored copies, add missing person boxes,
and refine helmeted-head extents. Workers1032, workers1033, workers1039 and
workers104 remain unresolved because cropped fragments, occlusion or missing
head labels prevent a confident complete annotation. Their supplied boxes
remain unaccepted drafts. No model proposal was automatically accepted.

The working review now contains 18 reviewed, 26 requiring adjudication and
3,956 pending images. The shared write lock protected the batch, and the
journal retains every previous and resulting record with the batch method.
All 4,000 source image/XML identities, split and policy hashes, and box
constraints passed validation. A real incomplete-export attempt was rejected
before creating a candidate directory. Raw metadata and baseline checkpoint
hashes remain unchanged. Evidence is in
`reports/annotation-batch006-validation.json`; the triage scan was refreshed.
The prior 74-test result remains the latest local software test run; tests
were not rerun for this annotation/documentation-only change.

The six-file local DVC review snapshot is
`829844cc7cef77e70a5e223d684903cc.dir` (10,055,946 bytes). It remains incomplete
review evidence, not training data. No remote dataset transfer, training,
model promotion or change to the complete-review requirement occurred.


## User-approved incremental annotation review (2026-10-17)

The user approved replacing bulk chat review of all 4,000 train/validation
images with a smaller corrected training subset and a fixed validation sample.
Policy version 3 supersedes the earlier full-review export requirement. Class
semantics, source immutability, original split membership, the three-class
quality target and separate final-test/monitoring integrity requirements remain.
No model-quality approval is implied by this scope change.

The first round freezes 200 training and 100 validation IDs in
`data/annotation-cohort.json`. Source helmet/head/person presence and crowding
(six or more supplied heads/helmets) define strata. Proportional allocation with
at least one per stratum and SHA-256 ordering using seed 42 select the IDs;
no predictions, correction status or apparent ease influence membership.
The manifest records population and sample counts. Source-label strata are
approximate, and rare strata are overrepresented by the minimum allocation;
unweighted subset metrics must not be claimed as full-validation accuracy.
The validation sample stays fixed between rounds. Insufficient class coverage
requires an explicitly recorded prediction-independent expansion and a fresh
comparison of all models, never replacing difficult selected images.

The 4,000 existing records are preserved exactly, including all 18 accepted
reviews and 26 unresolved records. Three accepted and three unresolved records
fall in the selected round; 294 selected records remain pending. The migration
journal records the user's decision and old/new policy fingerprints without
pretending to perform a new visual review. The prior working files are retained
under `reports/person-review-v1/before-incremental-policy`, and the prior DVC
snapshot `829844cc7cef77e70a5e223d684903cc.dir` remains in local cache.

Export now requires every selected image to be reviewed without issues and
writes only the selected images/labels, plus the exact selection and cohort
provenance. Missing, changed, duplicate or held-out cohort membership is
rejected. All original source identities still receive structural validation.
The local editor displays the 300 selected records and can add confidence-filtered
person suggestions as editable drafts. Loading drafts never accepts labels.
No new inference pass, bulk chat image inspection or remote transfer was needed.

All 79 local tests passed in 65.456 seconds. Tests verify sampling, cohort
identity, blocked incomplete subset exports, exclusion of unselected images,
and scan/browser queue membership in addition to the existing safeguards.
The live queue contained exactly the frozen 300 IDs; JavaScript syntax passed.
Original raw metadata and the baseline checkpoint hash remained unchanged.
Evidence is in `reports/incremental-review-validation.json` and
`reports/incremental-review-tests.log`. No new CI run was dispatched.

The new seven-file local DVC review snapshot is
`2d4cc7af68bbed2e21130529fbae63ac.dir`, including a copy of the cohort manifest.
It is still incomplete annotation evidence, not a training dataset. Review the
selected remainder locally, then export and integrate the corrected cohort into
an isolated experiment. The baseline training entry point still stages original
processed data; it must not be used to represent a corrected-data experiment.
Compare the baseline and candidate on the same corrected validation subset,
expand training only when development evidence warrants it, and keep final-test
selection, stable approval and actual rollback validation as later gates.
