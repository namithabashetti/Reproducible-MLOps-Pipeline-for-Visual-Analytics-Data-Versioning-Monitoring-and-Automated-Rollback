# Approved project decisions

These records describe the user's approved implementation choices. They do not
claim that the pipeline stages have been implemented or accepted.

## Storage amendment to the PRD

**Original PRD storage:** DVC + Google Cloud Storage.

**Approved implementation storage:** DVC + Backblaze B2.

**Reason:** The student project uses Backblaze B2 for object storage while retaining
DVC for dataset version control.

**Impact:** FR-03, AC-02, and the storage references in architecture, setup,
traceability, and demonstration evidence use B2. Preserve the original PRD and
describe this amendment explicitly instead of claiming an unchanged implementation.

Use DVC's S3-compatible remote mechanism and the B2 endpoint. There is no custom
storage wrapper, AWS account dependency, or GCS setup. The implementation path is:

```text
Existing Kaggle-attributed dataset -> local raw files -> DVC -> Backblaze B2
```

The approved bucket is `varshith-visual-analytics-dvc` in region `us-east-005`.
Its privacy and encryption are user-confirmed, not independently verified through
the provider API. The prefix `visual-analytics/dvc` keeps this project's DVC objects
together. The remote URL, endpoint, and region now live in `.dvc/config`.
Authenticated transfer requires the replacement key entered locally.

The previous application key was exposed. It must not be used by this project.
The user confirmed replacement-key creation and revocation of the exposed key
during Phase 2. Use the replacement only. Do not inspect or print secret values
as part of status checks.

The user delegated selection of a minimal budget: target USD 0/month, ceiling
USD 1/month. This is not a configured provider-side cap. No remote transfers occur
in Phase 1, and actual retained bytes and egress must be considered before upload.

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

Preserve `helmet=0`, `head=1`, and `person=2`. Keep all three classes and report
per-class results because person annotations are much less frequent. Scene/group
checks precede split creation. The final test set stays outside tuning; monitoring
reference and current images are disjoint. Any needed split-count adjustment must
be documented rather than silently breaking group separation.

## Local execution and dependencies

All training and services run on the Windows laptop. Use Python 3.11.9 in `.venv`,
without inheriting global packages. Existing compatible versions may be reused in
the isolated environment, but global packages are not upgraded or uninstalled.

The initial model is YOLOv8n. Ultralytics 8.3.229 and OpenCV 4.11.0.86 are the
approved starting versions. The global torch 2.2.2 installation is CPU-only. The
proposed GPU combination is torch 2.2.2 / torchvision 0.17.2 with CUDA 12.1 wheels;
it is not yet installed or verified in the project environment. GPU training is
gated on real compatibility and smoke tests. No CUDA compatibility is inferred
solely from the NVIDIA driver version.

Phase 2 installed DVC 3.67.1, dvc-s3 3.3.0, and Pillow 10.2.0 alongside PyYAML
6.0.3. `requirements-lock.txt` records this Windows/Python 3.11 environment.
MLflow 3.16.1 and Evidently 0.7.23 remain candidates for later phases. They have
not been installed or tested together. Add dependencies only in the relevant phase.

MLflow uses local SQLite metadata and local artifacts. SQLite is an embedded
metadata file, not a new database service. Compose is approved for the real local
services, with host ports bound to localhost. No frontend or unrelated API is in
scope. Local model artifacts and metadata require preservation for recovery.

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

## Delivery gates

The private GitHub repository already exists; use it and preserve remote changes.
The user approved Phase 1 foundation after Phase 0 review. Subsequent phases are
delivered incrementally with their own validation reports.

The self-hosted GitHub Actions runner needs a separate, concrete setup/security
review and approval before registration. Do not imply that GitHub-hosted jobs can
reach local services without an approved mechanism. No cloud deployment is planned.

Still deferred: authenticated DVC push/pull until the local credential prompt runs,
scene/group review, CUDA training verification,
later dependency resolution, runner approval, and measured threshold calibration.
These do not prevent the repository foundation from being completed.
