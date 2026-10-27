# Reproducible MLOps Pipeline for Visual Analytics, Data Versioning and Automated Rollback

A local construction-safety demonstration: image/camera detection with YOLOv8n,
DVC data versions, MLflow experiments and registry, Docker/FastAPI serving,
Prometheus/Grafana monitoring, Evidently drift evidence and verified automatic rollback.

**College demonstration, not production quality acceptance.** Person detection and
source-label completeness remain limitations. The original dataset URL/version is
unverified. No main-registry stable model is approved. Controlled demonstrations
use a separate registry and deliberate faults; no fake detections or metrics.

## Run the existing laptop installation

Open this folder in VS Code, then run in PowerShell:

```powershell
Set-Location -LiteralPath 'F:\major project implementation'
.\scripts\start_demo.ps1
```

Open [the main console](http://127.0.0.1:8767/). It includes detection, named test
images, model/data comparisons and automatic rollback controls. The start script
uses the saved API container; it does not train, download data or upload to storage.

- [Faculty demonstration guide](docs/faculty-demo.md): exact sequence and answers.
- [Current audit and remaining work](docs/final-audit.md): verified evidence and gaps.
- [Additional dataset and experiments](docs/additional-data.md): provenance and splits.
- [Annotation policy](docs/annotation-policy.md): what label review does and does not prove.
- [Decision history](docs/decisions.md): earlier approvals and phase records.

## Current data and models

| Dataset / protocol | Train | Validation | Test | Other |
|---|---:|---:|---:|---|
| Original 5,000 images | 3,500 | 500 | 500 | 500 monitoring images |
| Expanded experiment v1 | 6,020 | 196 | 77 | Preserved historical export |
| Fresh grouped 70/15/15 protocol | 5,385 | 1,154 | 1,154 | Monitoring replays belong to validation |

The fresh protocol contains all 5,000 original images and 2,693 retained additional
images. Related/augmented groups stay together. Earlier protected evaluation images
remain outside training. Some old training images were reassigned to the new test
set, so old checkpoints cannot claim held-out results on that new test set.

| Registered source model | Training | Shared 196-image AP50-95 |
|---|---|---:|
| 1 | Original images/labels, 30 epochs | 32.44% |
| 2 | Original images, revised labels, 30 epochs | 32.63% |
| 3 | Original images, proposed person labels, 30 epochs | 33.90% |
| 4 | Expanded images, 60 epochs | 44.91% |
| 5 | Fresh grouped protocol, 30 epochs | 42.25% |

The completed 27 October comparison uses explicit settings: 416 pixels, confidence
0.001, IoU 0.7, max detections 300, `rect=False`, no test-time augmentation.
Do not mix these with older evaluation scores. Early gains are small and do not
establish statistical significance. Model 4 is the normal experimental detector.
Model 5 completed all 30 epochs and registered on 27 October. It scored below
model 4 on the same validation cohort, including person AP50–95 (25.67% versus
30.16%), so model 4 remains the live detector. The corrected split establishes a
separate evaluation protocol; it does not guarantee a better model.
Model 5's separate one-time test on 1,154 images completed at 50.25% AP50–95.
That test is not the shared validation cohort and is not a model 4 comparison.

## Lifecycle and source map

1. Inspect source data, validate XML/boxes/classes, preserve original bytes.
2. Version snapshots with DVC; use the approved private **Google Drive** remote.
3. Split related groups and create derived preprocessing outputs.
4. Train reproducibly; record source hashes, settings, inputs, validation and losses.
5. Register completed, verified artifacts in local MLflow. Approval is separate.
6. Test and build through the existing GitHub Actions workflow.
7. Stage verified registry bytes and deploy one Docker/FastAPI service.
8. Measure requests, errors, latency, confidence and detected classes in Prometheus.
9. Compute PSI/KS drift evidence; missing or stale evidence is unavailable.
10. Combine fresh operational or labeled-quality evidence with configured policy.
11. Restore the captured approved predecessor and verify recovery; bound retries.

| Location | Purpose |
|---|---|
| `src/data`, `scripts/prepare_additional_data.py` | Validation, split, preprocessing and restored-byte checks |
| `src/training`, `src/tracking` | Training, evaluation, lineage and MLflow registry |
| `src/serving` | API, presentation UI, verified deployment |
| `src/monitoring`, `scripts/demo_rollback.py` | Drift, health decisions, audited recovery and controlled demonstrations |
| `configs`, `params.yaml` | Approved policies and reproducible settings |
| `tests`, `.github/workflows` | Local regression tests and CI |
| `dataset`, `archive (2)` | Preserved original local inputs; not Git blobs |
| `data/*.dvc`, `dataset.dvc`, `demo-images.dvc` | Versioned data identities |
| `artifacts` | Immutable training evidence, checkpoints, local registry and deployment receipts |
| `reports` | Runtime audits, restore evidence, test logs and screenshots |
| `demo-images` | Four named faculty examples with source/license records |

## Local development and verification

Python 3.11.9, the existing `.venv`, Docker Desktop Linux containers/WSL2 and
NVIDIA CUDA training are used on this laptop. API inference is CPU-based.
`requirements-lock.txt` pins the local environment; `requirements-serving.txt`
pins the Linux CPU image. Do not replace either with unpinned packages.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts/check_monitoring.py
$dockerExe = Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe'
& $dockerExe build -t visual-analytics-api:faculty-20261026 .
```

The current image is deployed through `python -m src.serving.deploy` with explicit
experimental opt-in. Stable/previous-stable aliases are never invented to pass a
check. Normal production rollback remains unavailable until real quality approval.

The comparison and final test below already completed. These are the recorded
commands for reproducibility, not startup steps; existing output is deliberately
not overwritten:

```powershell
.\.venv\Scripts\python.exe -m src.training.evaluate --compare-run artifacts/training/expanded-css-v1-60e --extra-run artifacts/training/protocol-v2-30e --output reports/final-audit-20261026/comparison-with-protocol
```

Run this only after the fresh experiment has completed and registered. Its final
test is a separate one-time operation, after checkpoint selection is frozen:

```powershell
.\.venv\Scripts\python.exe -m src.training.evaluate --test-run artifacts/training/protocol-v2-30e
```

The test command rejects old protocols, training overlap, related-group leakage,
changed DVC bytes and existing test output. Never tune using final-test scores.
The new candidate is not deployed merely because its version is newer.

## Storage and preservation

Google Drive supersedes original GCS and the temporary B2 remote by user approval.
Legacy B2 recovery tools remain for historical reproducibility and tests; they are
not the active default. Credentials stay in secure local configuration and are
excluded from Git. No keys belong in chat, screenshots or committed files.
Latest corrected data and faculty image snapshots are locally DVC-tracked;
remote synchronization of those new snapshots is not yet claimed.

The original data, all completed model checkpoints, training evidence and registry
history are preserved. Before the current cleanup, source/configuration/docs were
saved in `reports/final-audit-20261026/before-changes.zip`, with hashes and a Git diff.
Old presentation guides were consolidated into `docs/faculty-demo.md` and kept in
the recoverable local cleanup archive. Historical scores remain in their original
reports and should be cited with their specific evaluation protocol.
