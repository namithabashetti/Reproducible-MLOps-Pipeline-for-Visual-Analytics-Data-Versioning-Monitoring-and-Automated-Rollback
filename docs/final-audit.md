# Final implementation audit — 22 October 2026

## Readiness and scope

**GO for the local college demonstration; NO-GO for production or full three-class model acceptance.**
Final checks: 95/95 local tests, seven live dashboard panels and two real quality rollback runs passed.
The presentation now includes actual image uploads and annotated predictions, a controlled
quality regression, real automatic predecessor redeployment, recovery predictions and monitoring.
Two quality rollback runs passed. The second was started from the browser on the hardened
serving image; the original experimental API and main registry were restored afterward.

The user approved Google Drive as the DVC remote, superseding GCS in the PRD and B2 in the
emergency checklist. The existing raw dataset remains unchanged. No new dataset, framework,
cloud service, paid transfer or production model promotion was introduced. The live console
is a later user-approved extension using existing FastAPI and plain HTML/JavaScript.

The original PRD, approval attachments, repository, tests, dataset, registry and local runtime
were inspected. No separate TRD or implementation-plan document was found. The original
source checkpoint is Git `9a2d309`; pre-audit files, hashes and diff are retained under
`reports/final-audit-20261022/source-before/`, `source-before-sha256.json`, and `diff-before.patch`.
This report supersedes earlier phase completion summaries; historical reports remain evidence
of their recorded runs, not proof that a service is currently running.

Start with [the exact Windows presentation guide](vscode-demo.md).

## Evidence index

All paths below are relative to the repository. `E` means `reports/final-audit-20261022/`.
Large generated evidence remains local and Git-ignored; the source, guides and test changes
are versioned. A source checkout alone does not contain the dataset, registry or trained weights.

| Reference | Evidence and scope |
| --- | --- |
| D | `E/dataset.json`, `E/preprocessing.json`: all 5,000 images and annotations; original hashes and derived consistency |
| V | `dataset.dvc`, `dvc.lock`, `reports/gdrive-restore.json`: exact raw identity and previously verified Drive restore; no new remote transfer |
| M | `E/model-evaluation-v2/comparison.json`, per-model plots, `E/repeat-and-fp16.json`: fixed 100-image validation comparison |
| T | `E/training-pilot/results.json`, `E/pilot-mlflow.json`: two real diagnostic GPU training runs |
| R1 | `reports/live-demo-0c5a105ec84e/result.json`: first real quality recovery |
| R2 | `reports/live-demo-f857ca1442d4/result.json`: repeated real quality recovery on the hardened image |
| Q | `E/final-tests-v2.log`: final regression run; `E/final-tests.log` preserves the earlier failure |
| O | `E/live-integration-v2/validation.json`: final live integration; original dashboard failure retained in `E/live-integration/` |
| S | `E/dependency-advisories.json`, `E/source-security-inventory.json`, `E/security-runtime-final.json`: bounded security inspection |
| G | `E/demo-gallery/manifest.json`, `index.html`, eight input/prediction folders, `contact-sheet.png` |
| B | `E/browser-upload.png`, `browser-controlled-failure.png`, `browser-recovered.png`: real browser demonstration |

## Requirements traceability

“Complete” below means the stated behavior was verified within the local demonstration scope.
It does not imply production quality approval. The following table covers all original FRs;
AC and NFR mappings follow. Initial status refers to the state before this audit's corrections.

| ID | Original requirement | Expected behavior | Source/modules | Verification | Initial status | Evidence/result | Corrective action | Final status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FR-01 | Obtain Kaggle data | Identifiable source/version | `src/data/validate.py` | Inspect original metadata | Partial | D: existing data valid; Kaggle URL/version unknown | Obtain trustworthy upstream provenance | Partial |
| FR-02 | DVC versions | Track exact dataset bytes | `dataset.dvc`, `dvc.yaml` | Hash/status checks | Complete | V: tracked raw snapshot | None | Complete |
| FR-03 | GCS storage | Store/retrieve versioned bytes | `.dvc/config`, `scripts/gdrive_session.py` | Prior full restore receipt | Complete, amended | V: approved Google Drive replacement | Preserve amendment; no new transfer | Complete, amended |
| FR-04 | Reproducibility | Recover run's data version | DVC + `src/tracking/mlflow_utils.py` | Hash/run lineage verification | Complete | D, V, M; exact artifacts | None | Complete locally |
| FR-05 | OpenCV preprocessing | Training images and aligned labels | `src/data/preprocess.py` | Fresh in-memory comparison | Complete | D: all pixels and 25,502 boxes match | None | Complete |
| FR-06 | YOLO/PyTorch training | Recorded completed training | `src/training/train.py` | Existing full runs + new pilots | Complete | T, registry; finite training outputs | No unsupported quality claim | Complete |
| FR-07 | Precision/Recall/mAP | Comparable evaluation | `src/training/evaluate.py` | Same fixed cohort/protocol | Partial | M: three versions and repeated baseline | Standardize comparison; report person weakness | Complete metrics; quality Partial |
| FR-08 | MLflow experiments | Parameters, metrics, artifacts | `src/tracking/mlflow_utils.py` | Retrieved hashes, pilot logging | Complete | M, T, `E/registry.json` | None | Complete locally |
| FR-09 | Registry stable candidate | Versioning plus quality approval | MLflow, `src/serving/deploy.py` | Registry/approval checks | Partial | Three experimental versions; main stable aliases empty | Correct labels/retrain before main promotion | Partial |
| FR-10 | GitHub Actions CI/CD | Approved build/test/deploy | `.github/workflows/ci.yml` | Local equivalents + saved prior runs | Complete historically | Prior remote success; current local checks | Added monitoring regression to workflow; current remote run not claimed | Partial for latest revision |
| FR-11 | Docker packaging | Reproducible serving image | `Dockerfile`, deployment module | Build and actual deployments | Complete | R2, build log, immutable image ID | Rebuild with decoder restriction | Complete |
| FR-12 | FastAPI serving | Validated real predictions | `src/serving/api.py`, `console.py` | API tests, real browser upload | Complete API; Missing live UI | B, R2; original coordinates/identity | Added bounded local presentation console | Complete uploads and camera transport |
| FR-13 | Prometheus | Actual operational metrics | API instrumentation, Compose | Fresh scrape + request counts | Complete | O and R2 recovery | None | Complete locally |
| FR-14 | Grafana | Usable live panels | `configs/monitoring/grafana/dashboard.json` | Promtool + real panel queries | Broken under idle series | Prior NaN found; O retest | Filter zero-traffic series before ratios/quantiles | Complete |
| FR-15 | Evidently drift | Reference/current analysis | `src/monitoring/drift.py` | Real replay and report export | Complete | O; controlled monitoring pools | None | Complete locally |
| FR-16 | PSI and KS | Configured statistics | `src/monitoring/drift.py` | Known-shift and real replay checks | Complete | Q, O, decision fixtures | None | Complete |
| FR-17 | Combined health | Fresh compatible signals | `src/monitoring/health.py` | Threshold/freshness matrix | Complete logic | Q, R2; absent evidence remains unknown | Added actual serving-quality measurement for demo | Complete logic; main quality input Partial |
| FR-18 | No drift-only rollback | Unhealthy policy condition required | Health + rollback modules | Synthetic matrix + regression tests | Complete | Q, `E/decision-scenarios.json` | Preserve thresholds | Complete |
| FR-19 | Previous stable selection | Recorded MLflow predecessor | `src/monitoring/rollback.py` | Actual isolated registry switch | Partial | R1/R2 choose captured predecessor; main has none | Main needs legitimate quality approval | Complete demo; main Blocked |
| FR-20 | Normal redeployment | Shared Docker/FastAPI path | `src/serving/deploy.py` | Actual replacement and probes | Complete outage demo | R1/R2; immutable image and artifact hashes | Added quality-fault scenario | Complete demo |
| FR-21 | Monitoring after recovery | Fresh restored identity | Recovery checks, Prometheus | Fresh post-switch samples | Complete outage demo | R1/R2: monitoring resumed | Repeat quality recovery | Complete demo |

### Acceptance criteria and non-functional requirements

The original AC text is preserved in `E/prd.txt`. “Main blocked” refers to legitimate stable
quality approval, not the isolated demonstration mechanism. NFR identifiers here are report
labels; the PRD names these attributes without numeric IDs.

| ID | Original requirement / expected behavior | Source | Verification | Initial status | Evidence | Required correction | Final status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AC-01 | Kaggle version tracked | FR-01/02 | Provenance + DVC | Partial | D,V | Recover upstream URL/version | Partial |
| AC-02 | Store/retrieve through GCS | FR-03 | Approved Drive restore | Complete, amended | V | None | Complete, amended |
| AC-03 | OpenCV training inputs | FR-05 | Full consistency check | Complete | D | None | Complete |
| AC-04 | YOLO/PyTorch training | FR-06 | Completed runs/pilots | Complete | T | None | Complete |
| AC-05 | Precision, recall, mAP | FR-07 | Fixed comparison | Partial | M | Protocol alignment performed | Complete metrics |
| AC-06 | MLflow experiments/artifacts | FR-08 | Actual records/hashes | Complete | M,T | None | Complete |
| AC-07 | Versioned selected model | FR-09 | Registry retrieval | Complete | `E/registry.json` | No promotion without approval | Complete registration |
| AC-08 | GitHub workflow executes | FR-10 | Prior remote receipt/local checks | Complete historically | Prior phase CI reports | Rerun current commit remotely | Partial latest revision |
| AC-09 | Docker/FastAPI serves | FR-11/12 | Browser + API | Complete | B,R2 | Hardened decoder | Complete |
| AC-10 | Prometheus visible in Grafana | FR-13/14 | Live panel checks | Broken idle series | O | Query regression fixed | Complete |
| AC-11 | Evidently drift analysis | FR-15 | Actual report | Complete | O | None | Complete local replay |
| AC-12 | PSI/KS available | FR-16 | Real/synthetic calculations | Complete | O,Q | None | Complete |
| AC-13 | Health classification | FR-17/18 | Threshold/freshness matrix | Complete logic | Q,R2 | Main quality evidence needed | Complete logic; Partial main inputs |
| AC-14 | Previous stable redeployed | FR-19/20 | Two actual quality failures | Partial | R1,R2 | Main predecessor quality approval | Complete demo; main Blocked |
| AC-15 | Monitoring resumes | FR-21 | Fresh recovered scrape | Complete outage demo | R1,R2 | None for demo | Complete demo |
| NFR-Reproducibility | Trace training to version/run | DVC/training/tracking | Hash lineage | Complete | D,V,M,T | Upstream identity remains separate gap | Complete local lineage |
| NFR-Traceability | Identify all relevant versions | DVC/MLflow | Artifact retrieval | Complete | V,M,R2 | None | Complete |
| NFR-Consistency | Consistent serving environment | Docker/deploy | Pinned image and restart | Complete | R2 | None | Complete locally |
| NFR-Observability | Measurable visible operations | Prometheus/Grafana | Panel queries | Broken idle series | O | Corrected queries | Complete |
| NFR-Recoverability | Restore previous stable | Rollback/deploy | Actual switch/failure tests | Partial | R1,R2,Q | Main stable approval | Complete demo; main Blocked |
| NFR-Repeatability | Repeat defined training/deploy | Training/deploy scripts | Pilot + repeated rollback | Complete locally | T,R1,R2 | Resume CLI not implemented | Complete process; resume Partial |
| NFR-Maintainability | Separate lifecycle components | `src/`, scripts/tests | Source/import inventory | Complete | `E/source-security-inventory.json` | No new framework or manager layer | Complete local scope |
| NFR-Decision clarity | Explain recorded signals | Health/watch receipts | Reasons/identity assertions | Complete | Q,R2 | None | Complete |

## Defects corrected and implementation changes

1. **Idle Grafana series produced NaN.** Error-rate division and latency quantiles now exclude
   series without traffic. Missing observations are not fabricated as healthy zero values.
   `scripts/check_monitoring.py` tests active and idle series with real promtool, and CI invokes it.
2. **No interactive detection/quality demonstration.** Added a loopback console, bounded
   upload proxy, camera controls and real quality audit. It draws boxes on the exact submitted
   frame and displays deployment identity. Empty frames request an audit only during the
   designated demo; they do not independently authorize rollback.
3. **Unnecessary image decoders reachable.** Pillow is explicitly restricted to PNG/JPEG
   in the API. Tests reject GIF/TIFF/BMP even if the caller advertises an allowed content type.
4. **Windows annotation editor could reset unauthorized POST connections.** The handler now
   consumes a size-limited request body with a three-second timeout before replying. It still
   checks origin/host/token before parsing JSON or saving. The existing HTTP regression
   reproduced the issue; the targeted twelve-test rerun passed after the fix.

The initial 95-test run exposed that connection reset and is preserved. The final rerun
passed **all 95 tests in 129.807 seconds** (`final-tests-v2.log`). The pre-fix failure
was not deleted or relabeled as a pass. The earlier baseline had 94 passing tests.

## Dataset and cleanup

The existing dataset contains **5,000 PNGs, 5,000 XMLs, 25,502 boxes**:
helmet 18,966; head 5,785; person 751. Only 158 images contain person annotations, despite
people being visible in many more images. Structural validity is not semantic completeness.
Some original PNGs already have mirrored/duplicated borders; the gallery records this fact.

Raw content SHA-256: `15b3504dac4555eafbfcfb752ae642a323da03217311c4685f36acaa2c21d401`.
Raw DVC identity: `cd17bcaf3c75b4bf1ba13f802ecc2055.dir`.
All 5,000 preprocessed images and 25,502 transformed boxes match fresh in-memory conversion.
Raw metadata and content were preserved. Fixed splits remain 3,500 train / 500 validation /
500 final test / 250 monitoring reference / 250 monitoring current. The 135 related-image
groups do not cross splits; similarity screening remains heuristic, not proof of no shared scenes.
The audit did not select models on the final test pool or modify review statuses.

`E/file-inventory.jsonl` classifies **185,361 files**, each with path, size, tracked state and
retention reason. `E/cleanup-summary.json` records **zero removed and zero archived**.
Source, raw data, annotation history, registry, useful checkpoints and receipts were retained.
Virtual environments, DVC cache and generated reports are intentionally ignored, useful local
state; age alone did not establish disposability. No dependency was removed without proof it
was unused. The snapshot and inventory precede newly generated audit outputs; new outputs
are retained as audit evidence under the same ignored reports directory.

## Model selection, metrics and resource experiments

The same frozen 100-image validation cohort and labels were used for all three versions:
416 pixels, `rect=False`, confidence .001, NMS IoU .7, batch 4, workers 0, FP32, CUDA.
These standardized settings differ from older evaluation defaults; do not combine the old
numbers with this table. Ultralytics computes AP using confidence-ranked detections and IoU
thresholds .50:.05:.95. Precision/recall refer to its selected operating point. The F1 below
is the harmonic mean of reported macro precision/recall, not mean per-class F1.

| Version | Precision | Recall | F1 as defined | AP50 | AP50–95 | GPU inference ms/image | Model bytes | Peak allocated GPU MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 baseline | .95991 | .57552 | .71960 | .65048 | **.41826** | 2.461 | 6,218,275 | 172.54 |
| 2 full training | .95588 | .58202 | .72351 | .64380 | .41358 | 1.752 | 6,218,339 | 183.99 |
| 3 assisted labels | .61715 | .66880 | .64194 | .63682 | .40533 | 1.713 | 6,218,339 | 195.71 |

Inverse inference-only rates are approximately 406 / 571 / 584 images/s, respectively.
They exclude preprocessing, postprocessing and HTTP overhead, are short sequential GPU
measurements with warm-up/order effects, and do not characterize CPU serving throughput.
No speed-superiority claim is justified. The FP32 baseline repeat reproduced AP exactly;
FP16 gave AP50–95 .41759 and 2.757 ms/image versus repeat FP32 2.504 ms/image. FP16 was
not adopted because no inference speed improvement was established.

| Class | v1 precision | v1 recall | v1 F1 | v1 AP50 | v1 AP50–95 | v3 AP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| helmet | .93054 | .92120 | .92585 | .96767 | .66733 | .65123 |
| head | .94920 | .80536 | .87138 | .90152 | .55238 | .54965 |
| person | 1.00000* | .00000 | .00000 | .08226 | .03506 | .01512 |

*The framework's precision value at an operating point with zero person recall does not
indicate good person detection. Only five cohort images have person labels (30 instances).
Missing labels further limit all quality conclusions. Full per-class results, confusion matrices
and prediction plots are retained in M. Gallery examples visibly show the person-box omissions.
Version 1 remains the best measured local demonstration checkpoint, **experimental**.
Version 3 used unverified machine-generated training person proposals; they are not ground truth.

Hardware: Python 3.11.9, i7-12650H (10 cores/16 threads), about 16 GB RAM,
RTX 3050 A Laptop GPU with 4,094 MiB VRAM; Docker Engine 29.8.2.
Existing training configuration: YOLOv8n, 30 epochs, 416 pixels, AdamW lr .001,
weight decay .0005, seed 42, deterministic, batch 4, workers 0, no image cache, AMP off.
Best/last checkpoints are retained. Patience equals the epoch budget, so useful early stopping
was not demonstrated; the project CLI does not implement checkpoint resume.

Two actual two-epoch pilots used the same 64 training / 32 validation smoke subset:
batch 4 took 7.629 s, peak 537.16 MiB, AP50–95 .05581; batch 8 took 4.291 s,
peak 911.49 MiB, AP50–95 .00211. This short timing is confounded by warm-up and fewer
optimizer updates; it does not prove an improved model. Batch 4 remains the default.
Pilots are logged in MLflow experiment `hardware-efficiency-pilots`, runs
`8e56f15242b149c4892a9aee14653a0d` and `f3fe7dcb91d84d21a95d823607bc88ff`.
No further full retraining was justified before label completeness is corrected. GPU utilization
time-series and a controlled worker/cache/AMP-training sweep were not completed.

### Model provenance

| Main version | MLflow run | Checkpoint SHA-256 |
| --- | --- | --- |
| 1 | `473308c0c5df40d2b1095f6a5779441c` | `5f725cb9f769a68ddb8f9518a979c7999b20e152d9e406e7df34f923df43fe4f` |
| 2 | `514b767c14e9487fb039e822857d8b2c` | `7d449a571b97f0e93601f8008a2adc79fcf6ad544fd4912466dafdebe4d9a3db` |
| 3 | `278cee189aeb46a4a86eb2e55543f0d6` | `ac5283d9471fba7d10b0bcb1f57c4d0d947ce9158f8be41abcaa5e1796f7a6bc` |

Main `stable` and `previous_stable` aliases remain unassigned. Demo aliases are restricted
to a copied registry and flagged as fixtures. Registration and demo stability are not quality approval.

## Actual rollback evidence and failure matrix

The controlled fault changes only the isolated candidate's recorded confidence to 1.0;
labels and the health policy remain unchanged. The watcher uses the normal staging/deployment
functions and the predecessor captured in the deployment receipt. It does not simply return an ID.
The same 250 frozen monitoring images/labels are measured through HTTP before, during and after.
Serving AP uses the deployed .25/1.0 thresholds, so these scores are distinct from offline AP.

| Run | Before | Faulty candidate | Recovered | Total sequence | Outcome |
| --- | ---: | ---: | ---: | ---: | --- |
| R1 | .42086015 | .00000000 | .42086015 | 212.98 s | Passed |
| R2 | .42086015 | .00000000 | .42086015 | 281.10 s | Passed |

Total sequence time includes startup, three 250-image audits, grace, recovery checks and restoring
the original API; it is not failure-to-recovery latency. R2 used immutable image
`sha256:08a59e671f6b1dfc1b318fce29dcfe31223db88eaab734ea3e260300bf1a3ddb`.
Each run recorded exactly one durable attempt; retry retained the recovered container.
Three readiness checks, twenty predictions and fresh monitoring passed. Main aliases were unchanged.

| Required scenario | Actual verification | Result and limitation |
| --- | --- | --- |
| Healthy, no drift | `test_health`, synthetic normal fixture | Healthy, observe |
| Drift, acceptable quality | Real PSI/KS on synthetic feature arrays + health fixture | Drift detected; healthy, observe. Not claimed as real image quality under domain shift |
| Drift + apparent quality drop without eligibility | 100-image synthetic audit below minimum 200 | Unknown, observe; no qualifying quality condition |
| Qualifying operational/quality threshold | Error/latency/readiness tests; R1/R2 quality fault | Correct policy reason; actual quality rollback passed |
| Candidate worse than stable | Same 250-image HTTP audit, drop .42086 > .05 | Automatic predecessor restoration; sequential canary comparison, not concurrent traffic routing |
| No previous stable | Missing-target and main-registry audit tests | Fail closed; no invented target |
| Multiple registry versions | Recorded numeric predecessor tests + three main versions | Receipt target selected, not newest version |
| Missing/unapproved/corrupt target | Registry/artifact/approval tests | Refused before switching; simulated negative cases |
| Recovered model unhealthy | Recovery and deployment failure tests | Never reports recovered; durable attempt prevents blind retry |
| Duplicate event | R1/R2 follow-up checks + durable-claim tests | One attempt, same recovered container |
| Restart/interrupted attempt | Stale deployment/restart/claim tests | Fail closed; no actual power-loss injection performed |
| Successful recovery predicts | R1/R2 twenty probes and 250-image recovery AP | Actual loaded checkpoint and predictions verified |

Q contains ten health tests and sixteen rollback tests. Negative cases use isolated/mocked
failure injection, not deliberate damage to the main deployment. R1/R2 are real Docker runs.
The watcher is run by the demo process; no Windows auto-start watcher service is installed.

## Monitoring, CI and security

The final integration run completed successfully with explicit acceptance gaps: **all seven
Grafana panel queries passed**, traffic windows contained exactly **110 and 250 successful
predictions**, and three client rejections were correctly excluded. Fresh Evidently evidence
was available with no drift detected. Calibrated p95 was 0.07917 seconds. Overall main-model
health remained **unknown (quality evidence unavailable)**; it was not falsely marked healthy.
Raw dataset metadata, deployment and registry aliases remained unchanged.
Do not generate extra predictions during its exact-count windows. Its acceptance section is
limited to that runner's scope; R1/R2 supply the separate new rollback evidence.
The monitoring regression uses the deployed Prometheus image and tests idle/active series.
The current workflow invokes it. Previous GitHub runs remain recorded; **no new remote CI
success for these edits is claimed**. Local compilation, dependency consistency, actionlint, JavaScript syntax, Git whitespace,
95 regression tests, real promtool query regression and the Docker build all passed.
`E/final-secret-scan.json` records the final bounded source scan with zero findings.

The API binds to loopback, runs as UID/GID 10001, has a read-only root/model mount, drops
all capabilities and sets no-new-privileges. Uploads enforce byte and decoded-pixel limits,
allowed image types and box validation. The console rejects cross-origin mutations and
requires its per-process token. Routes do not accept arbitrary file paths, URLs or shell commands.
Secrets remain in ignored local mechanisms; targeted tracked-file pattern checks found no
matches. This is not an exhaustive Git-history/transitive-dependency security certification.

**Unresolved dependency risk:** installed torch 2.2.2 and Pillow 10.2.0 have advisory records.
[PyTorch's critical advisory](https://github.com/pytorch/pytorch/security/advisories/GHSA-53q9-r3pm-6pq6)
includes this torch version and warns that even `weights_only=True` can execute code when
loading malicious checkpoints. Hash verification provides provenance, not a safe deserializer.
Only the existing trusted local checkpoints are used; arbitrary uploaded model loading is absent.
Pillow's decoder restriction reduces exposure but does not patch the dependency. A tested
torch/torchvision/CUDA and Pillow upgrade is still required before production use. The audit
did not blindly replace the locked training environment or invalidate its DVC dependencies.
PyPI advisory record counts include duplicate aliases and are not distinct vulnerability counts.

## Literature comparison

The PRD cites twelve papers. Publisher-deposited DOI/title metadata was checked in
`E/paper-metadata.json`; supplied full paper PDFs or runnable reference implementations were
not found. Dataset/protocol/accuracy numbers cannot be reconstructed from titles or the PRD's
claims. The following domains are PRD descriptions, not independent full-text validation.

| PRD paper | DOI | Described scope / comparability |
| --- | --- | --- |
| P1 | [10.1109/ICIIP68302.2025.11346334](https://doi.org/10.1109/ICIIP68302.2025.11346334) | Vehicle detection/R-CNN; different classes/data; no compatible benchmark verified |
| P2, primary base | [10.1109/ICTAI66417.2025.00076](https://doi.org/10.1109/ICTAI66417.2025.00076) | Hyperspectral classification; not this hard-hat detection protocol |
| P3 | [10.1109/REACS67479.2025.11413418](https://doi.org/10.1109/REACS67479.2025.11413418) | Drought prediction; different domain/metrics |
| P4 | [10.1109/ICOSEC67334.2025.11459598](https://doi.org/10.1109/ICOSEC67334.2025.11459598) | DRIFT-ACT; full protocol/reference not verified |
| P5 | [10.1109/BigData66926.2025.11401445](https://doi.org/10.1109/BigData66926.2025.11401445) | Feature drift; no compatible end-to-end benchmark verified |
| P6 | [10.1109/CAIN66642.2025.00018](https://doi.org/10.1109/CAIN66642.2025.00018) | Experiment-driven MLOps; conceptual comparison only |
| P7 | [10.1109/ICNP65844.2025.11192460](https://doi.org/10.1109/ICNP65844.2025.11192460) | Medical cloud/edge scope; different application |
| P8 | [10.1109/ISAC364032.2025.11156530](https://doi.org/10.1109/ISAC364032.2025.11156530) | Recommendation scope; different task |
| P9 | [10.1109/BigDataService65758.2025.00019](https://doi.org/10.1109/BigDataService65758.2025.00019) | AutoDrift/C-MAPSS; different data/metric |
| P10 | [10.1109/CSCloud66326.2025.00025](https://doi.org/10.1109/CSCloud66326.2025.00025) | MLDaaS; performance protocol unavailable |
| P11 | [10.1109/ETFA61755.2024.10710938](https://doi.org/10.1109/ETFA61755.2024.10710938) | FactoryML; no same-hardware recovery comparison |
| P12 | [10.1109/MS.2024.3441101](https://doi.org/10.1109/MS.2024.3441101) | Cyber-physical production systems; different application scope |

The [P6 authors' institutional abstract](https://research.vu.nl/en/publications/towards-continuous-experiment-driven-mlops/)
describes controllable, traceable and repeatable experimentation implemented in ExtremeXp.
This supports thematic relevance, not a claim that its system lacks our features.
Our measurable evidence covers local DVC/MLflow lineage, actual CPU API operation, PSI/KS,
policy decisions and repeated recovery. Accuracy, latency, recovery time and reliability have
**no controlled cross-paper comparison**. More components do not establish superiority.
The only controlled model comparison completed here is between our three existing versions.

## Demonstration assets and exact commands

Open `E/demo-gallery/index.html` for eight visually inspected examples chosen from actual
saved serving predictions. Each includes the raw original PNG/XML, submitted processed PNG,
YOLO source labels, prediction image, class/confidence/box JSON and hashes in the manifest.
All displayed predictions match available source labels at IoU >= .5; this selection is biased
toward successful examples and is not a generalization metric. Small targets, mixed helmet/head
scenes, multiple workers and occlusion are represented. Person omissions remain visible.

For normal / faulty candidate / post-recovery batches, R2 stores `batch-sha256.json` and
`baseline-predictions.json`, `candidate-predictions.json`, `recovered-predictions.json`.
The source is exactly `data/processed/{images,labels}/monitoring_current/`, 250 fixed names
from `data/splits.json`. No ground truth is edited. Drift-only and insufficient-evidence cases
are explicitly synthetic in `E/decision-scenarios.json`; they are not real camera observations.

Startup, upload, camera caveats and both rollback modes: [vscode-demo.md](vscode-demo.md).
All commands below run in PowerShell from `F:\major project implementation`; the existing
environment/data/artifacts must be present. Setup for a new machine additionally requires
Python 3.11.9, the locked dependencies, Docker, authorized DVC restore and MLflow artifacts.
Do not redownload the dataset on this laptop.

```powershell
# Environment checks and local regression
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m compileall -q src scripts tests
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe scripts/check_monitoring.py
.\.venv\Scripts\python.exe -m dvc status

# Read-only dataset validation; choose a new report name
.\.venv\Scripts\python.exe -m src.data.validate --report reports/presentation-dataset-validation.json

# Fresh complete live integration; no concurrent camera, inference or deployment
$run = 'reports/presentation-validation-' + (Get-Date -Format 'yyyyMMdd-HHmmss')
.\.venv\Scripts\python.exe scripts/validate_pipeline.py --output $run --reference artifacts/monitoring/phase9/reference.json

# Drift-only/insufficient-evidence decision fixtures, clearly synthetic
.\.venv\Scripts\python.exe reports/final-audit-20261022/decision_scenarios.py

# Hardware checks and optional training; not needed for the presentation
.\.venv\Scripts\python.exe -m src.training.train --check-only
.\.venv\Scripts\python.exe -m src.training.train --smoke --run-name presentation-smoke-new
# A new 30-epoch run, only if deliberately requested later:
# .\.venv\Scripts\python.exe -m src.training.train --run-name presentation-full-new
```

Offline comparison and pilot reproduction scripts are `E/evaluate_models.py` and
`E/training_pilot.py`; inspect their fixed output paths and choose fresh destinations before
rerunning. Saved plots and inputs are retained; do not overwrite experiment evidence.
No additional install or training is needed to present the current laptop deployment.
Camera capture was verified in the browser: 640x480 frames, real API predictions and a
successful stop (video ready state returned to 0). No camera frames were saved. Evidence:
`E/camera-verification.json`. This verifies capture/transport, not hard-hat detection accuracy
in a particular live scene.

## Remaining limitations and completion categories

| Category | Item / exact blocker | Impact and next action |
| --- | --- | --- |
| Verified complete | Actual uploads, annotated predictions, two real quality rollback runs, identity checks, recovery probes | Ready for controlled local demonstration |
| Improved with measured evidence | Idle-panel query regression; restricted decoders; Windows POST handling | Retain tests and hardened serving image |
| Implemented, not fully verified | Live camera hard-hat accuracy under real presentation conditions | Camera transport verified at 640x480 with actual API responses and successful stop; test detection with a real hard hat before presenting |
| Not improved / inconclusive | Full-train/assisted models and batch/FP16 experiments | No superior three-class model or speed win claimed; keep measured baseline |
| Blocked by missing evidence | Original Kaggle URL/version; complete person labels; quality-approved stable predecessor | Obtain provenance and audited labels, retrain/evaluate, then approve a main stable model |
| External / unverified | Current Google Drive availability, latest remote GitHub workflow, full paper protocols | Prior storage/CI receipts retained; rerun when needed; no new remote success claim |
| Not completed in this audit | Secure dependency migration, training-resume CLI, early-stopping benefit, worker/cache/AMP sweep, GPU utilization trace, real crash/power-loss injection | Not needed to demonstrate existing local flow; required as applicable before broader deployment |
| Production blocker | Vulnerable locked dependencies and unaccepted model quality | Upgrade and regression-test the environment; establish quality gates before network/public use |

No promise of fully autonomous natural camera-error detection is made. Natural quality failure
needs trustworthy labeled evidence and an approved predecessor. The college demonstration
shows the complete mechanism with an explicit controlled fault and isolated stability fixtures.
