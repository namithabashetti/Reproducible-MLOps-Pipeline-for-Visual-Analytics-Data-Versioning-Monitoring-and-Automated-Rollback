# Research quality upgrade — 28 October 2026

The user selected a stronger college/research model with measured quality.
The existing model 4 remains the live demonstration model while the new candidate
is prepared and evaluated. None of these steps grants production approval.
The user briefly deferred training, then explicitly resumed it on 28 October.
The 30-epoch corrected-label experiment is now authorized; record its actual
completion and comparison before claiming an accuracy improvement.

## Original release identity resolved

The local 5,000 PNGs and 5,000 XMLs match **Safety Helmet Detection, Kaggle,
andrewmvd/hard-hat-detection, version 1**, released 11 August 2020.
All 10,000 paths, file lengths and CRC32 values match the official versioned
archive's central directory; the total is 1,320,154,239 bytes. All 5,000 image
SHA256 values also match the pinned Voxel51 mirror revision
`d11e7d6b69ba1a22422db3230c0415d39a699d7c` and SHEL5K v4 publisher metadata.
No image payload was downloaded. CRC32 is an integrity check, not a cryptographic
proof; the independent image comparisons use SHA256.

`data/source-provenance.json` binds this evidence to the original local content
fingerprint. Validation fails if current bytes differ. Kaggle lists CC0; its
description credits Make ML. The historical download date/account and upstream
ownership of every image remain unknown. Those are distinct from the now-verified
public release identity. Older frozen reports retain their historical status.

- [Original publisher](https://www.kaggle.com/datasets/andrewmvd/hard-hat-detection)
- [Official metadata](https://www.kaggle.com/api/v1/datasets/view/andrewmvd/hard-hat-detection)
- [Pinned image mirror](https://huggingface.co/datasets/Voxel51/hard-hat-detection/tree/d11e7d6b69ba1a22422db3230c0415d39a699d7c)

## Published replacement labels

Gochoo, Munkhjargal (2022), *Safety Helmet detection with Extended Labels 5K
images (SHEL5K)*, Mendeley Data, version 4, DOI
[10.17632/9rcv8mm682.4](https://data.mendeley.com/datasets/9rcv8mm682/4), CC BY 4.0.
The associated [SHEL5K paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8950768/)
describes correcting the incomplete original dataset.

Only the 5,000 XML annotations are downloaded (19,154,616 bytes), with publisher
SHA256 and length checks. Original data and previous correction records remain
unchanged. These are **publisher annotations**, not model-generated pseudo-labels
or a claim that this project's reviewer inspected all images.

The candidate keeps three outputs: `helmet` -> helmet, `head` -> uncovered head,
and `person_with_helmet` / `person_no_helmet` / the few generic `person` records
-> person. `face` and `head_with_helmet` are omitted to avoid duplicate head
targets. The candidate's helmet box encloses the actual helmet; earlier source
boxes mixed helmets and helmeted heads. This is an explicit annotation-policy
change for this candidate, not a rewrite of prior experiments. Person boxes use
the publisher's visible-body convention. Four initial development-image spot
checks showed recovered person labels; artificial mirrored image borders still
need caution. Published labels and structural validation do not prove perfect
visual completeness. Additional CSS images retain their supplied labels.

## Frozen experiment, before training

- Preserve the grouped protocol-v2 membership: 5,385 training, 1,154 validation,
  and 1,154 test images. No test image moves into training. Preserve related groups.
- Replace only original-image labels; retain all additional training images.
  Created `data/expanded-shel5k-v1`, tracked by DVC as
  `ec398edfa361f9d9b4952019bb04411e.dir` (15,394 files, 1,917,518,119 bytes).
- Use fresh YOLOv8n pretrained initialization and the existing 416-pixel,
  batch-4 AdamW settings for one 30-epoch experiment. This matches model 5's
  budget and isolates the label change. More epochs are not a quality guarantee.
- Evaluate model 5 and the new model on exactly the same corrected validation
  images and settings. Report all three AP50, AP50–95, precision and recall values,
  class counts, and original/CSS cohorts separately.
- The research improvement target is at least +0.05 absolute person AP50–95
  versus model 5, no reduction in overall AP50–95, and no more than 0.02 absolute
  AP50–95 regression for either helmet or head. This is a predeclared experimental
  selection rule, not a scientifically validated production threshold.
- For the live model 4 comparison, use only the original shared 196-image
  validation cohort, relabeled consistently. Model 4 trained on some images in
  the larger protocol validation pool, so its full-pool scores would be biased.
  Require no person or overall AP50–95 regression, and at most 0.02 regression
  for helmet or head on this shared cohort, in addition to the model-5 target.
- The protocol test was already consumed by model 5. Do not tune on it or call
  another evaluation a fresh final acceptance test. This experiment reports
  development-set improvement; independent final acceptance remains separate.
- Promote no model automatically from training completion. Retain existing
  artifacts and rollback targets; update the live demo only after comparison.

## Local evidence and preparation

Evidence is in `reports/quality-upgrade-20261027/` (folder name retained from the
start of the investigation). `shel5k-audit.json` records actual downloaded counts,
mapping, source hashes and any blocking structural errors. The output manifest
records source attribution and remaining limitations. Run from the project root:

```powershell
.\.venv\Scripts\python.exe -m scripts.prepare_shel5k --fetch --evidence reports/quality-upgrade-20261027 --output data/expanded-shel5k-v1
```

Export refuses an existing destination, changed publisher bytes, unknown classes,
invalid/duplicate boxes, or split leakage. Do not mark existing pending reviews
as reviewed to make this command pass.

To measure the existing models against the corrected development labels:

```powershell
.\.venv\Scripts\python.exe -m src.training.evaluate --research-data data/expanded-shel5k-v1 --output reports/quality-upgrade-20261027/baselines
```

This stages validation copies to keep evaluator cache files out of DVC data. It
checks completed-run identities and rejects training/cohort overlap. Model 5 is
scored on the full validation set and its original/CSS subsets; model 4 is scored
only on the shared 196. After training, use a new output directory and
`--extra-run artifacts/training/shel5k-v1-30e` to include the candidate. CPU
evaluation settings are fixed across the comparison. Original test files are
never staged by this command.

When the user resumes training, connect AC power and run:

```powershell
Set-Location -LiteralPath 'F:\major project implementation'
.\.venv\Scripts\python.exe -m src.training.train --expanded-data data/expanded-shel5k-v1 --run-name shel5k-v1-30e --epochs 30
```

This creates a new run from pretrained YOLOv8n and preserves model 4. If the run
already exists after an interruption, inspect its evidence and use the existing
`--resume-run` path rather than overwriting it. After completion, compare using:

```powershell
.\.venv\Scripts\python.exe -m src.training.evaluate --research-data data/expanded-shel5k-v1 --extra-run artifacts/training/shel5k-v1-30e --output reports/quality-upgrade-20261027/candidate-comparison
```

The report records both predeclared selection checks. Passing them does not
automatically deploy a model or approve a production stable alias.

## Baseline results on the replacement labels
These scores were recorded before any new training. They use the new helmet-object convention and more complete person labels, so they must not be mixed with earlier published scores.
| Model | Evaluation cohort | Overall AP50-95 | Helmet AP50-95 | Head AP50-95 | Person AP50-95 |
|---|---|---:|---:|---:|---:|
| model 5 | combined (1154 images) | 19.52% | 5.26% | 37.88% | 15.43% |
| model 5 | original (1033 images) | 20.05% | 5.13% | 38.79% | 16.22% |
| model 5 | additional (121 images) | 31.64% | 26.45% | 22.97% | 45.50% |
| model 4 | shared196 (196 images) | 19.08% | 5.09% | 33.63% | 18.50% |

Evidence: `reports/quality-upgrade-20261027/baselines/comparison.json`. Original/CSS cohort scores are separate. The low scores under the corrected labels reinforce why label repair and new training are needed; no accuracy improvement is claimed yet.


## Completed five-model research evaluation — 29 October 2026

All five corrected-label jobs completed their requested 30/30/30/60/70 epochs and final evaluations, registered as versions 6–10. Model 5 achieved validation AP50–95 0.555811 and reused-test AP50–95 0.539095; test person AP50–95 was 0.59450. The baseline table above remains a historical pre-training measurement. These newer results use corrected label conventions and do not establish independent production acceptance. Full metrics, per-class results, run IDs and limitations are in [PROJECT_EXECUTION_REPORT.md](../PROJECT_EXECUTION_REPORT.md).
