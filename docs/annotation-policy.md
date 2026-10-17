# Three-class annotation correction policy, version 3

This policy governs a separate derived correction set. The original annotation
guidelines are unknown. It defines the intended visible-person task without
claiming to recover the original annotator's intent.

- Preserve the original PNG/XML files, all three classes, and split membership.
- Review a frozen development cohort from the existing training and validation
  splits, including images with no supplied person labels. The first round uses
  200 training and 100 validation images. Select by source class presence and
  crowding, with proportional allocation (at least one per stratum) and seed 42.
  Selection must not use predictions, accuracy, review status or apparent ease.
  Preserve all 4,000 review records; nonselected records remain outside the
  candidate rather than being silently accepted. Membership is recorded in
  `data/annotation-cohort.json` and bound to the review by SHA-256.
- Annotate each clearly identifiable visible person, including partial people.
  Use a tight rectangle enclosing visible body parts, including the head and worn
  helmet. Do not extend the rectangle to an imagined hidden body or outside the
  image. Independently visible people receive separate boxes even when occluded.
- Reflections and artificial mirrored borders already exist in these source
  images. A separately visible reflected person can need a separate box. Do not
  merge two copies into one person. Mark uncertain fragments for adjudication;
  an unresolved image cannot count as reviewed or enter the corrected export.
- Initialize helmet/head boxes from the supplied annotations, then review all
  three classes. Preserve correct boxes and explicitly record any correction in
  the reason. Helmet annotations retain the source convention of enclosing the
  helmeted head; head denotes an uncovered head. Do not label a helmeted head as
  both helmet and head. Person boxes can contain these independently nested boxes.
  Uncertain class identity or boundaries require adjudication.
- Review proposed boxes against the actual image. Model outputs may prioritize
  inspection, but cannot automatically become reviewed ground truth. Record the
  reviewer, UTC review time, and reason, including for an unchanged/empty result.
- Coordinates are zero-origin pixel boundaries in the **raw** PNG: `[xmin, ymin,
  xmax, ymax]`, with positive width/height and boundaries within image dimensions.
  The export uses the existing letterbox transformation for YOLO labels.
- Preserve image hashes, XML hashes, split hash, and this policy's hash. Any input
  or policy change invalidates the review instead of silently carrying it forward.
- Every selected training and validation record must be reviewed with no open
  issues before exporting that cohort as a candidate. Unselected pending records
  do not block this export and are never included in its labels. A successful structural check
  cannot prove that a reviewer actually looked at an image or judged it correctly.

The final-test and monitoring images are absent from this review set. Freeze the
policy before a separate, prediction-independent integrity review of those pools.
Use final-test predictions only after selecting the model. Corrected training and
validation labels do not by themselves repair the labeled monitoring comparison.

Keep the 100 validation IDs fixed across development rounds and evaluate the
baseline and each candidate on the same corrected validation labels. This sample
is stratified using incomplete source labels, so its coverage is approximate;
report per-class instance counts and metrics, and do not present subset results
as full-validation accuracy. The minimum allocation overrepresents rare strata.
If a class has insufficient examples or uncertainty remains, expand validation
with a recorded, prediction-independent selection before further comparison;
rerun all compared models on the expanded set. Never swap difficult images for
easy ones or use the final test to choose a round. Add training examples based on
development failure analysis, retaining prior training membership and audit history.
Training on a smaller corrected subset may still fail the quality target.

An export is a new candidate dataset, not a model-quality approval. Track the
finished candidate with DVC, retain the original baseline, train a new isolated
run, and compare all three validation classes on the same corrected labels.
Only a model with reviewed quality evidence can be proposed for a stable alias.
Actual approved-model rollback and post-rollback monitoring remain required.

## Local review commands

```powershell
.\.venv\Scripts\python.exe -m scripts.review_annotations init --review reports/person-review-v1
.\.venv\Scripts\python.exe -m scripts.review_annotations check --review reports/person-review-v1
# Only after every selected record has been reviewed and all its issues resolved:
.\.venv\Scripts\python.exe -m scripts.review_annotations export --review reports/person-review-v1 --output data/person-corrected-v1
```

`review.json` schema 2 starts with supplied boxes for all classes as **pending**.
Edit `helmet_boxes`, `head_boxes`, `person_boxes`, `status` (`pending`,
`needs_adjudication`, or `reviewed`),
`reviewer`, `reviewed_at`, `reason`, and `issues` for each image. Reviewed records
require a UTC ISO timestamp and an empty issues list. Do not edit source identity
fields. `check` returns exit code 2 for an incomplete review and 1 for invalid
records. Export rejects incomplete reviews, existing destinations, and paths
outside a new `data/person-corrected-*` directory.

Version 2 permits reviewed head/helmet corrections after an uncovered-head
omission was found during the person audit. A version-1 record cannot be silently
treated as a version-2 review: retain the old snapshot, copy source head/helmet
boxes explicitly, update the policy fingerprint, and recheck all previously
reviewed images under the expanded class review. No model predictions may perform
this migration's visual review.

Version 3 changes review scope, not class definitions or box geometry. The user
approved incremental review on 2026-10-17. Preserve the version-2 policy, records,
journal and DVC snapshot before explicitly updating the policy fingerprint.
Existing reviewed records keep their decisions, attribution and timestamps;
this administrative migration is not a new visual review. Existing model
suggestions remain unreviewed drawing aids in the local editor. No bulk image
review through chat or automatic model promotion is part of this workflow.
