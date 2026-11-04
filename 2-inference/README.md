# Standalone checkpoint inference on DigitalHub

`../2-inference.ipynb` is an optional branch after training. It registers
`main:predict_classifier`, builds the image, and runs the original `predict.py`
without training. Supports patch ABMIL, mean/max pooling and slide MLP models.

Inputs: one training-result artifact and four labeled feature artifacts (train,
validation, internal test and external test). The original inference program
requires training labels/features for class weights and class metadata even when
`save_train=False`. Preserve the training class-ID mapping and feature dimension.
This is labeled-cohort evaluation, not an unlabeled serving endpoint.

Set `run_name` when the training artifact contains multiple runs. The default
checkpoint is `best_model.pt`; `checkpoint_name` can select another relative
checkpoint path within that run. Architecture, pooling, feature directory,
seed, patch count and gating are explicitly passed from `config.json`. This
avoids upstream argparse defaults overriding checkpoint settings. Batch size,
worker count and optional training-set evaluation are configurable.

All cohort paths are replaced with job-local downloads. No Hugging Face token
or foundation-model weights are needed. Use one GPU for ABMIL; the pinned
TRIDENT ABMIL forward assumes CUDA. Mean/max and slide MLP inference may run on
CPU by removing GPU resources and using profile=None.

The output artifact preserves the training run and adds a unique
`predictions_inference-*/` folder with metrics.json, summary.csv, three prediction
CSVs, optional preds_train.csv, and digitalhub_inputs.json. Existing training
predictions are unchanged. The artifact is directly compatible with step 3's
collection handler, which recognizes `predictions_*` directories. Shift analyses
and step 4 still use original test predictions at the run root; standalone
inference on different cohorts does not replace them automatically.

All four `upstream/` modules are unchanged copies from the source repository at
`c85d68e87237c9075dde9694c92803944d8a085f`, copied from the existing step-3 snapshot.
Dependencies use the same pinned TRIDENT revision as steps 1–3. No existing
function versions or artifacts are modified by editing these files.

Wrapper tests: `python -m pytest 2-inference/tests`. They verify command dispatch,
artifact handling, architecture restoration and ambiguous-run rejection with
mocked subprocesses; they do not validate GPU execution or downloaded models.
