# Final statistics on DigitalHub

Run `../4-statistics.ipynb` after steps 2 and 3 have completed. This is a separate
CPU-only job because it consumes saved results rather than embeddings or GPUs.
It uses the same source-directory, Python handler, build, job, and output-artifact
pattern as the earlier notebooks.

## Inputs and outputs

Five artifacts are required: step-2 training results, step-3 collection, raw-FM
shift, default-layer shift, and top64 shift. An input may contain multiple result
folders for multiple experiments. Run names must be unique. Bundle each report
once; duplicated rows are rejected rather than counted twice. Training and
collection run sets must match, and layer/top64 inputs must cover every ABMIL run.
Use the same cohorts and sampling settings across shift jobs, with
`exclude_external_ssl=True`, which is the original statistics notebook's choice.
The notebook supports explicit artifact entity IDs to pin input versions.

The output artifact contains:

- `training_runs_with_shift_metrics.csv`: collection plus reconstructed enrichment.
- `fm_repeats.csv`, `layer_repeats.csv`, `top64_repeats.csv`: combined input tables.
- `figures/`: upstream performance, method, hyperparameter, correlation, and
  instability tables and PNG/PDF plots.
- `analysis_status.json` and `README.md`: completed/skipped panels and limitations.
- `enrichment_definitions.json`, `training_configs.json`,
  `source_shift_provenance.json`, and `digitalhub_inputs.json`: reproducibility details.

No training or feature extraction is performed. Prediction probabilities are
read from `preds_internal_test.csv` and `preds_external_test.csv`. An empty cohort
after filtering, invalid probabilities, incompatible reports, or runtime errors
fail the job before publishing a success artifact.

If statistics reports fewer than two `prob_class_*` columns, inspect the saved
prediction headers and the matched `slide_label` values in the step-2 cohorts.
A model trained with only class `0` writes only `prob_class_0 = 1.0`; its margin
is undefined and its predictions cannot support these reports. Correct the
labels and cohort composition to cover at least two classes with contiguous IDs
starting at zero, ensuring the training cohort contains those classes. Regenerate
the feature artifacts with corrected labels, rerun step 2 and the dependent
step-3 collection/layer/top64 analyses, and select the new artifact versions.
Rerun raw-FM shift too if the cohorts changed. Adding a fabricated probability
column does not repair the trained model.

## Source and adaptation

`upstream/results_Statistics.ipynb` is an unchanged source snapshot from
https://github.com/Shahryar-Noei/colorectal-fm-local-validation at
`c85d68e87237c9075dde9694c92803944d8a085f`.
`analysis.py` converts its cells into `render_reports`; comments identify the
original cell numbers. `report.py` connects the artifact tables and supplies the
missing enrichment. `main.py` downloads artifacts and invokes it in a subprocess.

The upstream Mann–Whitney, Holm correction, Kruskal–Wallis, Spearman, repeated
correlation summaries, and instability calculations are retained. Adaptations:

- Replace hard-coded report dates and `Paper/figures` with downloaded input paths
  and a job-local output directory. Use the noninteractive Agg plotting backend.
- Read hyperparameters from `config.json`; configuration labels include actual
  values instead of assuming the paper's run-name patterns. Present configuration
  and hyperparameter levels dynamically, including non-paper settings.
- Limit plots to available supported FMs (the same five patch FMs as upstream).
  Method plots support available ABMIL/mean/max runs. The upstream statistics
  deliberately excludes PRISM/TITAN from these patch-FM comparisons.
- Record unavailable comparisons: skip hyperparameter panels without variation,
  instability panels without at least two runs per FM, and the correlation
  boxplot when all correlations are undefined. CSV correlation values remain NaN.
- Keep undefined uncertainty as NaN in tables; omit singleton error bars in
  figures. All-identical Kruskal inputs produce undefined results instead of a
  crash. Adjust performance axes so low or negative MCC values remain visible.
- The optional synthetic illustration from the last cell is disabled by default
  and named `illustration_*` to distinguish it from observed data.

## Reconstructed enrichment — not verified author definitions

The repository has no generator for `training_runs_with_shift_metrics.csv`.
This adaptation computes `method_family` from mode/pooling and defines:

- Per-slide entropy: `-sum(p * ln(p))`, unnormalized natural logarithm;
  zero probabilities contribute zero.
- Confidence: maximum class probability.
- Margin: largest minus second-largest class probability.
- Entropy shift: mean external entropy minus mean internal entropy.
- Confidence/margin drop: internal mean minus external mean.

All means weight slides equally. External predictions use the same pure-SSL
filename regex as upstream collection; they are not sampled for this enrichment.
These choices are explicitly recorded in each output. They make the final stage
executable, but exact equivalence to the authors' unpublished enrichment cannot
be claimed without their definitions.

## Interpretation and verification

A single CONCH run is enough to execute this step and obtain descriptive results,
but not to infer differences between configurations. Correlations need at least
three observations with nonconstant values. Undefined tests are not zero effects.
As upstream, repetitions represent cohort resampling, not independent retraining.
The first performance panel retains upstream standard-deviation error bars,
despite its historical `_bar_sem` filename; method-panel bars use SEM.

The wrapper checks matching sampling settings, cohort artifact versions (per
feature directory), repetition keys, counts, and layer MCC values. Upstream does
not export sampled slide IDs, so equal cohort membership/order across different
encoders cannot be proved from these reports alone. Supply matched cohorts.

Local verification: `python -m pytest 4-statistics/tests` in an environment with
`4-statistics/requirements.txt` and pytest installed. Tests use synthetic artifacts
and do not submit DigitalHub jobs.
