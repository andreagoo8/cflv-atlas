# DigitalHub collection and domain-shift analysis

Execute ../3-analysis.ipynb from the workspace root. It uploads local inputs or
retrieves the completed step-two training artifact and the internal/external
test feature artifacts. Set the artifact names and GPU profile before running.
The collection job requires CPU only; each shift job can use one GPU or CPU.

The notebook declares two Python functions and submits four jobs:

| Handler/analysis | Output artifact |
| --- | --- |
| collect_results | 3-collection |
| analyze_shift / fm | 3-fm-shift |
| analyze_shift / layers | 3-abmil-layer-shift |
| analyze_shift / top64 | 3-abmil-top64-shift |

Every shift job writes four explicitly distinct output paths inside its own
workspace. Results contain repeats.csv, summary.csv, best_frequency.csv,
metadata.json and digitalhub_inputs.json (input versions and filtering settings).
Collection logs the full original report directory. All errors propagate and
failed jobs do not log success artifacts. Input data and checkpoints are not
re-uploaded in reports.

The training artifact can contain one run directly or several named run
directories. The handler stages runs/<config.run_name>/ and passes explicit
--run-dir entries, so discovery does not depend on folder names containing abmil.
The cohorts must have labels.csv and feature directories matching the training
configuration. Saved prediction CSVs must correspond to these same cohorts.

## Upstream and adaptations

Files under upstream/ originate from
https://github.com/Shahryar-Noei/colorectal-fm-local-validation at commit
c85d68e87237c9075dde9694c92803944d8a085f.

Changes to shift scripts:
- --feature-subdirs selects foundation-model directories; default is CONCH v1.5
  instead of attempting all seven paper encoders.
- --include-external-ssl disables the study-specific external filename exclusion.
  Default remains the original exclusion, exposed as exclude_external_ssl=True
  in the handler and documented in the notebook.
- Top64 attention extraction handles a one-slide batch with multiple attention
  heads (including the four-head README configuration used by step 2).

Other upstream modules are copied unchanged. predict.py, datasets.py and train.py
are included because layer-model construction imports them. TRIDENT is pinned
to the same revision as steps 1 and 2.

## Interpretation

Collection produces both full external and no-SSL subset reports.
The exclusion relies on a filename pattern, not a separate diagnosis field.
When exclusion is disabled for shift, original column names containing no_ssl
remain unchanged; use digitalhub_inputs.json to identify actual filtering.
Repeated sampling measures sensitivity to chosen slides; it is not repeated
training. Counts are clipped to available data; very small cohorts can produce
identical repetitions and unstable estimates.

topk64_positive_evidence averages the pre-attention representations of the
64 highest-attention patches in each sampled bag (or fewer if the bag is smaller).
It does not explicitly compute positive-class evidence.
The upstream knn_domain_mixing field is entropy of neighboring domain labels:
higher values indicate greater mixing. Distance metrics generally indicate
greater separation as they increase. Do not compare different metrics' numerical
scales directly. With one encoder or run, comparative ranking stability provides
no useful model-selection evidence.

## Final statistical analysis

Run `../4-statistics.ipynb` after these jobs complete. It consumes the four report
artifacts and step-2 training results, and publishes the upstream statistical
tables and figures. Its README documents the reconstructed prediction-output
enrichment, which is not defined by the published upstream scripts.
