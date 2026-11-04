# Feature extraction

Select `extraction_preset` in `../1-trident.ipynb` before submitting extraction.
The input-cohort setup remains user-configured.

| Preset | Magnification / patch size | Output feature subdirectory | Training mode |
| --- | --- | --- | --- |
| conch | 20x / 512 | `20x_512px_0px_overlap/features_conch_v15` | patch |
| prism | 20x / 224 | `20x_224px_0px_overlap/slide_features_prism` | slide |
| titan | 20x / 512 | `20x_512px_0px_overlap/slide_features_titan` | slide |

The new `slide_encoder` handler argument is forwarded to the pinned TRIDENT
batch script. TRIDENT automatically extracts Virchow patch features for PRISM
and CONCH v1.5 features for TITAN before slide aggregation. Use `task="all"` to
perform segmentation, coordinates and features in one job. Labels are copied
unchanged. PRISM's additional environs/sacremoses dependencies are installed.

Request model access and pass the existing HF_TOKEN secret: `paige-ai/Prism`
and `paige-ai/Virchow`, or `MahmoodLab/TITAN` and `MahmoodLab/conchv1_5`.
The CONCH cache preflight runs for TITAN but not PRISM. The presets use GPUs.
Other TRIDENT slide encoder names can be passed, but their optional dependencies
are not installed by these PRISM/TITAN presets.

Use separate output artifact names for each encoder and select matching cohorts
in step 2. Re-declare/rebuild the function to package these changes. The unrelated
initialization/cohort issues reserved for manual correction were not changed.
