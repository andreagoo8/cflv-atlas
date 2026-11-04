# ABMIL training on DigitalHub

The existing `train_abmil` handler name is retained, but `mode` and `tile_pooling`
now expose all upstream classifier branches. Select `classifier_preset` in the
notebook: `abmil`, `mean`, `max`, `prism`, or `titan`.

| Selection | mode | tile_pooling | Input |
| --- | --- | --- | --- |
| ABMIL | patch | abmil | Patch embeddings |
| Mean / max | patch | mean / max | Same patch embeddings; all patches are pooled |
| PRISM / TITAN | slide | abmil (ignored) | Slide embeddings from step 1 |

Mean/max and slide modes train the original `SlideMLPClassifier`. `num_patches`
and attention-head/gating settings only affect patch ABMIL. Hidden dimension,
dropout and optimizer settings apply to all classifiers. Select the matching
four input artifact names. Non-ABMIL presets use distinct default run/output
names. For CPU MLP jobs, set profile=None and remove the GPU resource; the
default one-GPU profile works for all modes.

Open `../2-abmil.ipynb` from the workspace root. The notebook registers or retrieves
four cohort artifacts, declares `main:train_abmil`, builds its image, and submits
a training/evaluation job. Adjust artifact names and the GPU profile for your project.
Use separate cohorts; the function does not create dataset splits.

The notebook and handler defaults now match the upstream README example:
2,000 patches, hidden dimension 512, dropout 0.2, four attention heads, head
dimension 256, gated attention, and the cosine scheduler enabled. The run name
is `patch_features_conch_v15_abmil_np2000_h4_hd256_hid512_do02`.
Re-declare/rebuild the function and submit a new run to use these settings;
existing function versions and trained artifacts are not changed.

Each artifact contains:

```text
labels.csv
20x_512px_0px_overlap/features_conch_v15/<slide_name>.h5
```

Labels require `slide_name` (without `.h5` or `.svs`) and integer `slide_label`
columns. Class IDs must be consistent across cohorts and contiguous from zero.
Patch HDF5 files must contain a nonempty 2D `features` dataset with the same
embedding dimension. Slide mode uses a 1D vector (or singleton slide-vector
array), flattened by the upstream loader. The loader skips labels without matching feature files;
the wrapper reports their count and rejects an empty cohort.

The output artifact contains the best checkpoint, configuration, training history,
metrics, summary, prediction CSVs for all three evaluation cohorts, and input
artifact keys. Absolute cohort paths in config.json refer to the completed job;
download the input artifacts and supply new paths when doing later inference.

The original training code selects one CUDA device automatically (CPU otherwise).
No foundation-model weights or HF_TOKEN are needed to train from saved features.
Change batch sizes/num_workers through notebook parameters when needed. No job is
submitted until you execute the notebook run cell.

After step 3, `../4-statistics.ipynb` consumes this result artifact together with
the collection, raw-FM, default-layer, and top64 reports for final statistics.

For standalone checkpoint evaluation, use `../2-inference.ipynb`. Collection in
step 3 supports every classifier mode. Raw-FM shift supports PRISM/TITAN feature
directories; ABMIL-layer/top64 analysis only applies to ABMIL runs. Step 4 retains
the upstream patch-FM scope: it compares mean/max baselines alongside ABMIL and
does not add PRISM/TITAN-specific statistical panels.

## Source provenance

`upstream/main.py`, `datasets.py`, `models.py`, and `train.py` are unchanged copies
from https://github.com/Shahryar-Noei/colorectal-fm-local-validation at commit
`c85d68e87237c9075dde9694c92803944d8a085f`.
Dependencies use the same pinned TRIDENT revision as step 1.
The wrapper preserves upstream training behavior, including its sampling behavior:
the upstream training sampler uses an unseeded NumPy default_rng, so passing seed
1234 alone does not guarantee identical training bags on repeated executions.
