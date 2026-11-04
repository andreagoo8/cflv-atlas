"""DigitalHub handler for upstream ABMIL, mean/max pooling, and slide MLP training."""

from __future__ import annotations

import csv
import json
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path


def _cohort_root(destination: Path, feature_subdir: str) -> Path:
    """Accept a cohort directly or inside one directory preserved by download."""
    candidates = [p.parent for p in destination.rglob('labels.csv')
                  if (p.parent / feature_subdir).is_dir()]
    if len(candidates) != 1:
        raise ValueError(
            f'{destination}: expected exactly one cohort containing labels.csv and '
            f'{feature_subdir}; found {len(candidates)}. Upload one cohort per artifact.'
        )
    return candidates[0]


def _validate_cohort(root: Path, feature_subdir: str) -> set[int]:
    """Check label format and feature matching before starting training."""
    available = {p.stem for p in (root / feature_subdir).glob('*.h5')}
    with (root / 'labels.csv').open(newline='') as stream:
        reader = csv.DictReader(stream)
        if not {'slide_name', 'slide_label'} <= set(reader.fieldnames or []):
            raise ValueError(f'{root}/labels.csv needs slide_name and slide_label columns')
        rows = list(reader)
    matched = [r for r in rows if r['slide_name'] in available and r['slide_label']]
    if not matched:
        raise ValueError(f'{root}: no labeled slides match the .h5 feature filenames')
    labels = {int(r['slide_label']) for r in matched}
    print(f'{root.name}: {len(matched)} matched labels; {len(rows) - len(matched)} unmatched/empty rows', flush=True)
    return labels


def train_abmil(
    project,
    train_features,
    val_features,
    internal_test_features,
    external_test_features,
    output_artifact_name='2-abmil-conch-v15-results',
    feature_subdir='20x_512px_0px_overlap/features_conch_v15',
    num_patches=2000,
    seed=1234,
    run_name='patch_features_conch_v15_abmil_np2000_h4_hd256_hid512_do02',
    epochs=40,
    patience=8,
    batch_size=8,
    eval_batch_size=16,
    num_workers=4,
    lr=3e-4,
    weight_decay=1e-4,
    hidden_dim=512,
    dropout=0.2,
    n_heads=4,
    head_dim=256,
    use_scheduler=True,
    no_gated=False,
    mode='patch',
    tile_pooling='abmil',
):
    """Download four cohorts, run upstream training, and upload all run outputs.

    Inputs are DigitalHub artifacts, each containing labels.csv and HDF5
    features. Other than paths, defaults match the upstream README training example.
    """
    if mode not in {'patch', 'slide'} or tile_pooling not in {'abmil', 'mean', 'max'}:
        raise ValueError('mode must be patch/slide; tile_pooling must be abmil/mean/max')
    subdir = Path(feature_subdir)
    if subdir.is_absolute() or '..' in subdir.parts:
        raise ValueError('feature_subdir must be a relative path within each cohort')
    if not run_name or run_name in {'.', '..'} or Path(run_name).name != run_name:
        raise ValueError('run_name must be a single directory name')
    if min(epochs, patience, batch_size, eval_batch_size) < 1 or num_workers < 0:
        raise ValueError('Training counts must be positive; num_workers may be zero')
    if mode == 'patch' and tile_pooling == 'abmil' and num_patches < 1:
        raise ValueError('ABMIL requires num_patches > 0')

    base = Path('local_data').resolve()
    base.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix='abmil-', dir=base))
    inputs = dict(train=train_features, val=val_features,
                  internal_test=internal_test_features, external_test=external_test_features)
    roots = {}
    class_ids = set()
    for split, artifact in inputs.items():
        destination = work / 'features' / split
        destination.mkdir(parents=True)
        print(f'Downloading {split} features to {destination}', flush=True)
        artifact.download(str(destination), overwrite=True)
        roots[split] = _cohort_root(destination, feature_subdir)
        class_ids.update(_validate_cohort(roots[split], feature_subdir))
    if len(class_ids) < 2:
        raise ValueError(
            f'Training requires at least two classes; matched slide_label values are '
            f'{sorted(class_ids)}. Check labels.csv and matching .h5 filenames in the '
            'input cohorts, then regenerate the feature artifacts with correct labels.'
        )
    if sorted(class_ids) != list(range(len(class_ids))):
        raise ValueError('slide_label must use contiguous class IDs starting at 0, consistently across all cohorts')

    output_root = work / 'runs'
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True)
    script = Path(__file__).resolve().parent / 'upstream' / 'main.py'
    options = dict(mode=mode, tile_pooling=tile_pooling, feature_subdir=feature_subdir,
                   **{f'{split}_root': str(root) for split, root in roots.items()},
                   num_patches=num_patches, seed=seed, output_root=str(output_root),
                   run_name=run_name, epochs=epochs, patience=patience,
                   batch_size=batch_size, eval_batch_size=eval_batch_size,
                   num_workers=num_workers, lr=lr, weight_decay=weight_decay,
                   hidden_dim=hidden_dim, dropout=dropout, n_heads=n_heads, head_dim=head_dim)
    command = [sys.executable, '-u', str(script)]
    for key, value in options.items():
        command.extend([f'--{key}', str(value)])
    for key, enabled in dict(use_scheduler=use_scheduler, no_gated=no_gated).items():
        if enabled:
            command.append(f'--{key}')
    print(f'Running: {shlex.join(command)}', flush=True)
    subprocess.run(command, check=True)

    expected = ['best_model.pt', 'config.json', 'history.json', 'metrics.json',
                'summary.csv', 'preds_val.csv', 'preds_internal_test.csv', 'preds_external_test.csv']
    missing = [name for name in expected if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError(f'Training finished without expected outputs: {missing}')
    (run_dir / 'input_artifacts.json').write_text(json.dumps(
        {split: artifact.key for split, artifact in inputs.items()}, indent=2
    ))
    project.log_artifact(name=output_artifact_name, kind='artifact', source=str(run_dir))
