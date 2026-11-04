"""Run the original checkpoint inference program as a DigitalHub job."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def predict_classifier(project, training_results, train_features, val_features,
                       internal_test_features, external_test_features,
                       output_artifact_name="2-inference-results", run_name=None,
                       checkpoint_name="best_model.pt", batch_size=32,
                       num_workers=4, save_train=False):
    """Re-evaluate a saved run without training; preserve upstream labeled-cohort semantics."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive; num_workers must be nonnegative")
    checkpoint = Path(checkpoint_name)
    if checkpoint.is_absolute() or '..' in checkpoint.parts:
        raise ValueError("checkpoint_name must be relative to the selected run")
    base = Path("local_data").resolve()
    base.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="inference-", dir=base))
    downloaded = work / "training"
    downloaded.mkdir()
    training_results.download(str(downloaded), overwrite=True)
    candidates = []
    for path in downloaded.rglob("config.json"):
        cfg = json.loads(path.read_text())
        if run_name is None or cfg.get("run_name", path.parent.name) == run_name:
            candidates.append((path.parent, cfg))
    if len(candidates) != 1:
        raise ValueError("Select exactly one training run using run_name when the artifact contains several")
    run_dir, cfg = candidates[0]
    if not (run_dir / checkpoint).is_file():
        raise FileNotFoundError(run_dir / checkpoint)
    required = ["mode", "tile_pooling", "feature_subdir", "seed", "num_patches",
                "hidden_dim", "dropout", "n_heads", "head_dim", "no_gated"]
    missing = set(required) - set(cfg)
    if missing:
        raise ValueError(f"Training config is missing {sorted(missing)}")
    subdir = Path(cfg["feature_subdir"])
    if subdir.is_absolute() or '..' in subdir.parts:
        raise ValueError("feature_subdir must be a relative cohort path")
    if cfg["mode"] not in {"patch", "slide"} or cfg["tile_pooling"] not in {"abmil", "mean", "max"}:
        raise ValueError("Unsupported training mode/pooling")
    # Keep all imports needed by the child in its own source directory.
    # Helpers are loaded by path to avoid collisions with other notebook modules.
    import importlib.util
    spec = importlib.util.spec_from_file_location("inference_cohorts", Path(__file__).with_name("cohorts.py"))
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    inputs = dict(train=train_features, val=val_features, internal_test=internal_test_features,
                  external_test=external_test_features)
    roots, classes = {}, set()
    for split, artifact in inputs.items():
        destination = work / split
        destination.mkdir()
        artifact.download(str(destination), overwrite=True)
        roots[split] = helpers._cohort_root(destination, cfg["feature_subdir"])
        classes.update(helpers._validate_cohort(roots[split], cfg["feature_subdir"]))
    if sorted(classes) != list(range(len(classes))):
        raise ValueError("Labels must be contiguous class IDs starting at zero")
    output = run_dir / ("predictions_" + work.name)
    # Upstream argparse defaults otherwise override config architecture/sampling.
    options = {key: cfg[key] for key in required if key != "no_gated"}
    options.update(run_dir=run_dir, checkpoint=run_dir / checkpoint, output_dir=output,
                   batch_size=batch_size, num_workers=num_workers,
                   **{f"{split}_root": root for split, root in roots.items()})
    command = [sys.executable, "-u", str(Path(__file__).parent / "upstream" / "predict.py")]
    for key, value in options.items():
        command.extend([f"--{key}", str(value)])
    if cfg["no_gated"]:
        command.append("--no_gated")
    if save_train:
        command.append("--save_train")
    subprocess.run(command, check=True)
    expected = ["metrics.json", "summary.csv", "preds_val.csv", "preds_internal_test.csv",
                "preds_external_test.csv"] + (["preds_train.csv"] if save_train else [])
    if any(not (output / name).is_file() for name in expected):
        raise RuntimeError("Inference did not produce all expected files")
    (output / "digitalhub_inputs.json").write_text(json.dumps({
        "inputs": {"training_results": training_results.key,
                   **{split: artifact.key for split, artifact in inputs.items()}},
        "training_config": cfg, "checkpoint_name": checkpoint_name,
        "settings": {"batch_size": batch_size, "num_workers": num_workers, "save_train": save_train},
        "upstream_commit": "c85d68e87237c9075dde9694c92803944d8a085f",
    }, indent=2))
    # Preserve the run/predictions_* layout understood by collect.py.
    project.log_artifact(name=output_artifact_name, kind="artifact", source=str(run_dir))
