"""DigitalHub handlers for result collection and repeated domain-shift analysis."""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _workspace():
    base = Path("local_data").resolve()
    base.mkdir(exist_ok=True)
    return Path(tempfile.mkdtemp(prefix="analysis-", dir=base))


def _download(artifact, destination):
    destination.mkdir(parents=True)
    print(f"Downloading {artifact.key} to {destination}", flush=True)
    artifact.download(str(destination), overwrite=True)


def _stage_runs(artifact, work):
    downloaded = work / "downloaded_runs"
    _download(artifact, downloaded)
    configs = sorted(downloaded.rglob("config.json"))
    if not configs:
        raise ValueError("Training artifact contains no config.json")
    runs = work / "runs"
    runs.mkdir()
    for config in configs:
        cfg = json.loads(config.read_text())
        name = cfg.get("run_name") or config.parent.name
        if not name or name in {".", ".."} or Path(name).name != name:
            raise ValueError(f"Invalid run_name in {config}: {name!r}")
        target = runs / name
        if target.exists():
            raise ValueError(f"Duplicate run name in training artifact: {name}")
        shutil.move(str(config.parent), str(target))
    return runs


def _stage_cohort(artifact, destination, subdirs):
    _download(artifact, destination)
    candidates = [p.parent for p in destination.rglob("labels.csv")
                  if all((p.parent / subdir).is_dir() for subdir in subdirs)]
    if len(candidates) != 1:
        raise ValueError(
            f"{destination}: expected one cohort with labels.csv and {subdirs}; "
            f"found {len(candidates)}. Check the selected feature directories."
        )
    for subdir in subdirs:
        if not any((candidates[0] / subdir).glob("*.h5")):
            raise ValueError(f"No HDF5 features in {candidates[0] / subdir}")
    return candidates[0]


def _execute(script, args):
    path = Path(__file__).resolve().parent / "upstream" / script
    command = [sys.executable, "-u", str(path), *map(str, args)]
    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True)


def _log(project, output, name, inputs, settings):
    (output / "digitalhub_inputs.json").write_text(json.dumps(
        {"inputs": {key: artifact.key for key, artifact in inputs.items()},
         "settings": settings,
         "upstream_commit": "c85d68e87237c9075dde9694c92803944d8a085f"},
        indent=2,
    ))
    project.log_artifact(name=name, kind="artifact", source=str(output))


def collect_results(project, training_results, output_artifact_name="3-collection", top_k=10):
    """Collect a step-two result artifact (or an artifact containing several runs)."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    work = _workspace()
    runs = _stage_runs(training_results, work)
    output = work / "reports"
    _execute("collect.py", ["--runs_root", runs, "--output_dir", output, "--top_k", top_k])
    if not (output / "final_report.md").is_file():
        raise RuntimeError("Collection produced no final_report.md")
    _log(project, output, output_artifact_name, {"training_results": training_results},
         {"top_k": top_k})


def analyze_shift(
    project, training_results, internal_features, external_features,
    output_artifact_name, analysis="fm",
    feature_subdirs=None, exclude_external_ssl=True,
    n_repeats=20, target_per_domain=80, sample_external=True, seed=1234,
    knn_k=15, sigma_sample_size=512, energy_sample_size=2048,
    energy_chunk_size=512, eps=1e-12, device="cuda", batch_size=None,
):
    """Run one of fm, layers, or top64 with explicit cohort paths and outputs."""
    if analysis not in {"fm", "layers", "top64"}:
        raise ValueError("analysis must be fm, layers, or top64")
    if device not in {"cuda", "cpu", "auto"}:
        raise ValueError("device must be cuda, cpu, or auto")
    if min(n_repeats, knn_k, sigma_sample_size, energy_sample_size, energy_chunk_size) < 1:
        raise ValueError("Repeat and sampling counts must be positive")
    if target_per_domain < 0 or (batch_size is not None and batch_size < 1):
        raise ValueError("Invalid target_per_domain or batch_size")
    subdirs = feature_subdirs or ["20x_512px_0px_overlap/features_conch_v15"]
    if isinstance(subdirs, str):
        subdirs = [subdirs]
    for subdir in subdirs:
        if Path(subdir).is_absolute() or ".." in Path(subdir).parts:
            raise ValueError("Feature directories must be relative cohort paths")
    if analysis != "fm" and len(subdirs) != 1:
        raise ValueError("Layer analysis requires one feature directory matching the trained model")

    work = _workspace()
    runs = _stage_runs(training_results, work)
    internal = _stage_cohort(internal_features, work / "internal_test", subdirs)
    external = _stage_cohort(external_features, work / "external_test", subdirs)
    output = work / "reports" / analysis
    output.mkdir(parents=True)
    paths = {"out-csv": output / "repeats.csv",
             "out-summary-csv": output / "summary.csv",
             "out-stability-csv": output / "best_frequency.csv",
             "out-json": output / "metadata.json"}
    options = {"runs-root": runs, "internal-root": internal, "external-root": external,
               "device": device, "n-repeats": n_repeats, "target-per-domain": target_per_domain,
               "seed": seed, "knn-k": knn_k, "sigma-sample-size": sigma_sample_size,
               "energy-sample-size": energy_sample_size, "energy-chunk-size": energy_chunk_size,
               "eps": eps, **paths}
    args = []
    for key, value in options.items():
        args.extend([f"--{key}", value])
    if sample_external:
        args.append("--sample-external")
    if not exclude_external_ssl:
        args.append("--include-external-ssl")
    if analysis == "fm":
        script = "compute_fm_shift_repeated.py"
        args.extend(["--feature-subdirs", *subdirs])
    else:
        script = "compute_abmil_layer_shift_repeated.py"
        args.extend(["--feature-subdir", subdirs[0]])
        for run_dir in sorted(runs.iterdir()):
            cfg = json.loads((run_dir / "config.json").read_text())
            if cfg.get("mode") != "patch" or cfg.get("tile_pooling", "abmil") != "abmil":
                raise ValueError(f"{run_dir.name} is not a patch ABMIL run")
            if cfg.get("feature_subdir") != subdirs[0]:
                raise ValueError(f"{run_dir.name}: requested features differ from training config")
            for name in ("best_model.pt", "preds_internal_test.csv", "preds_external_test.csv"):
                if not (run_dir / name).is_file():
                    raise FileNotFoundError(run_dir / name)
            args.extend(["--run-dir", run_dir])
        layers = ("topk64_positive_evidence" if analysis == "top64"
                  else "pre_attention_mean,post_attention,penultimate")
        args.extend(["--layers", layers, "--topk", 64])
        if batch_size is not None:
            args.extend(["--batch-size", batch_size])
    _execute(script, args)
    if not all(p.is_file() for p in paths.values()):
        raise RuntimeError("Shift analysis did not produce all four output files")
    _log(project, output, output_artifact_name,
         dict(training_results=training_results, internal_features=internal_features,
              external_features=external_features),
         {"analysis": analysis, "feature_subdirs": subdirs,
          "exclude_external_ssl": exclude_external_ssl, "sample_external": sample_external,
          "command_options": {k: str(v) for k, v in options.items()}})
