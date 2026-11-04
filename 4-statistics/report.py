"""Validate report inputs, supply the missing enrichment, and run upstream analyses."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import render_reports


PURE_SSL_REGEX = r"^[^_]+_sessile[_ ]serrated(?:[_ ]lesion)?(?:[_ ]displasia)?$"
METRICS = ["mmd_raw", "fid_raw", "coral_raw", "energy_raw", "knn_domain_mixing"]
KNOBS = ["num_patches", "n_heads", "head_dim", "hidden_dim", "dropout",
         "no_gated", "lr", "weight_decay"]
DEFINITIONS = {
    "provenance": "Adaptation-defined enrichment; original generating code was not published.",
    "entropy": "-sum(p * ln(p)); natural logarithm, unnormalized; 0*ln(0)=0",
    "confidence": "max class probability per slide",
    "margin": "largest minus second-largest class probability per slide",
    "aggregation": "Arithmetic mean over slides, separately for each cohort",
    "entropy_shift": "mean external entropy minus mean internal entropy",
    "confidence_drop": "mean internal confidence minus mean external confidence",
    "margin_drop": "mean internal margin minus mean external margin",
    "external_cohort": "Same filename-based pure-SSL exclusion as collect.py and upstream statistics",
}


def read_tables(directory, filename, required, unique):
    files = sorted(Path(directory).rglob(filename))
    if not files:
        raise ValueError(f"No {filename} in {directory}")
    frames = []
    for path in files:
        df = pd.read_csv(path)
        missing = set(required) - set(df.columns)
        if missing or df.empty:
            raise ValueError(f"{path}: empty table or missing columns {sorted(missing)}")
        frames.append(df)
    result = pd.concat(frames, ignore_index=True)
    if result[unique].isna().any().any() or result.duplicated(unique).any():
        raise ValueError(f"{filename}: missing or duplicate keys {unique}; bundle each report once")
    return result


def prediction_summary(path, exclude_ssl):
    df = pd.read_csv(path)
    if "slide_name" not in df or df.slide_name.duplicated().any():
        raise ValueError(f"{path}: missing or duplicate slide names")
    columns = sorted((c for c in df if c.startswith("prob_class_")),
                     key=lambda c: int(c.rsplit("_", 1)[1]))
    if len(columns) < 2:
        detail = ("The saved model has only one class probability column. "
                  if len(columns) == 1 else "The saved predictions contain no class probabilities. ")
        raise ValueError(
            f"{path}: need at least two prob_class_* columns; found {columns}. "
            f"{detail}Check matched slide_label values in the step-2 input cohorts "
            "and train with at least two classes. Regenerate training results and "
            "dependent step-3 reports before rerunning statistics."
        )
    if exclude_ssl:
        df = df.loc[~df.slide_name.astype(str).str.contains(
            PURE_SSL_REGEX, case=False, regex=True, na=False)]
    if df.empty:
        raise ValueError(f"{path}: no slides remain after filtering")
    p = df[columns].to_numpy(dtype=float)
    if (not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError(f"{path}: invalid class probabilities")
    entropy = -(p * np.log(np.where(p > 0, p, 1))).sum(axis=1)
    ordered = np.sort(p, axis=1)
    return {"entropy": float(entropy.mean()), "confidence": float(ordered[:, -1].mean()),
            "margin": float((ordered[:, -1] - ordered[:, -2]).mean()), "n": len(df),
            "probability_columns": columns}


def enrich(collection, training_directory):
    runs = {}
    configs = {}
    for path in sorted(Path(training_directory).rglob("config.json")):
        cfg = json.loads(path.read_text())
        name = cfg.get("run_name", path.parent.name)
        if name in runs:
            raise ValueError(f"Duplicate training run {name}")
        runs[name], configs[name] = path.parent, cfg
    if set(collection.run_name) != set(runs):
        raise ValueError("Collection run names must exactly match the supplied training runs")
    result = collection.copy()
    rows = []
    for row in result.itertuples(index=False):
        cfg = configs[row.run_name]
        for key in ("mode", "tile_pooling", "feature_subdir"):
            if cfg.get(key) != getattr(row, key):
                raise ValueError(f"{row.run_name}: collection/config mismatch for {key}")
        if cfg["mode"] == "patch" and cfg["tile_pooling"] == "abmil":
            if set(KNOBS) - set(cfg):
                raise ValueError(f"{row.run_name}: config is missing ABMIL hyperparameters")
        root = runs[row.run_name]
        internal = prediction_summary(root / "preds_internal_test.csv", False)
        external = prediction_summary(root / "preds_external_test.csv", True)
        if internal["probability_columns"] != external["probability_columns"]:
            raise ValueError(f"{row.run_name}: inconsistent probability columns")
        rows.append({
            "method_family": f"patch_{cfg['tile_pooling']}" if cfg["mode"] == "patch" else "slide",
            "entropy_shift": external["entropy"] - internal["entropy"],
            "confidence_drop": internal["confidence"] - external["confidence"],
            "margin_drop": internal["margin"] - external["margin"],
            "prediction_n_internal": internal["n"], "prediction_n_external_no_ssl": external["n"],
        })
    for key in rows[0]:
        result[key] = [r[key] for r in rows]
    return result, configs


def validate_shift_inputs(paths, training, fm, layers, top64):
    """Reject incompatible settings instead of correlating unrelated repetitions."""
    signatures = []
    provenance = []
    for key, expected in [("fm_shift", "fm"), ("layer_shift", "layers"), ("top64_shift", "top64")]:
        for csv in Path(paths[key]).rglob("repeats.csv"):
            manifest = csv.with_name("digitalhub_inputs.json")
            if not manifest.exists():
                raise ValueError(f"Missing step-3 provenance: {manifest}")
            data = json.loads(manifest.read_text())
            settings = data["settings"]
            if settings["analysis"] != expected or not settings["exclude_external_ssl"]:
                raise ValueError("Use matching step-3 reports with exclude_external_ssl=True, as upstream statistics requires")
            options = settings["command_options"]
            rows = pd.read_csv(csv)
            if not set(rows.feature_subdir) <= set(settings["feature_subdirs"]):
                raise ValueError(f"{csv}: feature directories disagree with report provenance")
            grouping = ["feature_subdir"] if expected == "fm" else ["run_name", "layer_name"]
            repetitions = set(range(int(options["n-repeats"])))
            for _, group in rows.groupby(grouping):
                if set(group.repetition) != repetitions:
                    raise ValueError(f"{csv}: incomplete or unexpected repetitions")
            signature = {name: options[name] for name in
                         ["seed", "n-repeats", "target-per-domain", "knn-k", "sigma-sample-size",
                          "energy-sample-size", "energy-chunk-size", "eps"]}
            signature["sample_external"] = settings["sample_external"]
            signature["internal"] = data["inputs"]["internal_features"]
            signature["external"] = data["inputs"]["external_features"]
            # Encoders may live in different artifacts, but cohort provenance must
            # match across the three analyses for each feature directory.
            for subdir in settings["feature_subdirs"]:
                signatures.append((subdir, signature))
            provenance.append(data)
    comparable = [{k: v for k, v in signature.items() if k not in {"internal", "external"}}
                  for _, signature in signatures]
    if any(value != comparable[0] for value in comparable[1:]):
        raise ValueError("All foundation models must use matching repeated-analysis settings")
    for subdir in {s for s, _ in signatures}:
        values = [v for s, v in signatures if s == subdir]
        if any(v != values[0] for v in values[1:]):
            raise ValueError(f"Mismatched repeated-analysis settings or cohort artifacts: {subdir}")
    abmil = training[(training['mode'] == 'patch') & (training['tile_pooling'] == 'abmil')]
    if set(layers.run_name) != set(abmil.run_name) or set(top64.run_name) != set(abmil.run_name):
        raise ValueError("Layer/top64 reports must cover exactly the supplied ABMIL runs")
    if set(layers.layer_name) != {"pre_attention_mean", "post_attention", "penultimate"}:
        raise ValueError("Default layer report must include all three upstream layers")
    if set(top64.layer_name) != {"topk64_positive_evidence"}:
        raise ValueError("Expected the top64 representation report")
    combined = pd.concat([layers, top64], ignore_index=True)
    for name, group in combined.groupby("run_name"):
        configured = training.set_index("run_name").loc[name, "feature_subdir"]
        if set(group.feature_subdir) != {configured}:
            raise ValueError(f"{name}: shift features differ from training")
        counts = group.groupby("repetition").layer_name.nunique()
        if not counts.eq(4).all():
            raise ValueError(f"{name}: missing layers or repetitions")
        for col in ["mcc_internal", "mcc_external", "mcc_drop"]:
            if not group.groupby("repetition")[col].nunique(dropna=False).eq(1).all():
                raise ValueError(f"{name}: inconsistent repeated predictions across layers")
    sizes = ["n_internal_total", "n_external_no_ssl_total", "n_internal_sample", "n_external_sample"]
    joined = combined.merge(fm, on=["feature_subdir", "repetition"], how="left",
                            suffixes=("_layer", "_fm"), validate="many_to_one", indicator=True)
    if not joined['_merge'].eq('both').all():
        raise ValueError("Missing matching raw FM repetitions")
    for size in sizes:
        if not joined[f"{size}_layer"].eq(joined[f"{size}_fm"]).all():
            raise ValueError(f"FM/layer cohort sizes differ for {size}; matched sampling is not established")
    return provenance


def build_reports(paths, output, include_illustration=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    collection = read_tables(paths["collection"], "training_runs.csv",
                             ["run_name", "mode", "tile_pooling", "feature_subdir",
                              "internal_test_mcc", "external_test_no_ssl_mcc"], ["run_name"])
    training, configs = enrich(collection, paths["training_results"])
    sizes = ["n_internal_total", "n_external_no_ssl_total", "n_internal_sample", "n_external_sample"]
    fm = read_tables(paths["fm_shift"], "repeats.csv", METRICS + sizes +
                     ["feature_subdir", "repetition", "fm_name"], ["feature_subdir", "repetition"])
    required = METRICS + sizes + ["run_name", "feature_subdir", "layer_name", "repetition",
                                 "mcc_internal", "mcc_external", "mcc_drop"]
    layers = read_tables(paths["layer_shift"], "repeats.csv", required,
                         ["run_name", "layer_name", "repetition"])
    top64 = read_tables(paths["top64_shift"], "repeats.csv", required,
                        ["run_name", "layer_name", "repetition"])
    provenance = validate_shift_inputs(paths, training, fm, layers, top64)
    training.to_csv(output / "training_runs_with_shift_metrics.csv", index=False)
    fm.to_csv(output / "fm_repeats.csv", index=False)
    layers.to_csv(output / "layer_repeats.csv", index=False)
    top64.to_csv(output / "top64_repeats.csv", index=False)
    (output / "enrichment_definitions.json").write_text(json.dumps(DEFINITIONS, indent=2))
    (output / "source_shift_provenance.json").write_text(json.dumps(provenance, indent=2))
    (output / "training_configs.json").write_text(json.dumps(configs, indent=2))
    status = render_reports(output / "training_runs_with_shift_metrics.csv",
                            output / "fm_repeats.csv", output / "layer_repeats.csv",
                            output / "top64_repeats.csv", output / "figures", configs,
                            include_illustration=include_illustration)
    (output / "analysis_status.json").write_text(json.dumps(status, indent=2))
    lines = ["# Statistical analysis", "",
             "Enrichment definitions are adaptation choices, not verified author formulas.",
             "See enrichment_definitions.json and source_shift_provenance.json.",
             "Figures and statistical CSV tables are in figures/.",
             "Undefined statistics remain NaN. Singleton error bars are omitted (CSV uncertainty stays NaN).",
             "Correlations need at least three nonconstant observations; instability needs multiple runs per FM.",
             "Same cohort artifacts/settings/counts are checked; upstream reports do not record sampled slide IDs.",
             "", "## Panel execution", ""]
    lines += [f"- {item.get('panel', 'Cell ' + str(item.get('cell')))}: {item['status']}"
              + (f" — {item['reason']}" if 'reason' in item else '') for item in status]
    (output / "README.md").write_text('\n'.join(lines) + '\n')


if __name__ == "__main__":
    build_reports(**json.loads(Path(sys.argv[1]).read_text()))
