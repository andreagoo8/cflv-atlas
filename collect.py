from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect training/inference run results and build a final report.")
    parser.add_argument(
        "--runs_root",
        type=str,
        default="/data/projects/D34H/git_repo/repo-shahry/wp-1/Classification_Trident/Codes/runs",
        help="Root folder that contains training run directories.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output report folder. Default: <runs_root>/reports/<timestamp>",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top runs to show in ranking tables.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _safe_metric(metrics: Dict[str, Any], split: str, name: str) -> float:
    try:
        return float(metrics[split][name])
    except Exception:
        return float("nan")


def _metrics_from_preds_df(df: pd.DataFrame) -> Dict[str, float]:
    if df.empty:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
            "mcc": float("nan"),
        }

    y_true = df["true_label"].astype(int).to_numpy()
    y_pred = df["pred_label"].astype(int).to_numpy()

    accuracy = float((y_true == y_pred).mean())

    labels_true = sorted(set(y_true.tolist()))
    labels_union = sorted(set(y_true.tolist()) | set(y_pred.tolist()))

    recalls = []
    f1s = []
    for c in labels_union:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())

        # F1 for macro average across labels seen in y_true or y_pred.
        denom_f1 = 2 * tp + fp + fn
        f1 = (2 * tp / denom_f1) if denom_f1 > 0 else 0.0
        f1s.append(f1)

    # Balanced accuracy: mean recall across labels present in y_true.
    for c in labels_true:
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        denom_rec = tp + fn
        rec = (tp / denom_rec) if denom_rec > 0 else 0.0
        recalls.append(rec)

    # Multiclass MCC from confusion matrix (Gorodkin formulation).
    labels_union_arr = np.array(labels_union, dtype=int)
    label_to_idx = {int(c): i for i, c in enumerate(labels_union_arr.tolist())}
    n_classes = len(labels_union_arr)
    cm = np.zeros((n_classes, n_classes), dtype=np.float64)
    for yt, yp in zip(y_true.tolist(), y_pred.tolist()):
        cm[label_to_idx[int(yt)], label_to_idx[int(yp)]] += 1.0

    s = float(cm.sum())
    c = float(np.trace(cm))
    p = cm.sum(axis=0)
    t = cm.sum(axis=1)
    denom = np.sqrt((s * s - float((p * p).sum())) * (s * s - float((t * t).sum())))
    mcc = float((c * s - float((p * t).sum())) / denom) if denom > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(np.mean(recalls) if recalls else float("nan")),
        "macro_f1": float(np.mean(f1s) if f1s else float("nan")),
        "mcc": mcc,
    }


def _external_no_ssl_metrics_from_preds(preds_csv_path: Path) -> Dict[str, float]:
    out = {
        "external_test_no_ssl_n": float("nan"),
        "external_test_no_ssl_macro_f1": float("nan"),
        "external_test_no_ssl_accuracy": float("nan"),
        "external_test_no_ssl_balanced_accuracy": float("nan"),
        "external_test_no_ssl_mcc": float("nan"),
    }
    if not preds_csv_path.exists():
        return out

    try:
        preds_df = pd.read_csv(preds_csv_path)
    except Exception:
        return out

    required = {"slide_name", "true_label", "pred_label"}
    if not required.issubset(preds_df.columns):
        return out

    slide_name = preds_df["slide_name"].astype(str)
    # SSL-only exclusion: match pure Sessile Serrated (optional Lesion/Displasia)
    # but keep mixed entities where adenoma is present in the same slide_name.
    is_pure_ssl = slide_name.str.contains(
        r"^[^_]+_sessile[_ ]serrated(?:[_ ]lesion)?(?:[_ ]displasia)?$",
        case=False,
        regex=True,
        na=False,
    )
    no_ssl_df = preds_df.loc[~is_pure_ssl].copy()

    out["external_test_no_ssl_n"] = float(len(no_ssl_df))
    if len(no_ssl_df) == 0:
        return out

    m = _metrics_from_preds_df(no_ssl_df)
    out["external_test_no_ssl_macro_f1"] = m["macro_f1"]
    out["external_test_no_ssl_accuracy"] = m["accuracy"]
    out["external_test_no_ssl_balanced_accuracy"] = m["balanced_accuracy"]
    out["external_test_no_ssl_mcc"] = m["mcc"]
    return out


def _split_metrics_from_preds(preds_csv_path: Path, split_name: str) -> Dict[str, float]:
    out = {
        f"{split_name}_accuracy": float("nan"),
        f"{split_name}_balanced_accuracy": float("nan"),
        f"{split_name}_macro_f1": float("nan"),
        f"{split_name}_mcc": float("nan"),
    }
    if not preds_csv_path.exists():
        return out
    try:
        preds_df = pd.read_csv(preds_csv_path)
    except Exception:
        return out
    required = {"true_label", "pred_label"}
    if not required.issubset(preds_df.columns):
        return out

    m = _metrics_from_preds_df(preds_df)
    out[f"{split_name}_accuracy"] = m["accuracy"]
    out[f"{split_name}_balanced_accuracy"] = m["balanced_accuracy"]
    out[f"{split_name}_macro_f1"] = m["macro_f1"]
    out[f"{split_name}_mcc"] = m["mcc"]
    return out


def _extract_split_from_summary(summary_df: pd.DataFrame, split: str, metric: str) -> float:
    if summary_df is None or summary_df.empty:
        return float("nan")
    sub = summary_df.loc[summary_df["split"] == split, metric]
    if len(sub) == 0:
        return float("nan")
    try:
        return float(sub.iloc[0])
    except Exception:
        return float("nan")


def _collect_training_run(run_dir: Path) -> Dict[str, Any]:
    config_path = run_dir / "config.json"
    metrics_path = run_dir / "metrics.json"
    summary_path = run_dir / "summary.csv"
    history_path = run_dir / "history.json"
    ckpt_path = run_dir / "best_model.pt"

    if not config_path.exists():
        return {}

    cfg = _load_json(config_path)
    metrics = _load_json(metrics_path) if metrics_path.exists() else {}
    summary_df = pd.read_csv(summary_path) if summary_path.exists() else None
    history = _load_json(history_path) if history_path.exists() else []

    train_root = cfg.get("train_root", cfg.get("imp1_root"))
    val_root = cfg.get("val_root", cfg.get("imp1_root"))
    internal_test_root = cfg.get("internal_test_root", cfg.get("imp_test_root"))
    external_test_root = cfg.get("external_test_root", cfg.get("apss_root"))
    tile_pooling = cfg.get("tile_pooling")
    if tile_pooling is None and cfg.get("mode") == "patch":
        tile_pooling = "abmil"

    row: Dict[str, Any] = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "mode": cfg.get("mode"),
        "tile_pooling": tile_pooling,
        "feature_subdir": cfg.get("feature_subdir"),
        "train_root": train_root,
        "val_root": val_root,
        "internal_test_root": internal_test_root,
        "external_test_root": external_test_root,
        "seed": cfg.get("seed"),
        "epochs_cfg": cfg.get("epochs"),
        "batch_size": cfg.get("batch_size"),
        "eval_batch_size": cfg.get("eval_batch_size"),
        "lr": cfg.get("lr"),
        "weight_decay": cfg.get("weight_decay"),
        "best_checkpoint_exists": ckpt_path.exists(),
        "has_metrics_json": metrics_path.exists(),
        "has_summary_csv": summary_path.exists(),
        "has_history_json": history_path.exists(),
    }

    if metrics:
        row.update(
            {
                "best_val_macro_f1": metrics.get("best_val_macro_f1", float("nan")),
                "val_macro_f1": _safe_metric(metrics, "val", "macro_f1"),
                "val_accuracy": _safe_metric(metrics, "val", "accuracy"),
                "val_balanced_accuracy": _safe_metric(metrics, "val", "balanced_accuracy"),
                "val_mcc": float("nan"),
                "internal_test_macro_f1": _safe_metric(metrics, "internal_test", "macro_f1"),
                "internal_test_accuracy": _safe_metric(metrics, "internal_test", "accuracy"),
                "internal_test_balanced_accuracy": _safe_metric(metrics, "internal_test", "balanced_accuracy"),
                "internal_test_mcc": float("nan"),
                "external_test_macro_f1": _safe_metric(metrics, "external_test", "macro_f1"),
                "external_test_accuracy": _safe_metric(metrics, "external_test", "accuracy"),
                "external_test_balanced_accuracy": _safe_metric(metrics, "external_test", "balanced_accuracy"),
                "external_test_mcc": float("nan"),
            }
        )
    else:
        row.update(
            {
                "best_val_macro_f1": float("nan"),
                "val_macro_f1": _extract_split_from_summary(summary_df, "val", "macro_f1"),
                "val_accuracy": _extract_split_from_summary(summary_df, "val", "accuracy"),
                "val_balanced_accuracy": _extract_split_from_summary(summary_df, "val", "balanced_accuracy"),
                "val_mcc": float("nan"),
                "internal_test_macro_f1": _extract_split_from_summary(summary_df, "internal_test", "macro_f1"),
                "internal_test_accuracy": _extract_split_from_summary(summary_df, "internal_test", "accuracy"),
                "internal_test_balanced_accuracy": _extract_split_from_summary(
                    summary_df, "internal_test", "balanced_accuracy"
                ),
                "internal_test_mcc": float("nan"),
                "external_test_macro_f1": _extract_split_from_summary(summary_df, "external_test", "macro_f1"),
                "external_test_accuracy": _extract_split_from_summary(summary_df, "external_test", "accuracy"),
                "external_test_balanced_accuracy": _extract_split_from_summary(
                    summary_df, "external_test", "balanced_accuracy"
                ),
                "external_test_mcc": float("nan"),
            }
        )

    row["trained_epochs"] = len(history) if isinstance(history, list) else np.nan
    row.update(_split_metrics_from_preds(run_dir / "preds_val.csv", "val"))
    row.update(_split_metrics_from_preds(run_dir / "preds_internal_test.csv", "internal_test"))
    row.update(_split_metrics_from_preds(run_dir / "preds_external_test.csv", "external_test"))
    row.update(_external_no_ssl_metrics_from_preds(run_dir / "preds_external_test.csv"))

    prediction_dirs = sorted([p for p in run_dir.glob("predictions_*") if p.is_dir()])
    row["num_prediction_runs"] = len(prediction_dirs)
    row["latest_prediction_dir"] = str(prediction_dirs[-1]) if prediction_dirs else ""
    return row


def collect_training_runs(runs_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not runs_root.exists():
        return pd.DataFrame()

    for p in sorted(runs_root.iterdir()):
        if not p.is_dir():
            continue
        row = _collect_training_run(p)
        if row:
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def collect_inference_runs(runs_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if not runs_root.exists():
        return pd.DataFrame()

    for metrics_path in sorted(runs_root.glob("**/predictions_*/metrics.json")):
        pred_dir = metrics_path.parent
        run_dir = pred_dir.parent
        data = _load_json(metrics_path)
        split_metrics = data.get("metrics", {})
        row = {
            "run_name": run_dir.name,
            "prediction_dir": str(pred_dir),
            "mode": data.get("mode"),
            "tile_pooling": data.get("tile_pooling"),
            "feature_subdir": data.get("feature_subdir"),
            "checkpoint": data.get("checkpoint"),
            "val_macro_f1": _safe_metric(split_metrics, "val", "macro_f1"),
            "internal_test_macro_f1": _safe_metric(split_metrics, "internal_test", "macro_f1"),
            "external_test_macro_f1": _safe_metric(split_metrics, "external_test", "macro_f1"),
            "val_accuracy": _safe_metric(split_metrics, "val", "accuracy"),
            "internal_test_accuracy": _safe_metric(split_metrics, "internal_test", "accuracy"),
            "external_test_accuracy": _safe_metric(split_metrics, "external_test", "accuracy"),
            "val_mcc": float("nan"),
            "internal_test_mcc": float("nan"),
            "external_test_mcc": float("nan"),
        }
        row.update(_split_metrics_from_preds(pred_dir / "preds_val.csv", "val"))
        row.update(_split_metrics_from_preds(pred_dir / "preds_internal_test.csv", "internal_test"))
        row.update(_split_metrics_from_preds(pred_dir / "preds_external_test.csv", "external_test"))
        row.update(_external_no_ssl_metrics_from_preds(pred_dir / "preds_external_test.csv"))
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _save_markdown_report(
    report_path: Path,
    train_df: pd.DataFrame,
    infer_df: pd.DataFrame,
    top_df: pd.DataFrame,
    grouped_df: pd.DataFrame,
) -> None:
    lines: List[str] = []
    lines.append("# Final Training Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Training runs found: {len(train_df)}")
    lines.append(f"- Inference prediction runs found: {len(infer_df)}")
    lines.append("")

    if len(train_df) == 0:
        lines.append("No training runs were found.")
        report_path.write_text("\n".join(lines))
        return

    cols = [
        "run_name",
        "mode",
        "tile_pooling",
        "feature_subdir",
        "val_macro_f1",
        "internal_test_macro_f1",
        "external_test_macro_f1",
        "external_test_mcc",
        "external_test_no_ssl_macro_f1",
        "val_accuracy",
        "internal_test_accuracy",
        "external_test_accuracy",
        "external_test_no_ssl_mcc",
        "external_test_no_ssl_accuracy",
    ]

    lines.append("## Top Runs (by external_test_macro_f1)")
    lines.append("")
    lines.append(top_df[cols].to_markdown(index=False))
    lines.append("")

    if len(grouped_df) > 0:
        lines.append("## Aggregated by Mode + Feature")
        lines.append("")
        lines.append(grouped_df.to_markdown(index=False))
        lines.append("")

    missing_metrics = train_df.loc[
        ~(train_df["has_metrics_json"].astype(bool) | train_df["has_summary_csv"].astype(bool)), "run_name"
    ].tolist()
    if missing_metrics:
        lines.append("## Runs Missing Metrics")
        lines.append("")
        for name in missing_metrics:
            lines.append(f"- {name}")
        lines.append("")

    report_path.write_text("\n".join(lines))


def _algo_label(row: pd.Series) -> str:
    mode = str(row.get("mode", ""))
    pooling = str(row.get("tile_pooling", ""))
    feat = str(row.get("feature_subdir", ""))
    feat_name = feat.split("/")[-1] if feat else ""
    if mode == "patch":
        return f"patch/{pooling}/{feat_name}"
    if mode == "slide":
        return f"slide/{feat_name}"
    return f"{mode}/{pooling}/{feat_name}".strip("/")


def build_report(train_df: pd.DataFrame, infer_df: pd.DataFrame, output_dir: Path, top_k: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_csv = output_dir / "training_runs.csv"
    infer_csv = output_dir / "inference_runs.csv"
    ranking_csv = output_dir / "ranking_external_macro_f1.csv"
    grouped_csv = output_dir / "grouped_by_mode_feature.csv"
    brief_runs_csv = output_dir / "brief_mcc_per_run.csv"
    brief_algo_csv = output_dir / "brief_mcc_best_per_algorithm.csv"
    report_md = output_dir / "final_report.md"

    if len(train_df) > 0:
        train_df = train_df.sort_values(["external_test_macro_f1", "internal_test_macro_f1"], ascending=False)
        train_df.to_csv(train_csv, index=False)

        top_df = train_df.head(top_k).copy()
        top_df.to_csv(ranking_csv, index=False)

        grouped_df = (
            train_df.groupby(["mode", "tile_pooling", "feature_subdir"], dropna=False)[
                [
                    "val_macro_f1",
                    "internal_test_macro_f1",
                    "external_test_macro_f1",
                    "external_test_mcc",
                    "external_test_no_ssl_macro_f1",
                    "external_test_no_ssl_mcc",
                ]
            ]
            .agg(["count", "mean", "std"])
            .reset_index()
        )
        grouped_df.columns = ["_".join(c).strip("_") for c in grouped_df.columns.to_flat_index()]
        grouped_df.to_csv(grouped_csv, index=False)

        # Compact, easy-to-read MCC report at run level.
        brief_df = train_df.copy()
        brief_df["algorithm"] = brief_df.apply(_algo_label, axis=1)
        brief_df = brief_df[
            [
                "run_name",
                "algorithm",
                "val_mcc",
                "internal_test_mcc",
                "external_test_mcc",
                "external_test_no_ssl_mcc",
            ]
        ].sort_values(["external_test_no_ssl_mcc", "external_test_mcc"], ascending=False)
        brief_df.to_csv(brief_runs_csv, index=False)

        # One best run per algorithm by external no-SSL MCC.
        brief_best_df = (
            brief_df.sort_values(["algorithm", "external_test_no_ssl_mcc", "external_test_mcc"], ascending=[True, False, False])
            .groupby("algorithm", as_index=False)
            .first()
            .sort_values(["external_test_no_ssl_mcc", "external_test_mcc"], ascending=False)
        )
        brief_best_df.to_csv(brief_algo_csv, index=False)
    else:
        top_df = pd.DataFrame()
        grouped_df = pd.DataFrame()
        pd.DataFrame().to_csv(train_csv, index=False)
        pd.DataFrame().to_csv(ranking_csv, index=False)
        pd.DataFrame().to_csv(grouped_csv, index=False)
        pd.DataFrame().to_csv(brief_runs_csv, index=False)
        pd.DataFrame().to_csv(brief_algo_csv, index=False)

    if len(infer_df) > 0:
        infer_df = infer_df.sort_values(["external_test_macro_f1", "internal_test_macro_f1"], ascending=False)
        infer_df.to_csv(infer_csv, index=False)
    else:
        pd.DataFrame().to_csv(infer_csv, index=False)

    _save_markdown_report(
        report_path=report_md,
        train_df=train_df,
        infer_df=infer_df,
        top_df=top_df,
        grouped_df=grouped_df,
    )


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else runs_root / "reports" / timestamp

    train_df = collect_training_runs(runs_root)
    infer_df = collect_inference_runs(runs_root)
    build_report(train_df, infer_df, output_dir=output_dir, top_k=args.top_k)

    print(f"Training runs collected: {len(train_df)}")
    print(f"Inference runs collected: {len(infer_df)}")
    print(f"Report saved to: {output_dir}")


if __name__ == "__main__":
    main()
