from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef

from compute_fm_shift_repeated import _compute_metrics, _stratified_indices
from metrics import PURE_SSL_REGEX, filter_available, read_labels


DEFAULT_RUNS_ROOT = "/data/projects/D34H/git_repo/repo-shahry/wp-1/Classification_Trident/Codes/runs"
DEFAULT_INTERNAL_ROOT = (
    "/data/projects/D34H/git_repo/repo-shahry/wp-1/Classification_Trident/embeddings/PRISM/IMP_TEST"
)
DEFAULT_EXTERNAL_ROOT = "/data/projects/D34H/git_repo/repo-shahry/wp-1/Classification_Trident/embeddings/PRISM/APSS"


FM_SUBDIRS = [
    "20x_512px_0px_overlap/features_conch_v15",
    "20x_224px_0px_overlap/features_virchow",
    "20x_224px_0px_overlap/features_virchow2",
    "20x_224px_0px_overlap/features_phikon_v2",
    "20x_256px_0px_overlap/features_uni_v2",
]


DEFAULT_LAYER_NAMES = ["pre_attention_mean", "post_attention", "penultimate"]
TOPK_LAYER_NAME = "topk64_positive_evidence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repeated internal/external domain-shift metrics on trained ABMIL internal representations. "
            "Extracts ABMIL internal representations and computes repeated domain-shift metrics."
        )
    )
    parser.add_argument("--runs-root", type=str, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--run-dir", action="append", default=None, help="ABMIL run directory. Can be repeated.")
    parser.add_argument("--run-glob", type=str, default="*abmil*", help="Glob under --runs-root when --run-dir is unset.")
    parser.add_argument("--checkpoint-name", type=str, default="best_model.pt")
    parser.add_argument(
        "--internal-root",
        type=str,
        default=None,
        help="Override internal test embedding root. If unset, uses config.json internal_test_root or the repo default.",
    )
    parser.add_argument(
        "--external-root",
        type=str,
        default=None,
        help="Override external test embedding root. If unset, uses config.json external_test_root or the repo default.",
    )
    parser.add_argument("--out-csv", type=str, default=None, help="Per-repeat long CSV path.")
    parser.add_argument("--out-summary-csv", type=str, default=None, help="Summary CSV path.")
    parser.add_argument("--out-stability-csv", type=str, default=None, help="Best-frequency CSV path.")
    parser.add_argument("--out-json", type=str, default=None, help="Run metadata JSON path.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--n-repeats", type=int, default=20)
    parser.add_argument(
        "--target-per-domain",
        type=int,
        default=0,
        help="If >0, uses this matched count per domain per repeat. If 0, uses all external no-SSL.",
    )
    parser.add_argument(
        "--sample-external",
        action="store_true",
        help="If set, sample external each repeat too; otherwise external is fixed full no-SSL set.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--knn-k", type=int, default=15)

    parser.add_argument("--num-patches", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Extraction batch size. If unset, uses config.json eval_batch_size, then batch_size, then 16.",
    )
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--no-gated", action="store_true", default=None)
    parser.add_argument("--feature-subdir", type=str, default=None)

    parser.add_argument("--sigma-sample-size", type=int, default=512)
    parser.add_argument("--energy-sample-size", type=int, default=2048)
    parser.add_argument("--energy-chunk-size", type=int, default=512)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument(
        "--layers",
        type=str,
        default="pre_attention_mean,post_attention,penultimate",
        help=(
            "Comma-separated layer names to compute. "
            f"Allowed: {', '.join(DEFAULT_LAYER_NAMES + [TOPK_LAYER_NAME])}."
        ),
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=64,
        help=f"K for {TOPK_LAYER_NAME}. Default matches paper setting (64).",
    )
    return parser.parse_args()


def _resolve_layer_names(args: argparse.Namespace) -> list[str]:
    allowed = set(DEFAULT_LAYER_NAMES + [TOPK_LAYER_NAME])
    raw = [x.strip() for x in str(args.layers).split(",") if x.strip()]
    if not raw:
        raise ValueError("--layers resolved to empty list.")
    unknown = [x for x in raw if x not in allowed]
    if unknown:
        raise ValueError(f"Unknown layer names in --layers: {unknown}. Allowed: {sorted(allowed)}")
    return raw


def _resolve_out_paths(args: argparse.Namespace, runs_root: Path) -> tuple[Path, Path, Path, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_csv:
        out_csv = Path(args.out_csv)
    else:
        out_csv = runs_root / "reports" / ts / "abmil_layer_shift_metrics_repeated.csv"
    if args.out_summary_csv:
        out_summary_csv = Path(args.out_summary_csv)
    else:
        out_summary_csv = out_csv.parent / "abmil_layer_shift_metrics_repeated_summary.csv"
    if args.out_stability_csv:
        out_stability_csv = Path(args.out_stability_csv)
    else:
        out_stability_csv = out_csv.parent / "abmil_layer_shift_metrics_repeated_best_frequency.csv"
    if args.out_json:
        out_json = Path(args.out_json)
    else:
        out_json = out_csv.parent / "abmil_layer_shift_metrics_repeated_meta.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    return out_csv, out_summary_csv, out_stability_csv, out_json


def _stable_int(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _root_from_config(
    override: str | None,
    cfg: dict[str, Any],
    new_key: str,
    old_key: str,
    default: str,
) -> Path:
    if override is not None:
        return Path(override)
    if cfg.get(new_key) is not None:
        return Path(str(cfg[new_key]))
    if cfg.get(old_key) is not None:
        return Path(str(cfg[old_key]))
    return Path(default)


def _eval_batch_size(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    if args.batch_size is not None:
        return int(args.batch_size)
    if cfg.get("eval_batch_size") is not None:
        return int(cfg["eval_batch_size"])
    if cfg.get("batch_size") is not None:
        return int(cfg["batch_size"])
    return 16


def _discover_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir:
        run_dirs = [Path(p) for p in args.run_dir]
    else:
        run_dirs = sorted(p for p in Path(args.runs_root).glob(args.run_glob) if p.is_dir())

    out = []
    for run_dir in run_dirs:
        if (run_dir / args.checkpoint_name).exists():
            out.append(run_dir)
        else:
            print(f"[skip] missing checkpoint: {run_dir / args.checkpoint_name}")
    if not out:
        raise ValueError("No ABMIL run directories with checkpoints were found.")
    return out


def _infer_feature_subdir(run_dir: Path, cfg: dict[str, Any], override: str | None) -> str:
    if override is not None:
        return override
    if cfg.get("feature_subdir"):
        return str(cfg["feature_subdir"])

    name = run_dir.name.lower()
    candidates = sorted(FM_SUBDIRS, key=len, reverse=True)
    for subdir in candidates:
        fm_tail = Path(subdir).name.replace("features_", "")
        if fm_tail.lower() in name:
            return subdir
    raise ValueError(
        f"Could not infer feature_subdir for {run_dir}. Provide config.json or pass --feature-subdir."
    )


def _state_dict_from_checkpoint(checkpoint_path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


def _infer_model_params(
    cfg: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    input_dim: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    classifier0 = state_dict.get("classifier.0.weight")
    classifier3 = state_dict.get("classifier.3.weight")
    if classifier3 is None:
        raise ValueError("Could not infer num_classes: missing classifier.3.weight in checkpoint.")

    hidden_dim = args.hidden_dim or cfg.get("hidden_dim")
    if hidden_dim is None and classifier0 is not None:
        hidden_dim = int(classifier0.shape[0])
    hidden_dim = int(hidden_dim or 256)

    n_heads = args.n_heads or cfg.get("n_heads")
    head_dim = args.head_dim or cfg.get("head_dim")
    attention_keys = [k for k in state_dict if ".attention_heads." in k and k.endswith(".weight")]
    if n_heads is None:
        head_ids = set()
        for key in attention_keys:
            parts = key.split(".attention_heads.", 1)[1].split(".")
            if parts and parts[0].isdigit():
                head_ids.add(int(parts[0]))
        n_heads = max(1, len(head_ids))
    if head_dim is None:
        for key in attention_keys:
            tensor = state_dict[key]
            if tensor.ndim == 2:
                head_dim = int(tensor.shape[0])
                break

    no_gated = args.no_gated
    if no_gated is None:
        no_gated = bool(cfg.get("no_gated", False))

    return {
        "mode": "patch",
        "tile_pooling": "abmil",
        "input_dim": int(input_dim),
        "num_classes": int(classifier3.shape[0]),
        "hidden_dim": hidden_dim,
        "dropout": float(args.dropout if args.dropout is not None else cfg.get("dropout", 0.1)),
        "n_heads": int(n_heads or 1),
        "head_dim": int(head_dim or 512),
        "no_gated": bool(no_gated),
    }


def _build_abmil_model(model_params: dict[str, Any]) -> torch.nn.Module:
    from predict import build_model

    return build_model(**model_params)


def _read_patch_features(h5_path: Path) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        feats = np.asarray(f["features"][()])
    if feats.ndim != 2:
        raise ValueError(f"ABMIL patch mode expects 2D features, got {feats.shape} in {h5_path}")
    if feats.shape[0] == 0:
        raise ValueError(f"No patch features in: {h5_path}")
    return feats


def _sample_patches(feats: np.ndarray, slide_name: str, seed: int, num_patches: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + _stable_int(slide_name))
    replace = feats.shape[0] < int(num_patches)
    idx = rng.choice(feats.shape[0], size=int(num_patches), replace=replace)
    return feats[idx]


def _build_label_frame(dataset_root: Path, feature_subdir: str, external_no_ssl: bool) -> pd.DataFrame:
    labels_df = read_labels(dataset_root)
    if external_no_ssl:
        is_pure_ssl = labels_df["slide_name"].str.contains(PURE_SSL_REGEX, case=False, regex=True, na=False)
        labels_df = labels_df.loc[~is_pure_ssl].copy()
    labels_df = filter_available(labels_df, dataset_root / feature_subdir)
    if labels_df.empty:
        raise ValueError(f"No slides with features in: {dataset_root / feature_subdir}")
    out = labels_df[["slide_name", "slide_label"]].copy().reset_index(drop=True)
    out = out.rename(columns={"slide_label": "label"})
    out["row_idx"] = np.arange(len(out), dtype=np.int64)
    return out


def _attach_predictions(labels_df: pd.DataFrame, preds_path: Path) -> pd.DataFrame:
    if not preds_path.exists():
        raise FileNotFoundError(f"Missing predictions CSV: {preds_path}")
    preds_df = pd.read_csv(preds_path)
    required = {"slide_name", "true_label", "pred_label"}
    missing = required - set(preds_df.columns)
    if missing:
        raise ValueError(f"{preds_path} missing columns: {sorted(missing)}")

    preds_df = preds_df[["slide_name", "true_label", "pred_label"]].copy()
    preds_df["slide_name"] = preds_df["slide_name"].astype(str).str.strip()
    preds_df["true_label"] = preds_df["true_label"].astype(int)
    preds_df["pred_label"] = preds_df["pred_label"].astype(int)

    merged = labels_df.merge(preds_df, on="slide_name", how="left", validate="one_to_one")
    missing_preds = merged["pred_label"].isna()
    if missing_preds.any():
        examples = merged.loc[missing_preds, "slide_name"].head(5).tolist()
        raise ValueError(f"{preds_path} is missing predictions for {int(missing_preds.sum())} slides, e.g. {examples}")
    if not np.array_equal(merged["label"].to_numpy(dtype=np.int64), merged["true_label"].to_numpy(dtype=np.int64)):
        raise ValueError(f"Label mismatch between filtered labels and predictions in: {preds_path}")
    return merged


def _mcc_for_subset(df: pd.DataFrame, idx: np.ndarray) -> float:
    y_true = df["true_label"].to_numpy(dtype=np.int64)[idx]
    y_pred = df["pred_label"].to_numpy(dtype=np.int64)[idx]
    return float(matthews_corrcoef(y_true, y_pred))


def _encoder_modules(model: torch.nn.Module) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    encoder = model.feature_encoder
    container = getattr(encoder, "model", encoder)
    pre = container["pre_attention_layers"] if isinstance(container, torch.nn.ModuleDict) else container.pre_attention_layers
    pooler = container["image_pooler"] if isinstance(container, torch.nn.ModuleDict) else container.image_pooler
    post = container["post_attention_layers"] if isinstance(container, torch.nn.ModuleDict) else container.post_attention_layers
    return pre, pooler, post


@torch.no_grad()
def _extract_batch_layers(
    model: torch.nn.Module,
    features: torch.Tensor,
    layer_names: list[str],
    topk: int,
) -> dict[str, torch.Tensor]:
    pre_layers, image_pooler, post_layers = _encoder_modules(model)
    pre_patch = pre_layers(features)
    out: dict[str, torch.Tensor] = {}

    if "pre_attention_mean" in layer_names:
        out["pre_attention_mean"] = pre_patch.mean(dim=1)

    need_post = ("post_attention" in layer_names) or ("penultimate" in layer_names)
    post_attention = None
    if need_post:
        pooled = image_pooler(pre_patch)
        if isinstance(pooled, tuple):
            pooled = pooled[0]
        if pooled.ndim == 3:
            if pooled.shape[1] != 1:
                raise ValueError(f"Expected pooled shape [B,1,F], got {tuple(pooled.shape)}")
            post_attention = pooled[:, 0, :]
        else:
            post_attention = pooled
        if "post_attention" in layer_names:
            out["post_attention"] = post_attention
        if "penultimate" in layer_names:
            out["penultimate"] = post_layers(post_attention)

    if TOPK_LAYER_NAME in layer_names:
        # Get raw attention from the actual ABMIL forward path.
        _, raw_attn = model(features, return_raw_attention=True)
        if isinstance(raw_attn, (tuple, list)):
            raw_attn = raw_attn[0]
        if not torch.is_tensor(raw_attn):
            raw_attn = torch.as_tensor(raw_attn, device=pre_patch.device)

        attn = raw_attn
        bsz, n_tok = pre_patch.shape[0], pre_patch.shape[1]

        # Normalize attention tensor to [B, N].
        # Seen shapes include:
        # [B,H,N], [B,N,H], [B,H,N,1], [B,1,H,N], [1,B,N], [H,B,N], ...
        if attn.ndim >= 3:
            # Remove trivial singleton dims first.
            attn = attn.squeeze()

        if attn.ndim == 1:
            # [N] -> [1,N], valid only if B==1
            if bsz != 1 or attn.shape[0] != n_tok:
                raise ValueError(
                    f"Unexpected 1D attention after squeeze: {tuple(attn.shape)} for pre_patch {tuple(pre_patch.shape)}"
                )
            attn = attn.unsqueeze(0)

        elif attn.ndim == 2:
            # Could already be [B,N], or transposed [N,B]
            if attn.shape == (bsz, n_tok):
                pass
            elif attn.shape == (n_tok, bsz):
                attn = attn.transpose(0, 1)
            elif bsz == 1 and attn.shape[0] == n_tok:
                attn = attn.unsqueeze(0)
            else:
                raise ValueError(
                    f"Unexpected 2D attention shape: {tuple(attn.shape)} for pre_patch {tuple(pre_patch.shape)}"
                )

        elif attn.ndim == 3:
            # Find which axis corresponds to batch and token.
            shape = tuple(attn.shape)
            idx_b = [i for i, s in enumerate(shape) if s == bsz]
            idx_n = [i for i, s in enumerate(shape) if s == n_tok]
            if not idx_b or not idx_n:
                raise ValueError(
                    f"Cannot infer batch/token axes from attention {shape} with B={bsz}, N={n_tok}"
                )
            # Prefer distinct axes; fallback to first.
            chosen = None
            for ib in idx_b:
                for in_ in idx_n:
                    if ib != in_:
                        chosen = (ib, in_)
                        break
                if chosen is not None:
                    break
            if chosen is None:
                chosen = (idx_b[0], idx_n[0])

            ib, in_ = chosen
            ih = [i for i in (0, 1, 2) if i not in (ib, in_)][0]
            # Reorder to [B, H, N] then average heads.
            attn = attn.permute(ib, ih, in_).contiguous()
            attn = attn.mean(dim=1)

        else:
            raise ValueError(f"Unexpected attention shape: {tuple(raw_attn.shape)} -> {tuple(attn.shape)}")

        if attn.shape[0] != pre_patch.shape[0] or attn.shape[1] != pre_patch.shape[1]:
            raise ValueError(
                f"Attention shape {tuple(attn.shape)} not aligned with pre_patch {tuple(pre_patch.shape)}"
            )

        k = max(1, min(int(topk), int(pre_patch.shape[1])))
        top_idx = torch.topk(attn, k=k, dim=1, largest=True, sorted=False).indices  # [B, k]
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, pre_patch.shape[-1])       # [B, k, D]
        top_tokens = torch.gather(pre_patch, dim=1, index=gather_idx)                 # [B, k, D]
        out[TOPK_LAYER_NAME] = top_tokens.mean(dim=1)                                 # [B, D]

    return out


def _build_layer_matrices(
    dataset_root: Path,
    feature_subdir: str,
    model: torch.nn.Module,
    device: torch.device,
    seed: int,
    num_patches: int,
    batch_size: int,
    external_no_ssl: bool,
    layer_names: list[str],
    topk: int,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    labels_df = _build_label_frame(dataset_root, feature_subdir, external_no_ssl=external_no_ssl)
    feats_dir = dataset_root / feature_subdir
    rows_by_layer: dict[str, list[np.ndarray]] = {name: [] for name in layer_names}

    model.eval()
    for start in range(0, len(labels_df), int(batch_size)):
        batch_df = labels_df.iloc[start : start + int(batch_size)]
        batch = []
        for r in batch_df.itertuples(index=False):
            feats = _read_patch_features(feats_dir / f"{r.slide_name}.h5")
            feats = _sample_patches(feats, r.slide_name, seed=seed, num_patches=num_patches)
            batch.append(feats.astype(np.float32, copy=False))

        x = torch.as_tensor(np.stack(batch, axis=0), dtype=torch.float32, device=device)
        layer_tensors = _extract_batch_layers(model, x, layer_names=layer_names, topk=topk)
        for layer_name, tensor in layer_tensors.items():
            rows_by_layer[layer_name].append(tensor.detach().cpu().numpy().astype(np.float64, copy=False))

    layer_matrices = {}
    for layer_name, chunks in rows_by_layer.items():
        x_layer = np.concatenate(chunks, axis=0)
        if x_layer.ndim != 2:
            raise ValueError(f"Expected 2D matrix for {layer_name}, got shape={x_layer.shape}")
        layer_matrices[layer_name] = x_layer
    return labels_df, layer_matrices


def _infer_input_dim(dataset_root: Path, feature_subdir: str) -> int:
    labels_df = filter_available(read_labels(dataset_root), dataset_root / feature_subdir)
    if labels_df.empty:
        raise ValueError(f"No slides with features in: {dataset_root / feature_subdir}")
    feats = _read_patch_features(dataset_root / feature_subdir / f"{labels_df.iloc[0]['slide_name']}.h5")
    return int(feats.shape[1])


def _write_summary_outputs(
    long_df: pd.DataFrame,
    out_summary_csv: Path,
    out_stability_csv: Path,
    n_repeats: int,
) -> None:
    metric_cols = ["mmd_raw", "fid_raw", "coral_raw", "energy_raw", "knn_domain_mixing"]
    perf_cols = ["mcc_internal", "mcc_external", "mcc_drop"]
    agg = {}
    for m in metric_cols + perf_cols:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_std"] = (m, "std")
        agg[f"{m}_median"] = (m, "median")
        agg[f"{m}_q25"] = (m, lambda s: float(np.nanquantile(s, 0.25)))
        agg[f"{m}_q75"] = (m, lambda s: float(np.nanquantile(s, 0.75)))
        agg[f"{m}_q05"] = (m, lambda s: float(np.nanquantile(s, 0.05)))
        agg[f"{m}_q95"] = (m, lambda s: float(np.nanquantile(s, 0.95)))

    summary_df = (
        long_df.groupby(["run_name", "feature_subdir", "layer_name"], as_index=False)
        .agg(n_repeats=("repetition", "nunique"), **agg)
        .sort_values(["run_name", "layer_name"])
        .reset_index(drop=True)
    )
    summary_df.to_csv(out_summary_csv, index=False)
    print(f"Wrote: {out_summary_csv} (rows={len(summary_df)})")

    maximize = {"knn_domain_mixing"}
    stability_rows = []
    for layer_name, layer_df in long_df.groupby("layer_name", sort=True):
        for m in metric_cols:
            best_counts: dict[str, float] = {}
            for _, g in layer_df.groupby("repetition", sort=True):
                best_val = g[m].max() if m in maximize else g[m].min()
                winners = g.loc[g[m] == best_val, "run_name"].tolist()
                for w in winners:
                    best_counts[w] = best_counts.get(w, 0.0) + (1.0 / len(winners))

            for run_name in sorted(layer_df["run_name"].unique().tolist()):
                c = float(best_counts.get(run_name, 0.0))
                stability_rows.append(
                    {
                        "layer_name": layer_name,
                        "metric_name": m,
                        "run_name": run_name,
                        "best_count": c,
                        "best_fraction": c / float(n_repeats),
                        "n_repeats": int(n_repeats),
                    }
                )

    stability_df = pd.DataFrame(stability_rows).sort_values(
        ["layer_name", "metric_name", "best_count"], ascending=[True, True, False]
    )
    stability_df.to_csv(out_stability_csv, index=False)
    print(f"Wrote: {out_stability_csv} (rows={len(stability_df)})")


def main() -> None:
    args = parse_args()
    layer_names = _resolve_layer_names(args)
    runs_root = Path(args.runs_root)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_csv, out_summary_csv, out_stability_csv, out_json = _resolve_out_paths(args, runs_root)
    print(f"Using device: {device}")

    run_dirs = _discover_run_dirs(args)
    long_rows = []
    run_meta = []

    for run_dir in run_dirs:
        cfg = _load_json(run_dir / "config.json")
        checkpoint_path = run_dir / args.checkpoint_name
        feature_subdir = _infer_feature_subdir(run_dir, cfg, args.feature_subdir)
        num_patches = int(args.num_patches or cfg.get("num_patches", 512))
        batch_size = _eval_batch_size(args, cfg)
        internal_root = _root_from_config(
            args.internal_root,
            cfg,
            new_key="internal_test_root",
            old_key="imp_test_root",
            default=DEFAULT_INTERNAL_ROOT,
        )
        external_root = _root_from_config(
            args.external_root,
            cfg,
            new_key="external_test_root",
            old_key="apss_root",
            default=DEFAULT_EXTERNAL_ROOT,
        )
        input_dim = _infer_input_dim(internal_root, feature_subdir)
        state_dict = _state_dict_from_checkpoint(checkpoint_path, device=device)
        model_params = _infer_model_params(cfg, state_dict, input_dim=input_dim, args=args)

        print(f"[build] {run_dir.name} <- {feature_subdir}")
        model = _build_abmil_model(model_params).to(device)
        model.load_state_dict(state_dict)

        preds_int_path = run_dir / "preds_internal_test.csv"
        preds_ext_path = run_dir / "preds_external_test.csv"
        int_df, int_layers = _build_layer_matrices(
            internal_root,
            feature_subdir,
            model,
            device=device,
            seed=int(args.seed),
            num_patches=num_patches,
            batch_size=batch_size,
            external_no_ssl=False,
            layer_names=layer_names,
            topk=int(args.topk),
        )
        ext_df, ext_layers = _build_layer_matrices(
            external_root,
            feature_subdir,
            model,
            device=device,
            seed=int(args.seed),
            num_patches=num_patches,
            batch_size=batch_size,
            external_no_ssl=True,
            layer_names=layer_names,
            topk=int(args.topk),
        )
        int_df = _attach_predictions(int_df, preds_int_path)
        ext_df = _attach_predictions(ext_df, preds_ext_path)

        n_ext_total = int(len(ext_df))
        n_int_total = int(len(int_df))
        n_match = int(args.target_per_domain) if int(args.target_per_domain) > 0 else int(n_ext_total)
        n_match = min(n_match, n_int_total, n_ext_total)
        if n_match <= 0:
            continue

        y_int = int_df["label"].to_numpy(dtype=np.int64)
        y_ext = ext_df["label"].to_numpy(dtype=np.int64)

        for layer_name in layer_names:
            x_int_all = int_layers[layer_name]
            x_ext_all = ext_layers[layer_name]
            for rep in range(int(args.n_repeats)):
                s = int(args.seed) + 10007 * rep
                idx_int = _stratified_indices(y_int, n_target=n_match, seed=s + 11)

                if args.sample_external:
                    idx_ext = _stratified_indices(y_ext, n_target=n_match, seed=s + 29)
                else:
                    idx_ext = np.arange(n_ext_total, dtype=np.int64)

                mcc_internal = _mcc_for_subset(int_df, idx_int)
                mcc_external = _mcc_for_subset(ext_df, idx_ext)
                metrics = _compute_metrics(
                    x_int=x_int_all[idx_int],
                    x_ext=x_ext_all[idx_ext],
                    device=device,
                    sigma_sample_size=int(args.sigma_sample_size),
                    energy_sample_size=int(args.energy_sample_size),
                    energy_chunk_size=int(args.energy_chunk_size),
                    knn_k=int(args.knn_k),
                    eps=float(args.eps),
                    seed=s + 101,
                )

                long_rows.append(
                    {
                        "run_name": run_dir.name,
                        "run_dir": str(run_dir),
                        "feature_subdir": feature_subdir,
                        "layer_name": layer_name,
                        "repetition": rep,
                        "n_internal_total": n_int_total,
                        "n_external_no_ssl_total": n_ext_total,
                        "n_internal_sample": int(len(idx_int)),
                        "n_external_sample": int(len(idx_ext)),
                        "mcc_internal": mcc_internal,
                        "mcc_external": mcc_external,
                        "mcc_drop": mcc_internal - mcc_external,
                        **metrics,
                    }
                )

        run_meta.append(
            {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "checkpoint": str(checkpoint_path),
                "preds_internal_test": str(preds_int_path),
                "preds_external_test": str(preds_ext_path),
                "feature_subdir": feature_subdir,
                "internal_root": str(internal_root),
                "external_root": str(external_root),
                "num_patches": num_patches,
                "batch_size": batch_size,
                "model_params": model_params,
            }
        )

    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        raise ValueError("No rows computed.")
    long_df = long_df.sort_values(["run_name", "layer_name", "repetition"]).reset_index(drop=True)
    long_df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv} (rows={len(long_df)})")

    _write_summary_outputs(
        long_df=long_df,
        out_summary_csv=out_summary_csv,
        out_stability_csv=out_stability_csv,
        n_repeats=int(args.n_repeats),
    )

    meta = {
        "internal_root_override": args.internal_root,
        "external_root_override": args.external_root,
        "default_internal_root": DEFAULT_INTERNAL_ROOT,
        "default_external_root": DEFAULT_EXTERNAL_ROOT,
        "runs_root": str(runs_root),
        "run_glob": str(args.run_glob),
        "layers": layer_names,
        "topk": int(args.topk),
        "n_repeats": int(args.n_repeats),
        "target_per_domain": int(args.target_per_domain),
        "sample_external": bool(args.sample_external),
        "seed": int(args.seed),
        "knn_k": int(args.knn_k),
        "sigma_sample_size": int(args.sigma_sample_size),
        "energy_sample_size": int(args.energy_sample_size),
        "energy_chunk_size": int(args.energy_chunk_size),
        "eps": float(args.eps),
        "device": str(device),
        "runs": run_meta,
        "out_csv": str(out_csv),
        "out_summary_csv": str(out_summary_csv),
        "out_stability_csv": str(out_stability_csv),
    }
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
