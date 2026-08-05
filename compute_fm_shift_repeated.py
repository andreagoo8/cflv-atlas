from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

from metrics import (
    PURE_SSL_REGEX,
    coral_distance,
    energy_distance,
    filter_available,
    frechet_gaussian,
    mmd_raw_on_arrays,
    normalize_by_internal,
    read_labels,
    vector_from_h5,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "FM-level repeated domain-shift metrics. Builds one embedding per slide for each FM, "
            "then repeats matched-size stratified internal/external sampling and computes metrics."
        )
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default="/data/projects/D34H/git_repo/repo-shahry/wp-1/Classification_Trident/Codes/runs",
    )
    parser.add_argument(
        "--internal-root",
        type=str,
        default="/data/projects/D34H/git_repo/repo-shahry/wp-1/Classification_Trident/embeddings/PRISM/IMP_TEST",
    )
    parser.add_argument(
        "--external-root",
        type=str,
        default="/data/projects/D34H/git_repo/repo-shahry/wp-1/Classification_Trident/embeddings/PRISM/APSS",
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

    parser.add_argument("--sigma-sample-size", type=int, default=512)
    parser.add_argument("--energy-sample-size", type=int, default=2048)
    parser.add_argument("--energy-chunk-size", type=int, default=512)
    parser.add_argument("--eps", type=float, default=1e-12)
    return parser.parse_args()


FM_SUBDIRS = [
    "20x_512px_0px_overlap/features_conch_v15",
    "20x_224px_0px_overlap/features_virchow",
    "20x_224px_0px_overlap/features_virchow2",
    "20x_224px_0px_overlap/features_phikon_v2",
    "20x_256px_0px_overlap/features_uni_v2",
    "20x_224px_0px_overlap/slide_features_prism",
    "20x_512px_0px_overlap/slide_features_titan",
]


def _resolve_out_paths(args: argparse.Namespace, runs_root: Path) -> tuple[Path, Path, Path, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_csv:
        out_csv = Path(args.out_csv)
    else:
        out_csv = runs_root / "reports" / ts / "fm_shift_metrics_repeated.csv"
    if args.out_summary_csv:
        out_summary_csv = Path(args.out_summary_csv)
    else:
        out_summary_csv = out_csv.parent / "fm_shift_metrics_repeated_summary.csv"
    if args.out_stability_csv:
        out_stability_csv = Path(args.out_stability_csv)
    else:
        out_stability_csv = out_csv.parent / "fm_shift_metrics_repeated_best_frequency.csv"
    if args.out_json:
        out_json = Path(args.out_json)
    else:
        out_json = out_csv.parent / "fm_shift_metrics_repeated_meta.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    return out_csv, out_summary_csv, out_stability_csv, out_json


def _feature_kind(feature_subdir: str) -> tuple[str, str]:
    if "slide_features_" in feature_subdir:
        return "slide", "abmil"
    return "patch", "mean"


def _fm_name_from_subdir(feature_subdir: str) -> str:
    tail = Path(feature_subdir).name
    if tail.startswith("features_"):
        return tail.replace("features_", "")
    if tail.startswith("slide_features_"):
        return tail.replace("slide_features_", "")
    return tail


def _build_fm_matrix(dataset_root: Path, feature_subdir: str, external_no_ssl: bool) -> pd.DataFrame:
    labels_df = read_labels(dataset_root)
    if external_no_ssl:
        is_pure_ssl = labels_df["slide_name"].str.contains(PURE_SSL_REGEX, case=False, regex=True, na=False)
        labels_df = labels_df.loc[~is_pure_ssl].copy()

    feats_dir = dataset_root / feature_subdir
    labels_df = filter_available(labels_df, feats_dir)
    if labels_df.empty:
        raise ValueError(f"No slides with features in: {feats_dir}")

    mode, tile_pooling = _feature_kind(feature_subdir)
    rows = []
    for r in labels_df.itertuples(index=False):
        vec = vector_from_h5(feats_dir / f"{r.slide_name}.h5", mode=mode, tile_pooling=tile_pooling)
        rows.append(vec.astype(np.float64, copy=False))

    x = np.asarray(rows, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D matrix for {feature_subdir}, got shape={x.shape}")

    out = labels_df[["slide_name", "slide_label"]].copy().reset_index(drop=True)
    out = out.rename(columns={"slide_label": "label"})
    out["row_idx"] = np.arange(len(out), dtype=np.int64)
    out.attrs["X"] = x
    return out


def _stratified_indices(labels: np.ndarray, n_target: int, seed: int) -> np.ndarray:
    n = int(labels.shape[0])
    if n_target >= n:
        return np.arange(n, dtype=np.int64)

    ser = pd.Series(labels)
    counts = ser.value_counts().sort_index()
    raw = counts / counts.sum() * float(n_target)
    base = np.floor(raw).astype(int)
    rem = int(n_target - base.sum())
    frac_order = (raw - base).sort_values(ascending=False).index.tolist()
    for c in frac_order:
        if rem <= 0:
            break
        if int(base[c]) < int(counts[c]):
            base[c] += 1
            rem -= 1

    rng = np.random.default_rng(int(seed))
    out_idx = []
    for c in counts.index.tolist():
        n_c = int(base[c])
        if n_c <= 0:
            continue
        idx_c = np.where(labels == c)[0]
        pick = rng.choice(idx_c, size=n_c, replace=False)
        out_idx.append(pick)

    if not out_idx:
        return rng.choice(np.arange(n), size=n_target, replace=False).astype(np.int64)
    out = np.concatenate(out_idx, axis=0)
    rng.shuffle(out)
    return out.astype(np.int64, copy=False)


def _knn_domain_mixing(x_int_n: np.ndarray, x_ext_n: np.ndarray, k: int) -> float:
    x = np.concatenate([x_int_n, x_ext_n], axis=0)
    d = np.concatenate(
        [np.zeros(x_int_n.shape[0], dtype=np.int64), np.ones(x_ext_n.shape[0], dtype=np.int64)],
        axis=0,
    )
    n = int(x.shape[0])
    k_eff = max(1, min(int(k), n - 1))
    nn = NearestNeighbors(n_neighbors=k_eff + 1, metric="euclidean")
    nn.fit(x)
    idx = nn.kneighbors(x, return_distance=False)[:, 1:]
    p_ext = d[idx].mean(axis=1)
    p_ext = np.clip(p_ext, 1e-12, 1 - 1e-12)
    h = -(p_ext * np.log(p_ext) + (1 - p_ext) * np.log(1 - p_ext)) / np.log(2.0)
    return float(np.mean(h))


def _compute_metrics(
    x_int: np.ndarray,
    x_ext: np.ndarray,
    device: torch.device,
    sigma_sample_size: int,
    energy_sample_size: int,
    energy_chunk_size: int,
    knn_k: int,
    eps: float,
    seed: int,
) -> dict[str, float]:
    x_int_n, x_ext_n = normalize_by_internal(x_int, x_ext, eps=eps)

    mmd_raw, sigma = mmd_raw_on_arrays(
        x_int_n,
        x_ext_n,
        device=device,
        sigma_sample_size=int(sigma_sample_size),
        eps=float(eps),
        seed=int(seed) + 11,
    )
    fid_raw = frechet_gaussian(x_int_n, x_ext_n, eps=float(eps))
    coral_raw = coral_distance(x_int_n, x_ext_n, eps=float(eps))
    energy_raw = energy_distance(
        x_int_n,
        x_ext_n,
        device=device,
        sample_size=int(energy_sample_size),
        chunk_size=int(energy_chunk_size),
        seed=int(seed) + 23,
    )
    knn_mix = _knn_domain_mixing(x_int_n, x_ext_n, k=int(knn_k))

    return {
        "mmd_raw": float(mmd_raw),
        "rbf_sigma_raw": float(sigma),
        "fid_raw": float(fid_raw),
        "coral_raw": float(coral_raw),
        "energy_raw": float(energy_raw),
        "knn_domain_mixing": float(knn_mix),
    }


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    internal_root = Path(args.internal_root)
    external_root = Path(args.external_root)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_csv, out_summary_csv, out_stability_csv, out_json = _resolve_out_paths(args, runs_root)
    print(f"Using device: {device}")

    fm_data: dict[str, dict] = {}
    for subdir in FM_SUBDIRS:
        fm = _fm_name_from_subdir(subdir)
        print(f"[build] {fm} <- {subdir}")
        int_df = _build_fm_matrix(internal_root, subdir, external_no_ssl=False)
        ext_df = _build_fm_matrix(external_root, subdir, external_no_ssl=True)
        fm_data[fm] = {
            "subdir": subdir,
            "int_df": int_df,
            "ext_df": ext_df,
            "X_int": int_df.attrs["X"],
            "X_ext": ext_df.attrs["X"],
        }

    long_rows = []
    for fm, dct in fm_data.items():
        int_df = dct["int_df"]
        ext_df = dct["ext_df"]
        x_int_all = dct["X_int"]
        x_ext_all = dct["X_ext"]

        n_ext_total = int(len(ext_df))
        n_int_total = int(len(int_df))
        n_match = int(args.target_per_domain) if int(args.target_per_domain) > 0 else int(n_ext_total)
        n_match = min(n_match, n_int_total, n_ext_total)
        if n_match <= 0:
            continue

        y_int = int_df["label"].to_numpy(dtype=np.int64)
        y_ext = ext_df["label"].to_numpy(dtype=np.int64)

        for rep in range(int(args.n_repeats)):
            s = int(args.seed) + 10007 * rep
            idx_int = _stratified_indices(y_int, n_target=n_match, seed=s + 11)

            if args.sample_external:
                idx_ext = _stratified_indices(y_ext, n_target=n_match, seed=s + 29)
            else:
                idx_ext = np.arange(n_ext_total, dtype=np.int64)

            x_int = x_int_all[idx_int]
            x_ext = x_ext_all[idx_ext]

            metrics = _compute_metrics(
                x_int=x_int,
                x_ext=x_ext,
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
                    "fm_name": fm,
                    "feature_subdir": dct["subdir"],
                    "repetition": rep,
                    "n_internal_total": n_int_total,
                    "n_external_no_ssl_total": n_ext_total,
                    "n_internal_sample": int(len(idx_int)),
                    "n_external_sample": int(len(idx_ext)),
                    **metrics,
                }
            )

    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        raise ValueError("No rows computed.")
    long_df = long_df.sort_values(["fm_name", "repetition"]).reset_index(drop=True)
    long_df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv} (rows={len(long_df)})")

    metric_cols = ["mmd_raw", "fid_raw", "coral_raw", "energy_raw", "knn_domain_mixing"]
    agg = {}
    for m in metric_cols:
        agg[f"{m}_mean"] = (m, "mean")
        agg[f"{m}_std"] = (m, "std")
        agg[f"{m}_median"] = (m, "median")
        agg[f"{m}_q25"] = (m, lambda s: float(np.nanquantile(s, 0.25)))
        agg[f"{m}_q75"] = (m, lambda s: float(np.nanquantile(s, 0.75)))
        agg[f"{m}_q05"] = (m, lambda s: float(np.nanquantile(s, 0.05)))
        agg[f"{m}_q95"] = (m, lambda s: float(np.nanquantile(s, 0.95)))

    summary_df = (
        long_df.groupby(["fm_name", "feature_subdir"], as_index=False)
        .agg(n_repeats=("repetition", "nunique"), **agg)
        .sort_values("fm_name")
        .reset_index(drop=True)
    )
    summary_df.to_csv(out_summary_csv, index=False)
    print(f"Wrote: {out_summary_csv} (rows={len(summary_df)})")

    maximize = {"knn_domain_mixing"}
    stability_rows = []
    for m in metric_cols:
        best_counts: dict[str, float] = {}
        for _, g in long_df.groupby("repetition", sort=True):
            best_val = g[m].max() if m in maximize else g[m].min()
            winners = g.loc[g[m] == best_val, "fm_name"].tolist()
            for w in winners:
                best_counts[w] = best_counts.get(w, 0.0) + (1.0 / len(winners))

        for fm in sorted(long_df["fm_name"].unique().tolist()):
            c = float(best_counts.get(fm, 0.0))
            stability_rows.append(
                {
                    "metric_name": m,
                    "fm_name": fm,
                    "best_count": c,
                    "best_fraction": c / float(args.n_repeats),
                    "n_repeats": int(args.n_repeats),
                }
            )

    stability_df = pd.DataFrame(stability_rows).sort_values(["metric_name", "best_count"], ascending=[True, False])
    stability_df.to_csv(out_stability_csv, index=False)
    print(f"Wrote: {out_stability_csv} (rows={len(stability_df)})")

    meta = {
        "internal_root": str(internal_root),
        "external_root": str(external_root),
        "fm_subdirs": FM_SUBDIRS,
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
        "out_csv": str(out_csv),
        "out_summary_csv": str(out_summary_csv),
        "out_stability_csv": str(out_stability_csv),
    }
    with open(out_json, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()