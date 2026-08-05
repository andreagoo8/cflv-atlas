from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
import h5py



PURE_SSL_REGEX = r"^[^_]+_sessile[_ ]serrated(?:[_ ]lesion)?(?:[_ ]displasia)?$"


def read_labels(dataset_root: Path) -> pd.DataFrame:
    labels_path = Path(dataset_root) / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels.csv: {labels_path}")

    df = pd.read_csv(labels_path)
    required = {"slide_name", "slide_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{labels_path} missing columns: {sorted(missing)}")

    out = df[["slide_name", "slide_label"]].dropna().copy()
    out["slide_name"] = out["slide_name"].astype(str).str.strip()
    out["slide_label"] = out["slide_label"].astype(int)
    return out


def filter_available(labels_df: pd.DataFrame, feats_dir: Path) -> pd.DataFrame:
    feats_dir = Path(feats_dir)
    if not feats_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {feats_dir}")
    available = {p.stem for p in feats_dir.glob("*.h5")}
    out = labels_df.loc[labels_df["slide_name"].isin(available)].copy()
    return out.reset_index(drop=True)


def vector_from_h5(h5_path: Path, mode: str, tile_pooling: str) -> np.ndarray:
    if h5py is None:
        raise ModuleNotFoundError("h5py is required to read .h5 files.")
    with h5py.File(h5_path, "r") as f:
        feats = np.asarray(f["features"][()])

    if feats.ndim == 1:
        return feats.astype(np.float64, copy=False)
    if feats.ndim != 2:
        raise ValueError(f"Unsupported feature shape {feats.shape} in {h5_path}")

    if mode == "slide":
        return feats.reshape(-1).astype(np.float64, copy=False)
    if tile_pooling == "mean":
        return feats.mean(axis=0).astype(np.float64, copy=False)
    if tile_pooling == "max":
        return feats.max(axis=0).astype(np.float64, copy=False)

    raise ValueError(f"Unsupported pooling for vector_from_h5: mode={mode}, tile_pooling={tile_pooling}")


def normalize_by_internal(x_int: np.ndarray, x_ext: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    mu = np.mean(x_int, axis=0)
    sd = np.std(x_int, axis=0)
    sd = np.where(sd < eps, 1.0, sd)
    return (x_int - mu) / sd, (x_ext - mu) / sd


def _to_device(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=torch.float64, device=device)


def _estimate_sigma_median(z: torch.Tensor, sample_size: int = 512, eps: float = 1e-12, seed: int = 1234) -> float:
    n = int(z.shape[0])
    if n < 2:
        raise ValueError("Need >=2 points to estimate sigma")

    g = torch.Generator(device=z.device.type if z.device.type != "cpu" else "cpu")
    g.manual_seed(int(seed))

    if n > sample_size:
        idx = torch.randperm(n, generator=g, device=z.device)[:sample_size]
        zs = z[idx]
    else:
        zs = z

    d2 = torch.cdist(zs, zs, p=2.0) ** 2
    tri = torch.triu_indices(d2.shape[0], d2.shape[1], offset=1, device=z.device)
    vals = d2[tri[0], tri[1]]
    vals = vals[torch.isfinite(vals)]
    vals = vals[vals > 0]
    if vals.numel() == 0:
        return 1.0
    med = torch.median(vals).item()
    return float(np.sqrt(max(med, eps)))


def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float, eps: float = 1e-12) -> torch.Tensor:
    sigma = max(float(sigma), float(eps))
    gamma = 1.0 / (2.0 * sigma * sigma)
    d2 = torch.cdist(x, y, p=2.0) ** 2
    return torch.exp(-gamma * d2)


def _mmd2_unbiased(x: torch.Tensor, y: torch.Tensor, sigma: float, eps: float = 1e-12) -> float:
    n = int(x.shape[0])
    m = int(y.shape[0])
    if n < 2 or m < 2:
        return float("nan")

    k_xx = _rbf_kernel(x, x, sigma, eps)
    k_yy = _rbf_kernel(y, y, sigma, eps)
    k_xy = _rbf_kernel(x, y, sigma, eps)

    term_x = (k_xx.sum() - torch.diagonal(k_xx).sum()) / (n * (n - 1))
    term_y = (k_yy.sum() - torch.diagonal(k_yy).sum()) / (m * (m - 1))
    term_xy = k_xy.mean()
    out = term_x + term_y - 2.0 * term_xy
    return float(out.item())


def mmd_raw_on_arrays(
    x_int: np.ndarray,
    x_ext: np.ndarray,
    device: torch.device,
    sigma_sample_size: int = 512,
    eps: float = 1e-12,
    seed: int = 1234,
) -> tuple[float, float]:
    t_int = _to_device(x_int, device)
    t_ext = _to_device(x_ext, device)
    t_all = torch.cat([t_int, t_ext], dim=0)
    sigma = _estimate_sigma_median(t_all, sample_size=sigma_sample_size, eps=eps, seed=seed)
    mmd_raw = _mmd2_unbiased(t_int, t_ext, sigma=sigma, eps=eps)
    return float(mmd_raw), float(sigma)


def _covariance(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n, d = x.shape
    if n <= 1:
        return np.eye(d, dtype=np.float64) * float(eps)
    c = np.cov(x, rowvar=False, bias=False)
    c = np.asarray(c, dtype=np.float64)
    if c.ndim == 0:
        c = np.array([[float(c)]], dtype=np.float64)
    c = 0.5 * (c + c.T)
    c = c + np.eye(c.shape[0], dtype=np.float64) * float(eps)
    return c


def coral_distance(x_int: np.ndarray, x_ext: np.ndarray, eps: float = 1e-12) -> float:
    c_int = _covariance(x_int, eps=eps)
    c_ext = _covariance(x_ext, eps=eps)
    d = float(c_int.shape[0])
    diff = c_int - c_ext
    return float(np.sum(diff * diff) / (4.0 * d * d))


def _sqrtm_psd(a: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = 0.5 * (a + a.T)
    w, v = np.linalg.eigh(a)
    w = np.clip(w, a_min=0.0, a_max=None)
    return (v * np.sqrt(w + float(eps))) @ v.T


def frechet_gaussian(x_int: np.ndarray, x_ext: np.ndarray, eps: float = 1e-12) -> float:
    m1 = np.mean(x_int, axis=0)
    m2 = np.mean(x_ext, axis=0)
    c1 = _covariance(x_int, eps=eps)
    c2 = _covariance(x_ext, eps=eps)

    dm = m1 - m2
    dm2 = float(dm @ dm)

    c1_sqrt = _sqrtm_psd(c1, eps=eps)
    middle = c1_sqrt @ c2 @ c1_sqrt
    middle_sqrt = _sqrtm_psd(middle, eps=eps)

    trace_term = float(np.trace(c1 + c2 - 2.0 * middle_sqrt))
    w2 = dm2 + trace_term
    if w2 < 0 and abs(w2) < 1e-10:
        w2 = 0.0
    return float(max(w2, 0.0))


def _sample_points(x: np.ndarray, n_target: int, seed: int) -> np.ndarray:
    n = int(x.shape[0])
    if n <= n_target:
        return x
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=int(n_target), replace=False)
    return x[idx]


def _mean_cdist(x: torch.Tensor, y: torch.Tensor, chunk_size: int) -> torch.Tensor:
    n = int(x.shape[0])
    total = torch.zeros((), device=x.device, dtype=x.dtype)
    count = 0
    for i in range(0, n, int(chunk_size)):
        xb = x[i : i + int(chunk_size)]
        d = torch.cdist(xb, y, p=2.0)
        total = total + d.sum()
        count += int(d.numel())
    return total / max(1, count)


def _mean_cdist_offdiag(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    n = int(x.shape[0])
    if n < 2:
        return torch.tensor(float("nan"), device=x.device, dtype=x.dtype)

    total = torch.zeros((), device=x.device, dtype=x.dtype)
    count = 0
    for i in range(0, n, int(chunk_size)):
        xb = x[i : i + int(chunk_size)]
        ib = int(xb.shape[0])
        d = torch.cdist(xb, x, p=2.0)
        cols = torch.arange(i, i + ib, device=x.device)
        rows = torch.arange(ib, device=x.device)
        d[rows, cols] = 0.0
        total = total + d.sum()
        count += int(ib * n - ib)
    return total / max(1, count)


def energy_distance(
    x_int: np.ndarray,
    x_ext: np.ndarray,
    device: torch.device,
    sample_size: int = 2048,
    chunk_size: int = 512,
    seed: int = 1234,
) -> float:
    xi = _sample_points(x_int, n_target=int(sample_size), seed=seed + 11)
    xe = _sample_points(x_ext, n_target=int(sample_size), seed=seed + 23)

    tx = torch.as_tensor(xi, dtype=torch.float32, device=device)
    ty = torch.as_tensor(xe, dtype=torch.float32, device=device)

    mean_xy = _mean_cdist(tx, ty, chunk_size=int(chunk_size))
    mean_xx = _mean_cdist_offdiag(tx, chunk_size=int(chunk_size))
    mean_yy = _mean_cdist_offdiag(ty, chunk_size=int(chunk_size))

    out = 2.0 * mean_xy - mean_xx - mean_yy
    return float(out.detach().cpu().item())