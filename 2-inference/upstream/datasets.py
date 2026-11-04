from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import hashlib

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class DatasetBundle:
    train: "H5FeatureDataset"
    val: "H5FeatureDataset"
    internal_test: "H5FeatureDataset"
    external_test: "H5FeatureDataset"
    input_dim: int
    num_classes: int
    stats: Dict[str, Dict[int, int]]


def _stable_int(text: str) -> int:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _read_labels_csv(dataset_root: Path) -> pd.DataFrame:
    labels_path = dataset_root / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels.csv: {labels_path}")

    df = pd.read_csv(labels_path)
    required = {"slide_name", "slide_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{labels_path} missing columns: {sorted(missing)}")

    df = df[["slide_name", "slide_label"]].dropna().copy()
    df["slide_name"] = df["slide_name"].astype(str)
    df["slide_label"] = df["slide_label"].astype(int)
    return df


def _filter_to_available_features(df: pd.DataFrame, features_dir: Path) -> Tuple[pd.DataFrame, List[str]]:
    if not features_dir.exists():
        raise FileNotFoundError(f"Feature directory not found: {features_dir}")

    available = {p.stem for p in features_dir.glob("*.h5")}
    keep_mask = df["slide_name"].isin(available)
    filtered = df.loc[keep_mask].copy()
    missing = sorted(df.loc[~keep_mask, "slide_name"].tolist())
    return filtered, missing


def _label_counts(df: pd.DataFrame) -> Dict[int, int]:
    return df["slide_label"].value_counts().sort_index().to_dict()


def _infer_input_dim(features_dir: Path, slide_name: str) -> int:
    h5_path = features_dir / f"{slide_name}.h5"
    with h5py.File(h5_path, "r") as f:
        arr = np.asarray(f["features"][()])
    if arr.ndim == 1:
        return int(arr.shape[0])
    if arr.ndim == 2:
        return int(arr.shape[1])
    raise ValueError(f"Unsupported feature shape {arr.shape} in {h5_path}")


class H5FeatureDataset(Dataset):
    def __init__(
        self,
        records: pd.DataFrame,
        features_dir: Path,
        mode: str,
        tile_pooling: str,
        is_train: bool,
        seed: int,
        num_patches: int = 512,
    ) -> None:
        if mode not in {"patch", "slide"}:
            raise ValueError(f"Unsupported mode: {mode}")
        if tile_pooling not in {"abmil", "mean", "max"}:
            raise ValueError(f"Unsupported tile_pooling: {tile_pooling}")
        if mode == "patch" and tile_pooling == "abmil" and num_patches <= 0:
            raise ValueError("num_patches must be > 0 for patch mode")

        self.df = records.reset_index(drop=True)
        self.features_dir = features_dir
        self.mode = mode
        self.tile_pooling = tile_pooling
        self.is_train = is_train
        self.seed = int(seed)
        self.num_patches = int(num_patches)

    def __len__(self) -> int:
        return len(self.df)

    @property
    def labels(self) -> List[int]:
        return self.df["slide_label"].astype(int).tolist()

    def _sample_patch_indices(self, n_rows: int, slide_name: str) -> np.ndarray:
        if self.is_train:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(self.seed + _stable_int(slide_name))

        replace = n_rows < self.num_patches
        return rng.choice(n_rows, size=self.num_patches, replace=replace)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        slide_name = row["slide_name"]
        slide_label = int(row["slide_label"])
        h5_path = self.features_dir / f"{slide_name}.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"Missing feature file: {h5_path}")

        with h5py.File(h5_path, "r") as f:
            feats = np.asarray(f["features"][()])

        if self.mode == "patch":
            if feats.ndim != 2:
                raise ValueError(
                    f"Patch mode expects 2D array [n_patches, feat_dim], got {feats.shape} in {h5_path}"
                )
            if feats.shape[0] == 0:
                raise ValueError(f"No patch features in: {h5_path}")
            if self.tile_pooling == "abmil":
                patch_idx = self._sample_patch_indices(feats.shape[0], slide_name)
                feats = feats[patch_idx]
            elif self.tile_pooling == "mean":
                # Mean-pool all tile embeddings to produce one slide-level vector.
                feats = feats.mean(axis=0)
            elif self.tile_pooling == "max":
                # Max-pool all tile embeddings to produce one slide-level vector.
                feats = feats.max(axis=0)
            else:
                raise ValueError(f"Unsupported tile_pooling in patch mode: {self.tile_pooling}")
        else:
            feats = np.asarray(feats).reshape(-1)

        x = torch.as_tensor(feats, dtype=torch.float32)
        y = torch.tensor(slide_label, dtype=torch.long)
        return x, y, slide_name


def build_datasets(
    train_root: str,
    val_root: str,
    internal_test_root: str,
    external_test_root: str,
    feature_subdir: str,
    mode: str,
    tile_pooling: str,
    seed: int,
    num_patches: int,
) -> DatasetBundle:
    train_root_p = Path(train_root)
    val_root_p = Path(val_root)
    internal_test_root_p = Path(internal_test_root)
    external_test_root_p = Path(external_test_root)

    train_feats = train_root_p / feature_subdir
    val_feats = val_root_p / feature_subdir
    internal_test_feats = internal_test_root_p / feature_subdir
    external_test_feats = external_test_root_p / feature_subdir

    train_df = _read_labels_csv(train_root_p)
    val_df = _read_labels_csv(val_root_p)
    internal_test_df = _read_labels_csv(internal_test_root_p)
    external_test_df = _read_labels_csv(external_test_root_p)

    train_df, train_missing = _filter_to_available_features(train_df, train_feats)
    val_df, val_missing = _filter_to_available_features(val_df, val_feats)
    internal_test_df, internal_test_missing = _filter_to_available_features(internal_test_df, internal_test_feats)
    external_test_df, external_test_missing = _filter_to_available_features(external_test_df, external_test_feats)

    empty_splits = []
    if len(train_df) == 0:
        empty_splits.append("train")
    if len(val_df) == 0:
        empty_splits.append("val")
    if len(internal_test_df) == 0:
        empty_splits.append("internal_test")
    if len(external_test_df) == 0:
        empty_splits.append("external_test")
    if empty_splits:
        raise ValueError(
            f"These splits have zero samples after matching labels with .h5 features: {', '.join(empty_splits)}."
        )

    input_dim = _infer_input_dim(train_feats, train_df.iloc[0]["slide_name"])
    all_labels = sorted(
        set(train_df["slide_label"])
        | set(val_df["slide_label"])
        | set(internal_test_df["slide_label"])
        | set(external_test_df["slide_label"])
    )
    num_classes = len(all_labels)

    datasets = DatasetBundle(
        train=H5FeatureDataset(
            train_df,
            train_feats,
            mode=mode,
            tile_pooling=tile_pooling,
            is_train=True,
            seed=seed,
            num_patches=num_patches,
        ),
        val=H5FeatureDataset(
            val_df,
            val_feats,
            mode=mode,
            tile_pooling=tile_pooling,
            is_train=False,
            seed=seed,
            num_patches=num_patches,
        ),
        internal_test=H5FeatureDataset(
            internal_test_df,
            internal_test_feats,
            mode=mode,
            tile_pooling=tile_pooling,
            is_train=False,
            seed=seed,
            num_patches=num_patches,
        ),
        external_test=H5FeatureDataset(
            external_test_df,
            external_test_feats,
            mode=mode,
            tile_pooling=tile_pooling,
            is_train=False,
            seed=seed,
            num_patches=num_patches,
        ),
        input_dim=input_dim,
        num_classes=num_classes,
        stats={
            "train": _label_counts(train_df),
            "val": _label_counts(val_df),
            "internal_test": _label_counts(internal_test_df),
            "external_test": _label_counts(external_test_df),
            "missing_feature_files": {
                "train": len(train_missing),
                "val": len(val_missing),
                "internal_test": len(internal_test_missing),
                "external_test": len(external_test_missing),
            },
        },
    )
    return datasets
