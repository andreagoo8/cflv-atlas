from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets import build_datasets
from models import PatchABMILClassifier, SlideMLPClassifier
from train import compute_class_weights, evaluate, save_metrics, seed_worker, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference from a trained WSI classifier checkpoint.")

    parser.add_argument("--run_dir", type=str, default=None, help="Training run directory containing config.json.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt).")

    parser.add_argument("--mode", choices=["patch", "slide"], default=None)
    parser.add_argument("--tile_pooling", choices=["abmil", "mean", "max"], default=None)
    parser.add_argument("--feature_subdir", type=str, default=None)

    parser.add_argument(
        "--train_root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--val_root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--internal_test_root",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--external_test_root",
        type=str,
        default=None,
    )

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--num_patches", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_train", action="store_true")

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_heads", type=int, default=1)
    parser.add_argument("--head_dim", type=int, default=512)
    parser.add_argument("--no_gated", action="store_true")

    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def _load_run_config(run_dir: Path | None) -> Dict[str, Any]:
    if run_dir is None:
        return {}
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r") as f:
        return json.load(f)


def _pick(arg_value, cfg: Dict[str, Any], key: str):
    if arg_value is not None:
        return arg_value
    return cfg.get(key)


def _pick_root(arg_value, cfg: Dict[str, Any], new_key: str, old_key: str, default: str):
    if arg_value is not None:
        return arg_value
    if new_key in cfg and cfg.get(new_key) is not None:
        return cfg.get(new_key)
    if old_key in cfg and cfg.get(old_key) is not None:
        return cfg.get(old_key)
    return default


def resolve_feature_subdir(mode: str, feature_subdir: str | None) -> str:
    if feature_subdir is not None:
        return feature_subdir
    if mode == "patch":
        return "20x_224px_0px_overlap/features_virchow2"
    return "20x_512px_0px_overlap/slide_features_titan"


def build_model(
    mode: str,
    tile_pooling: str,
    input_dim: int,
    num_classes: int,
    hidden_dim: int,
    dropout: float,
    n_heads: int,
    head_dim: int,
    no_gated: bool,
) -> torch.nn.Module:
    if mode == "patch" and tile_pooling == "abmil":
        return PatchABMILClassifier(
            input_feature_dim=input_dim,
            num_classes=num_classes,
            n_heads=n_heads,
            head_dim=head_dim,
            dropout=dropout,
            gated=not no_gated,
            hidden_dim=hidden_dim,
        )
    return SlideMLPClassifier(
        input_feature_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir) if args.run_dir is not None else None
    cfg = _load_run_config(run_dir)

    mode = _pick(args.mode, cfg, "mode")
    if mode is None:
        raise ValueError("Mode is required. Set --mode or provide --run_dir with config.json.")
    if mode == "patch":
        tile_pooling = _pick(args.tile_pooling, cfg, "tile_pooling") or "abmil"
    else:
        tile_pooling = _pick(args.tile_pooling, cfg, "tile_pooling") or "abmil"

    feature_subdir = resolve_feature_subdir(mode, _pick(args.feature_subdir, cfg, "feature_subdir"))
    seed = int(_pick(args.seed, cfg, "seed") if args.seed is not None else 1234)
    num_patches = int(_pick(args.num_patches, cfg, "num_patches") if args.num_patches is not None else 512)

    hidden_dim = int(_pick(args.hidden_dim, cfg, "hidden_dim") if args.hidden_dim is not None else 256)
    dropout = float(_pick(args.dropout, cfg, "dropout") if args.dropout is not None else 0.1)
    n_heads = int(_pick(args.n_heads, cfg, "n_heads") if args.n_heads is not None else 1)
    head_dim = int(_pick(args.head_dim, cfg, "head_dim") if args.head_dim is not None else 512)
    no_gated = bool(_pick(args.no_gated, cfg, "no_gated") if args.no_gated is not None else False)

    checkpoint = args.checkpoint
    if checkpoint is None and run_dir is not None:
        checkpoint = str(run_dir / "best_model.pt")
    if checkpoint is None:
        raise ValueError("Checkpoint is required. Set --checkpoint or provide --run_dir with best_model.pt.")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    set_seed(seed)
    train_root = _pick_root(
        args.train_root,
        cfg,
        new_key="train_root",
        old_key="imp1_root",
        default="/data/projects/D34H/git_repo/repo-shahry/wp-1/Classification_Trident/embeddings/PRISM/IMP_2",
    )
    val_root = _pick_root(
        args.val_root,
        cfg,
        new_key="val_root",
        old_key="imp1_root",
        default="/data/projects/D34H/git_repo/repo-shahry/wp-1/Classification_Trident/embeddings/PRISM/IMP_1",
    )
    internal_test_root = _pick_root(
        args.internal_test_root,
        cfg,
        new_key="internal_test_root",
        old_key="imp_test_root",
        default="/data/projects/D34H/git_repo/repo-shahry/wp-1/Classification_Trident/embeddings/PRISM/IMP_TEST",
    )
    external_test_root = _pick_root(
        args.external_test_root,
        cfg,
        new_key="external_test_root",
        old_key="apss_root",
        default="/data/projects/D34H/git_repo/repo-shahry/wp-1/Classification_Trident/embeddings/PRISM/APSS",
    )

    bundle = build_datasets(
        train_root=train_root,
        val_root=val_root,
        internal_test_root=internal_test_root,
        external_test_root=external_test_root,
        feature_subdir=feature_subdir,
        mode=mode,
        tile_pooling=tile_pooling,
        seed=seed,
        num_patches=num_patches,
    )

    model = build_model(
        mode=mode,
        tile_pooling=tile_pooling,
        input_dim=bundle.input_dim,
        num_classes=bundle.num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
        n_heads=n_heads,
        head_dim=head_dim,
        no_gated=no_gated,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)

    class_weights = compute_class_weights(bundle.train.labels, num_classes=bundle.num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    pin_memory = torch.cuda.is_available()
    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )

    split_loaders = {
        "val": DataLoader(bundle.val, **loader_kwargs),
        "internal_test": DataLoader(bundle.internal_test, **loader_kwargs),
        "external_test": DataLoader(bundle.external_test, **loader_kwargs),
    }
    if args.save_train:
        split_loaders["train"] = DataLoader(bundle.train, **loader_kwargs)

    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    elif run_dir is not None:
        out_dir = run_dir / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        out_dir = Path(__file__).resolve().parent / "predictions" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: Dict[str, Dict[str, object]] = {}
    summary_rows = []

    for split_name, loader in split_loaders.items():
        metrics, preds_df = evaluate(model, loader, device, criterion, num_classes=bundle.num_classes)
        all_metrics[split_name] = metrics
        preds_df.to_csv(out_dir / f"preds_{split_name}.csv", index=False)
        summary_rows.append({"split": split_name, **{k: v for k, v in metrics.items() if k != "confusion_matrix"}})

    save_metrics(
        {
            "mode": mode,
            "tile_pooling": tile_pooling,
            "feature_subdir": feature_subdir,
            "checkpoint": str(checkpoint_path),
            "train_root": train_root,
            "val_root": val_root,
            "internal_test_root": internal_test_root,
            "external_test_root": external_test_root,
            "dataset_stats": bundle.stats,
            "metrics": all_metrics,
        },
        out_dir / "metrics.json",
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    print("Inference summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved predictions to: {out_dir}")


if __name__ == "__main__":
    main()
