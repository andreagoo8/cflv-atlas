from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasets import build_datasets
from models import PatchABMILClassifier, SlideMLPClassifier
from train import compute_class_weights, evaluate, fit, save_metrics, seed_worker, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train WSI classifier on Trident embeddings.")

    parser.add_argument("--mode", choices=["patch", "slide"], default="patch")
    parser.add_argument(
        "--tile_pooling",
        choices=["abmil", "mean", "max"],
        default="abmil",
        help="For patch mode: 'abmil' uses sampled tile bags, 'mean' averages all tiles per slide, 'max' max-pools tiles per slide.",
    )
    parser.add_argument(
        "--feature_subdir",
        type=str,
        default=None,
        help="Path inside each dataset root, e.g. '20x_224px_0px_overlap/features_virchow2' or "
        "'20x_512px_0px_overlap/slide_features_titan'.",
    )

    parser.add_argument(
        "--train_root",
        type=str,
        default="/data/projects/D34H/git_repo/repo-shahry/wp-1/Classification_Trident/embeddings/PRISM/IMP_2",
    )
    parser.add_argument(
        "--val_root",
        type=str,
        default="/data/projects/D34H/git_repo/repo-shahry/wp-1/Classification_Trident/embeddings/PRISM/IMP_1",
    )
    parser.add_argument(
        "--internal_test_root",
        type=str,
        default="/data/projects/D34H/git_repo/repo-shahry/wp-1/Classification_Trident/embeddings/PRISM/IMP_TEST",
    )
    parser.add_argument(
        "--external_test_root",
        type=str,
        default="/data/projects/D34H/git_repo/repo-shahry/wp-1/Classification_Trident/embeddings/PRISM/APSS",
    )

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--use_scheduler", action="store_true")

    parser.add_argument("--num_patches", type=int, default=512, help="Used only in patch mode.")

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_heads", type=int, default=1, help="ABMIL only.")
    parser.add_argument("--head_dim", type=int, default=512, help="ABMIL only.")
    parser.add_argument("--no_gated", action="store_true", help="Disable gated attention in ABMIL.")

    parser.add_argument(
        "--output_root",
        type=str,
        default="/data/projects/D34H/git_repo/repo-shahry/wp-1/Classification_Trident/Codes/runs",
    )
    parser.add_argument("--run_name", type=str, default=None)

    return parser.parse_args()


def resolve_feature_subdir(mode: str, feature_subdir: str | None) -> str:
    if feature_subdir is not None:
        return feature_subdir
    if mode == "patch":
        return "20x_224px_0px_overlap/features_virchow2"
    return "20x_512px_0px_overlap/slide_features_titan"


def build_model(args: argparse.Namespace, input_dim: int, num_classes: int) -> torch.nn.Module:
    if args.mode == "patch" and args.tile_pooling == "abmil":
        model = PatchABMILClassifier(
            input_feature_dim=input_dim,
            num_classes=num_classes,
            n_heads=args.n_heads,
            head_dim=args.head_dim,
            dropout=args.dropout,
            gated=not args.no_gated,
            hidden_dim=args.hidden_dim,
        )
    else:
        model = SlideMLPClassifier(
            input_feature_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
    return model


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    feature_subdir = resolve_feature_subdir(args.mode, args.feature_subdir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.mode == "patch":
        run_name = args.run_name or f"{args.mode}_{args.tile_pooling}_{Path(feature_subdir).name}_{timestamp}"
    else:
        run_name = args.run_name or f"{args.mode}_{Path(feature_subdir).name}_{timestamp}"
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(
        json.dumps({**vars(args), "feature_subdir": feature_subdir, "run_name": run_name}, indent=2)
    )

    bundle = build_datasets(
        train_root=args.train_root,
        val_root=args.val_root,
        internal_test_root=args.internal_test_root,
        external_test_root=args.external_test_root,
        feature_subdir=feature_subdir,
        mode=args.mode,
        tile_pooling=args.tile_pooling,
        seed=args.seed,
        num_patches=args.num_patches,
    )

    print("Dataset stats:")
    print(json.dumps(bundle.stats, indent=2))
    print(f"Input dim: {bundle.input_dim}, Num classes: {bundle.num_classes}")
    if args.mode == "patch" and args.tile_pooling in {"mean", "max"}:
        if args.tile_pooling == "mean":
            print("[INFO] tile_pooling=mean: averaging all tiles per slide; num_patches is ignored.")
        else:
            print("[INFO] tile_pooling=max: max-pooling all tiles per slide; num_patches is ignored.")

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        bundle.train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        bundle.val,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )
    internal_loader = DataLoader(
        bundle.internal_test,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )
    external_loader = DataLoader(
        bundle.external_test,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, bundle.input_dim, bundle.num_classes).to(device)

    class_weights = compute_class_weights(bundle.train.labels, num_classes=bundle.num_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs) if args.use_scheduler else None
    )

    fit_info = fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        num_classes=bundle.num_classes,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        run_dir=run_dir,
    )

    ckpt = torch.load(fit_info["best_ckpt_path"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    val_metrics, val_preds = evaluate(model, val_loader, device, criterion, num_classes=bundle.num_classes)
    internal_metrics, internal_preds = evaluate(
        model, internal_loader, device, criterion, num_classes=bundle.num_classes
    )
    external_metrics, external_preds = evaluate(
        model, external_loader, device, criterion, num_classes=bundle.num_classes
    )

    val_preds.to_csv(run_dir / "preds_val.csv", index=False)
    internal_preds.to_csv(run_dir / "preds_internal_test.csv", index=False)
    external_preds.to_csv(run_dir / "preds_external_test.csv", index=False)

    all_metrics = {
        "val": val_metrics,
        "internal_test": internal_metrics,
        "external_test": external_metrics,
        "best_val_macro_f1": float(ckpt["best_val_macro_f1"]),
        "feature_subdir": feature_subdir,
        "mode": args.mode,
        "tile_pooling": args.tile_pooling,
        "train_root": args.train_root,
        "val_root": args.val_root,
        "internal_test_root": args.internal_test_root,
        "external_test_root": args.external_test_root,
    }
    save_metrics(all_metrics, run_dir / "metrics.json")

    summary_df = pd.DataFrame(
        [
            {"split": "val", **{k: v for k, v in val_metrics.items() if k != "confusion_matrix"}},
            {"split": "internal_test", **{k: v for k, v in internal_metrics.items() if k != "confusion_matrix"}},
            {"split": "external_test", **{k: v for k, v in external_metrics.items() if k != "confusion_matrix"}},
        ]
    )
    summary_df.to_csv(run_dir / "summary.csv", index=False)

    print("\nFinal results:")
    print(summary_df.to_string(index=False))
    print(f"\nArtifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
