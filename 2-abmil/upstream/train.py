from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    labels_np = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels_np, minlength=num_classes).astype(np.float32)
    if np.any(counts == 0):
        counts[counts == 0] = 1.0
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, object]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist(),
    }
    return metrics


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    num_classes: int,
) -> Tuple[Dict[str, object], pd.DataFrame]:
    model.eval()

    losses: List[float] = []
    all_probs: List[np.ndarray] = []
    all_preds: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_names: List[str] = []

    for features, labels, slide_names in dataloader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(features)
        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        losses.append(float(loss.item()))
        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_names.extend(list(slide_names))

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)

    metrics = _compute_metrics(y_true, y_pred, num_classes=num_classes)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")

    pred_df = pd.DataFrame(
        {
            "slide_name": all_names,
            "true_label": y_true,
            "pred_label": y_pred,
        }
    )
    for c in range(num_classes):
        pred_df[f"prob_class_{c}"] = probs[:, c]

    return metrics, pred_df


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses: List[float] = []

    for features, labels, _ in dataloader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else float("nan")


def fit(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    num_classes: int,
    device: torch.device,
    epochs: int,
    patience: int,
    run_dir: Path,
) -> Dict[str, List[Dict[str, float]]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = run_dir / "best_model.pt"

    history: List[Dict[str, float]] = []
    best_f1 = -1.0
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics, _ = evaluate(model, val_loader, device, criterion, num_classes=num_classes)

        if scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        entry = {
            "epoch": epoch,
            "lr": float(lr),
            "train_loss": float(train_loss),
            "val_loss": float(val_metrics["loss"]),
            "val_macro_f1": float(val_metrics["macro_f1"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
        }
        history.append(entry)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = float(val_metrics["macro_f1"])
            stale_epochs = 0
            torch.save({"model_state_dict": model.state_dict(), "best_val_macro_f1": best_f1}, best_ckpt_path)
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (no val_macro_f1 improvement for {patience} epochs).")
                break

    history_path = run_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))
    return {"history": history, "best_ckpt_path": str(best_ckpt_path)}


def save_metrics(metrics: Dict[str, object], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(metrics, indent=2))
