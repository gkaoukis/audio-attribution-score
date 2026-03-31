"""
Training script for the attribution model.

Usage:
    python -m model.train
    python -m model.train --feature_set basic
    python -m model.train --use_lyrics
    python -m model.train --ablation
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset

from .dataset import PairDataset, EchoesValDataset, collate_pairs
from .network import AttributionModel
from .losses import AttributionLoss

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _move_to_device(batch_dict: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}


def evaluate(model, loader, criterion, device):
    """Run evaluation and return metrics."""
    model.eval()
    total_loss = 0
    n_batches = 0
    all_preds = {"sim": [], "ai_a": [], "ai_b": [], "attr": []}
    all_labels = {"sim": [], "ai_a": [], "ai_b": [], "attr": []}

    with torch.no_grad():
        for batch_dict, label_dict in loader:
            batch_dict = _move_to_device(batch_dict, device)
            label_dict = _move_to_device(label_dict, device)

            out = model(batch_dict)
            losses = criterion(out, label_dict)
            total_loss += losses["total_loss"].item()
            n_batches += 1

            all_preds["sim"].append(torch.sigmoid(out["sim_logit"].squeeze(-1)))
            all_preds["ai_a"].append(torch.sigmoid(out["ai_logit_a"].squeeze(-1)))
            all_preds["ai_b"].append(torch.sigmoid(out["ai_logit_b"].squeeze(-1)))
            all_preds["attr"].append(torch.sigmoid(out["attr_logit"].squeeze(-1)))

            all_labels["sim"].append(label_dict["similarity"])
            all_labels["ai_a"].append(label_dict["is_ai_a"])
            all_labels["ai_b"].append(label_dict["is_ai_b"])
            all_labels["attr"].append(label_dict["is_attribution"])

    metrics = {"val_loss": total_loss / max(n_batches, 1)}
    for task in ["sim", "ai_a", "ai_b", "attr"]:
        preds = torch.cat(all_preds[task])
        labels = torch.cat(all_labels[task])
        
        # Keep binary accuracy for rough estimates
        correct = ((preds > 0.5).float() == (labels > 0.5).float()).float().mean()
        metrics[f"acc_{task}"] = correct.item()
        
        mae = torch.nn.functional.l1_loss(preds, labels)
        metrics[f"mae_{task}"] = mae.item()

    model.train()
    return metrics


def train(args, feature_set: str = None):
    """Train a single model configuration. Returns best metrics."""
    device = _get_device()
    fs = feature_set or args.feature_set
    logger.info(f"=== Training with feature_set='{fs}' ===")
    logger.info(f"Device: {device}")

    # Datasets
    logger.info("Loading training pairs...")
    train_ds = PairDataset(
        data_dir=args.data_dir, cache_dir=args.cache_dir, 
        neg_ratio=args.neg_ratio, split="train", val_ratio=0.15
    )
    logger.info("Loading validation pairs...")
    val_main_ds = PairDataset(
        data_dir=args.data_dir, cache_dir=args.cache_dir, 
        neg_ratio=args.neg_ratio, split="val", val_ratio=0.15
    )
    val_echoes_ds = EchoesValDataset(data_dir=args.data_dir, cache_dir=args.cache_dir)
    if len(val_echoes_ds) > 0:
        val_ds = ConcatDataset([val_main_ds, val_echoes_ds])
        logger.info(f"Train: {len(train_ds)} pairs")
        logger.info(f"Val: {len(val_main_ds)} (Main) + {len(val_echoes_ds)} (Echoes) = {len(val_ds)} pairs")
    else:
        val_ds = val_main_ds
        logger.info(f"Train: {len(train_ds)} pairs | Val: {len(val_ds)} pairs (Echoes not found)")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_pairs, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_pairs, num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    ) if len(val_ds) > 0 else None

    # Model
    model = AttributionModel(
        hidden_dim=args.hidden_dim,
        feature_set=fs,
        use_lyrics=args.use_lyrics,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    lambda_ai = args.lambda_ai
    criterion = AttributionLoss(
        lambda_sim=args.lambda_sim,
        lambda_ai=lambda_ai,
        lambda_attr=args.lambda_attr,
    )

    # Resume
    start_epoch = 0
    best_val_loss = float("inf")
    best_metrics = {}
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info(f"Resumed from epoch {start_epoch}")

    ckpt_dir = Path(args.checkpoint_dir) / fs
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop with early stopping
    patience_counter = 0
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        epoch_losses = {"loss_sim": 0, "loss_ai": 0, "loss_attr": 0}
        n_batches = 0
        t0 = time.time()

        for batch_dict, label_dict in train_loader:
            batch_dict = _move_to_device(batch_dict, device)
            label_dict = _move_to_device(label_dict, device)

            out = model(batch_dict)
            losses = criterion(out, label_dict)

            optimizer.zero_grad()
            losses["total_loss"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += losses["total_loss"].item()
            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        task_str = " | ".join(
            f"{k}={v / max(n_batches, 1):.4f}" for k, v in epoch_losses.items()
        )
        logger.info(
            f"[{fs}] Epoch {epoch+1}/{args.epochs} | "
            f"loss={avg_loss:.4f} | {task_str} | "
            f"lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s"
        )

        # Evaluate every epoch
        if val_loader:
            metrics = evaluate(model, val_loader, criterion, device)
            acc_str = " | ".join(
                f"{k}={v:.3f}" for k, v in metrics.items() if k.startswith("acc")
            )
            logger.info(f"  Val loss={metrics['val_loss']:.4f} | {acc_str}")

            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                best_metrics = metrics.copy()
                best_metrics["feature_set"] = fs
                best_metrics["epoch"] = epoch + 1
                patience_counter = 0
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_loss": best_val_loss,
                    "config": {
                        "hidden_dim": args.hidden_dim,
                        "feature_set": fs,
                        "use_lyrics": args.use_lyrics,
                        "dropout": args.dropout,
                    },
                }, ckpt_dir / "best.pt")
                logger.info(f"  Saved best model (val_loss={best_val_loss:.4f})")
            else:
                patience_counter += 1
                if args.patience > 0 and patience_counter >= args.patience:
                    logger.info(f"  Early stopping at epoch {epoch+1} (patience={args.patience})")
                    break

    # Final evaluation
    if val_loader:
        final_metrics = evaluate(model, val_loader, criterion, device)
        if not best_metrics:
            best_metrics = final_metrics
            best_metrics["feature_set"] = fs

    logger.info(f"[{fs}] Training complete. Best val_loss={best_val_loss:.4f}")
    return best_metrics


def run_ablation(args):
    """Run ablation study across feature sets."""
    results = {}

    for fs in ["basic", "embedding", "advanced", "mix"]:
        logger.info(f"\n{'='*60}")
        logger.info(f"ABLATION: feature_set = {fs}")
        logger.info(f"{'='*60}")
        metrics = train(args, feature_set=fs)
        if metrics:
            results[fs] = metrics

    # Print comparison
    logger.info(f"\n{'='*60}")
    logger.info("ABLATION RESULTS")
    logger.info(f"{'='*60}")
    header = f"{'Feature Set':<15} {'Val Loss':<10} {'Sim Acc':<10} {'AI_A Acc':<10} {'AI_B Acc':<10} {'Attr Acc':<10}"
    logger.info(header)
    logger.info("-" * len(header))
    for fs, m in results.items():
        logger.info(
            f"{fs:<15} {m.get('val_loss', 0):<10.4f} "
            f"{m.get('acc_sim', 0):<10.3f} {m.get('acc_ai_a', 0):<10.3f} "
            f"{m.get('acc_ai_b', 0):<10.3f} {m.get('acc_attr', 0):<10.3f}"
        )

    # Save results
    results_path = Path(args.checkpoint_dir) / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train the attribution model")
    # Data
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--cache_dir", type=str, default="feature_cache_cpu")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)

    # Model
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--feature_set", type=str, default="advanced",
                        choices=["basic", "embedding", "advanced", "mix"])
    parser.add_argument("--use_lyrics", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--neg_ratio", type=float, default=1.5)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (0 = disabled)")

    # Loss weights
    parser.add_argument("--lambda_sim", type=float, default=1.0)
    parser.add_argument("--lambda_ai", type=float, default=0.5)
    parser.add_argument("--lambda_attr", type=float, default=1.0)

    # Ablation
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation across basic/advanced feature sets")

    args = parser.parse_args()

    if args.ablation:
        run_ablation(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
