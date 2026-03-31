"""
Evaluate the trained Attribution Model and plot score distributions 
for both the held-out Main dataset and the Out-of-Distribution Echoes dataset.
"""

import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader

from model.network import AttributionModel
from model.dataset import PairDataset, EchoesValDataset, collate_pairs

sns.set_theme(style="whitegrid")

def _get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def run_inference(model, loader, device, dataset_name):
    """Run model over a dataloader and collect predictions and labels."""
    model.eval()
    results = []
    
    print(f"Evaluating {dataset_name} ({len(loader.dataset)} pairs)...")
    with torch.no_grad():
        for batch_dict, label_dict in loader:
            # Move to device
            batch_dict = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch_dict.items()}
            
            # Forward pass
            out = model(batch_dict)
            
            # Apply sigmoid to get 0-1 scores
            sim_preds = torch.sigmoid(out["sim_logit"].squeeze(-1)).cpu().numpy()
            attr_preds = torch.sigmoid(out["attr_logit"].squeeze(-1)).cpu().numpy()
            
            sim_labels = label_dict["similarity"].numpy()
            attr_labels = label_dict["is_attribution"].numpy()
            
            for sp, ap, sl, al in zip(sim_preds, attr_preds, sim_labels, attr_labels):
                results.append({
                    "Dataset": dataset_name,
                    "Predicted_Similarity": sp,
                    "Predicted_Attribution": ap,
                    "True_Similarity": sl,
                    "True_Attribution": al,
                    # Define binary ground truth for coloring the plots
                    "Pair_Type": "Positive (Related)" if al > 0.5 or sl > 0.5 else "Negative (Irrelevant)"
                })
                
    return pd.DataFrame(results)

def main(args):
    device = _get_device()
    print(f"Using device: {device}")

    # 1. Load Checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    config = ckpt.get("config", {"hidden_dim": 256, "feature_set": "advanced", "use_lyrics": False, "dropout": 0.1})
    
    model = AttributionModel(
        hidden_dim=config["hidden_dim"],
        feature_set=config["feature_set"],
        use_lyrics=config.get("use_lyrics", False),
        dropout=config.get("dropout", 0.0)
    ).to(device)
    
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 2. Load Datasets (Validation Split Only!)
    val_main_ds = PairDataset(
        data_dir=args.data_dir, cache_dir=args.cache_dir, 
        neg_ratio=1.5, split="val", val_ratio=0.15
    )
    val_echoes_ds = EchoesValDataset(data_dir=args.data_dir, cache_dir=args.cache_dir)

    main_loader = DataLoader(val_main_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pairs)
    echoes_loader = DataLoader(val_echoes_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pairs) if len(val_echoes_ds) > 0 else None

    # 3. Collect Results
    df_main = run_inference(model, main_loader, device, "Main Validation (15% Split)")
    df_echoes = run_inference(model, echoes_loader, device, "Echoes (OOD)") if echoes_loader else pd.DataFrame()
    
    df_all = pd.concat([df_main, df_echoes], ignore_index=True)

    # 4. Plot Distributions
    print("Generating distribution plots...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot formatting helper
    def plot_dist(data, col_pred, title, ax):
        if data.empty: return
        sns.histplot(
            data=data, x=col_pred, hue="Pair_Type", 
            kde=True, bins=30, ax=ax, palette={"Positive (Related)": "blue", "Negative (Irrelevant)": "red"},
            alpha=0.5, stat="density", common_norm=False
        )
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Predicted Score (0.0 to 1.0)")
        ax.set_xlim(0, 1)

    # Top Row: Main Validation Split
    plot_dist(df_main, "Predicted_Similarity", "Main Val: Musical Similarity Score", axes[0, 0])
    plot_dist(df_main, "Predicted_Attribution", "Main Val: AI Attribution Score", axes[0, 1])
    
    # Bottom Row: Echoes Dataset
    if not df_echoes.empty:
        plot_dist(df_echoes, "Predicted_Similarity", "Echoes (OOD): Musical Similarity Score", axes[1, 0])
        plot_dist(df_echoes, "Predicted_Attribution", "Echoes (OOD): AI Attribution Score", axes[1, 1])
    else:
        axes[1, 0].text(0.5, 0.5, "Echoes Data Not Found", ha='center', va='center')
        axes[1, 1].text(0.5, 0.5, "Echoes Data Not Found", ha='center', va='center')

    plt.tight_layout()
    plot_path = "evaluation_distributions.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\n✅ Evaluation complete! Saved plots to: {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to your best.pt checkpoint")
    parser.add_argument("--cache_dir", type=str, default="feature_cache_merged")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()
    main(args)