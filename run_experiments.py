import os
import sys
import pandas as pd
import torch

# Ensure current directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.dataloader import get_dataloaders
from src.models.baseline_models import EarlyFusionModel, LateFusionModel
from src.models.transformer_fusion import TextAnchoredCrossAttentionTransformer
from src.training.train import train_model

def run_experiment(exp_id, desc, model, train_loader, valid_loader, test_loader, epochs, lr, device, save_name):
    print(f"\n{'='*50}\nStarting Experiment {exp_id}: {desc}\n{'='*50}")
    metrics = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        save_path=f"data/{save_name}.pt"
    )
    return {
        "Experiment": exp_id,
        "Description": desc,
        "MAE": metrics["mae"],
        "Binary Accuracy": metrics["accuracy"],
        "Weighted F1": metrics["f1"],
        "Pearson r": metrics["corr"]
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-mode", action="store_true", help="Run 1 epoch only for quick testing")
    args = parser.parse_args()

    # Use mosei dataset as it is confirmed to be available locally
    dataset = "mosei"  
    batch_size = 32
    epochs = 1 if args.test_mode else 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading data for {dataset}...")
    try:
        train_loader, valid_loader, test_loader = get_dataloaders(dataset_name=dataset, data_dir="data", batch_size=batch_size)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    sample_batch = next(iter(train_loader))
    text_dim = sample_batch['text'].shape[-1]
    audio_dim = sample_batch['audio'].shape[-1]
    vision_dim = sample_batch['vision'].shape[-1]
    
    results = []

    # Phase 1: Baseline Comparison
    print("\n>>> Phase 1: Baseline Comparison")
    
    # Early Fusion
    model_early = EarlyFusionModel(text_dim, audio_dim, vision_dim, hidden_dim=128, dropout=0.2)
    results.append(run_experiment("1A", "Early Fusion", model_early, train_loader, valid_loader, test_loader, epochs, 1e-3, device, "early_best"))

    # Late Fusion
    model_late = LateFusionModel(text_dim, audio_dim, vision_dim, hidden_dim=128, dropout=0.2)
    results.append(run_experiment("1B", "Late Fusion", model_late, train_loader, valid_loader, test_loader, epochs, 1e-3, device, "late_best"))

    # Cross-Attention
    model_tf = TextAnchoredCrossAttentionTransformer(text_dim, audio_dim, vision_dim, embed_dim=128, num_heads=4, hidden_dim=128, dropout=0.2)
    results.append(run_experiment("1C", "Cross-Attention (Proposed)", model_tf, train_loader, valid_loader, test_loader, epochs, 5e-4, device, "tf_best"))

    # Phase 2: Ablation Study
    print("\n>>> Phase 2: Ablation Study")
    
    # Text Only
    model_ab_text = TextAnchoredCrossAttentionTransformer(text_dim, audio_dim, vision_dim, embed_dim=128, num_heads=4, hidden_dim=128, dropout=0.2)
    model_ab_text.ablate_audio = True
    model_ab_text.ablate_vision = True
    results.append(run_experiment("2A", "Text Only", model_ab_text, train_loader, valid_loader, test_loader, epochs, 5e-4, device, "tf_text_only"))

    # Text + Audio
    model_ab_ta = TextAnchoredCrossAttentionTransformer(text_dim, audio_dim, vision_dim, embed_dim=128, num_heads=4, hidden_dim=128, dropout=0.2)
    model_ab_ta.ablate_vision = True
    results.append(run_experiment("2B", "Text + Audio", model_ab_ta, train_loader, valid_loader, test_loader, epochs, 5e-4, device, "tf_text_audio"))

    # Text + Vision
    model_ab_tv = TextAnchoredCrossAttentionTransformer(text_dim, audio_dim, vision_dim, embed_dim=128, num_heads=4, hidden_dim=128, dropout=0.2)
    model_ab_tv.ablate_audio = True
    results.append(run_experiment("2C", "Text + Vision", model_ab_tv, train_loader, valid_loader, test_loader, epochs, 5e-4, device, "tf_text_vision"))

    # Text + Audio + Vision
    model_ab_full = TextAnchoredCrossAttentionTransformer(text_dim, audio_dim, vision_dim, embed_dim=128, num_heads=4, hidden_dim=128, dropout=0.2)
    results.append(run_experiment("2D", "Text + Audio + Vision", model_ab_full, train_loader, valid_loader, test_loader, epochs, 5e-4, device, "tf_full"))

    # Phase 3: Hyperparameter Tuning
    print("\n>>> Phase 3: Hyperparameter Tuning")
    
    for heads in [8, 16]:
        model_tune = TextAnchoredCrossAttentionTransformer(text_dim, audio_dim, vision_dim, embed_dim=128, num_heads=heads, hidden_dim=128, dropout=0.2)
        results.append(run_experiment(f"3A (H={heads})", f"Num Heads = {heads}", model_tune, train_loader, valid_loader, test_loader, epochs, 5e-4, device, f"tf_h{heads}"))

    for drop in [0.1, 0.5]:
        model_tune = TextAnchoredCrossAttentionTransformer(text_dim, audio_dim, vision_dim, embed_dim=128, num_heads=4, hidden_dim=128, dropout=drop)
        results.append(run_experiment(f"3B (D={drop})", f"Dropout = {drop}", model_tune, train_loader, valid_loader, test_loader, epochs, 5e-4, device, f"tf_d{str(drop).replace('.','')}"))

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("experiment_results.csv", index=False)
    
    print("\n\n" + "="*50)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*50)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
