import os
import sys
import time
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Ensure current directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.dataloader import get_dataloaders
from src.models.baseline_models import EarlyFusionModel, LateFusionModel
from src.models.transformer_fusion import TextAnchoredCrossAttentionTransformer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    csv_path = "experiment_results.csv"
    if not os.path.exists(csv_path):
        print(f"Error: '{csv_path}' not found.")
        return
        
    df = pd.read_csv(csv_path)
    print("Loaded experiment_results.csv successfully.")
    
    print("\nLoading test data for inference latency measurement...")
    try:
        _, _, test_loader = get_dataloaders(dataset_name="mosei", data_dir="data", batch_size=32)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
    
    sample_batch = next(iter(test_loader))
    text_dim = sample_batch['text'].shape[-1]
    audio_dim = sample_batch['audio'].shape[-1]
    vision_dim = sample_batch['vision'].shape[-1]
    total_samples = len(test_loader.dataset)
    
    # Mapping experiment IDs to their corresponding saved weights and architecture configs
    exp_mapping = {
        "1A": {"path": "data/early_best.pt", "type": "early"},
        "1B": {"path": "data/late_best.pt", "type": "late"},
        "1C": {"path": "data/tf_best.pt", "type": "tf", "kwargs": {}},
        "2A": {"path": "data/tf_text_only.pt", "type": "tf", "ablate": ["audio", "vision"], "kwargs": {}},
        "2B": {"path": "data/tf_text_audio.pt", "type": "tf", "ablate": ["vision"], "kwargs": {}},
        "2C": {"path": "data/tf_text_vision.pt", "type": "tf", "ablate": ["audio"], "kwargs": {}},
        "2D": {"path": "data/tf_full.pt", "type": "tf", "kwargs": {}},
        "3A (H=8)": {"path": "data/tf_h8.pt", "type": "tf", "kwargs": {"num_heads": 8}},
        "3A (H=16)": {"path": "data/tf_h16.pt", "type": "tf", "kwargs": {"num_heads": 16}},
        "3B (D=0.1)": {"path": "data/tf_d01.pt", "type": "tf", "kwargs": {"dropout": 0.1}},
        "3B (D=0.5)": {"path": "data/tf_d05.pt", "type": "tf", "kwargs": {"dropout": 0.5}}
    }
    
    latencies = {}
    
    print("\n" + "="*50)
    print("Measuring Latency (ms/sample) for each experiment")
    print("="*50)
    
    for idx, row in df.iterrows():
        exp_id = row['Experiment']
        if exp_id not in exp_mapping:
            continue
            
        cfg = exp_mapping[exp_id]
        
        # Instantiate correct model architecture based on updated hyperparams (128 dims)
        if cfg["type"] == "early":
            model = EarlyFusionModel(text_dim, audio_dim, vision_dim, hidden_dim=128, dropout=0.2)
        elif cfg["type"] == "late":
            model = LateFusionModel(text_dim, audio_dim, vision_dim, hidden_dim=128, dropout=0.2)
        elif cfg["type"] == "tf":
            kwargs = {
                "embed_dim": 128, 
                "num_heads": 4, 
                "hidden_dim": 128, 
                "dropout": 0.2
            }
            kwargs.update(cfg.get("kwargs", {}))
            model = TextAnchoredCrossAttentionTransformer(text_dim, audio_dim, vision_dim, **kwargs)
            
            if "ablate" in cfg:
                if "audio" in cfg["ablate"]:
                    model.ablate_audio = True
                if "vision" in cfg["ablate"]:
                    model.ablate_vision = True
                    
        if not os.path.exists(cfg["path"]):
            print(f"[{exp_id}] Warning: '{cfg['path']}' not found. Skipping.")
            continue
            
        # Load the saved checkpoint
        model.load_state_dict(torch.load(cfg["path"], map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        
        # GPU warmup (perform a few batches before starting timer)
        with torch.no_grad():
            for _ in range(2):
                batch = next(iter(test_loader))
                _ = model(batch['text'].to(device), batch['audio'].to(device), batch['vision'].to(device))
                
        # Synchronize CUDA to ensure accurate timing
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        # Run inference and measure time
        start_time = time.time()
        with torch.no_grad():
            for batch in test_loader:
                _ = model(batch['text'].to(device), batch['audio'].to(device), batch['vision'].to(device))
                
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        end_time = time.time()
        
        total_time_ms = (end_time - start_time) * 1000
        ms_per_sample = total_time_ms / total_samples
        latencies[exp_id] = ms_per_sample
        print(f"[{exp_id:<10}] {ms_per_sample:.4f} ms/sample")
        
    # Append latencies to the dataframe
    df['Latency_ms'] = df['Experiment'].map(latencies)
    
    # Drop rows that failed to load
    plot_df = df.dropna(subset=['Latency_ms'])
    
    print("\n" + "="*50)
    print("Generating Scatter Plot")
    print("="*50)
    
    os.makedirs("visualizations", exist_ok=True)
    
    plt.figure(figsize=(10, 7))
    
    phases = {
        "Phase 1 (Baselines)": {"prefix": "1", "color": "royalblue"},
        "Phase 2 (Ablation)": {"prefix": "2", "color": "darkorange"},
        "Phase 3 (Tuning)": {"prefix": "3", "color": "forestgreen"}
    }
    
    for phase_name, p_info in phases.items():
        phase_df = plot_df[plot_df['Experiment'].str.startswith(p_info["prefix"])]
        if not phase_df.empty:
            plt.scatter(
                phase_df['Latency_ms'], 
                phase_df['Binary Accuracy'], 
                c=p_info["color"], 
                label=phase_name,
                s=120,
                alpha=0.8,
                edgecolors='black',
                linewidth=1.2
            )
            
            # Annotate each point with its Experiment ID
            for idx, row in phase_df.iterrows():
                plt.annotate(
                    row['Experiment'],
                    (row['Latency_ms'], row['Binary Accuracy']),
                    xytext=(6, 4), 
                    textcoords='offset points',
                    fontsize=9,
                    fontweight='bold'
                )
                
    plt.title("Latency vs Accuracy Trade-off Across All Experiments", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Inference Latency (ms per sample)", fontsize=12, fontweight='bold')
    plt.ylabel("Binary Accuracy", fontsize=12, fontweight='bold')
    
    # Format Y axis as percentages if values are decimals (e.g. 0.79 -> 79%)
    if plot_df['Binary Accuracy'].max() <= 1.0:
        vals = plt.gca().get_yticks()
        plt.gca().set_yticklabels(['{:,.1%}'.format(x) for x in vals])

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    out_path = "visualizations/latency_vs_accuracy.png"
    plt.savefig(out_path, dpi=300)
    print(f"Plot successfully saved to: {out_path}")

if __name__ == "__main__":
    main()
