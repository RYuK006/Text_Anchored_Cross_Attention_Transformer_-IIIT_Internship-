import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

# Ensure current directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.dataloader import get_dataloaders
from src.models.transformer_fusion import TextAnchoredCrossAttentionTransformer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("\nLoading test data...")
    try:
        _, _, test_loader = get_dataloaders(dataset_name="mosei", data_dir="data", batch_size=32)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
        
    sample_batch = next(iter(test_loader))
    text_dim = sample_batch['text'].shape[-1]
    audio_dim = sample_batch['audio'].shape[-1]
    vision_dim = sample_batch['vision'].shape[-1]
    
    print("\nLoading Best Model (H=16)...")
    model = TextAnchoredCrossAttentionTransformer(
        text_dim=text_dim, 
        audio_dim=audio_dim, 
        vision_dim=vision_dim, 
        embed_dim=128, 
        num_heads=16, 
        hidden_dim=128, 
        dropout=0.2
    )
    
    model_path = "data/tf_h16.pt"
    if not os.path.exists(model_path):
        print(f"Error: Checkpoint '{model_path}' not found.")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    pos_data = None
    neg_data = None
    
    # Iterate through the test set to find strong sentiment examples
    print("Searching for a highly positive and a highly negative test sample...")
    with torch.no_grad():
        for batch in test_loader:
            labels = batch['label'].to(device)
            if len(labels.shape) > 1 and labels.shape[-1] == 1:
                labels = labels.squeeze(-1)
                
            # Find highly positive (> 2.0 on the [-3, 3] scale)
            if pos_data is None:
                pos_mask = labels >= 2.0
                if pos_mask.any():
                    idx = pos_mask.nonzero()[0].item()
                    pos_data = {k: v[idx:idx+1].to(device) for k, v in batch.items()}
                    
            # Find highly negative (< -2.0 on the [-3, 3] scale)
            if neg_data is None:
                neg_mask = labels <= -2.0
                if neg_mask.any():
                    idx = neg_mask.nonzero()[0].item()
                    neg_data = {k: v[idx:idx+1].to(device) for k, v in batch.items()}
                    
            if pos_data is not None and neg_data is not None:
                break
                
    if pos_data is None or neg_data is None:
        print("Warning: Could not find extreme samples (+2 or -2). Using first two samples in the test set instead.")
        pos_data = {k: v[0:1].to(device) for k, v in sample_batch.items()}
        neg_data = {k: v[1:2].to(device) for k, v in sample_batch.items()}
        
    print("\nExtracting attention weights and computing argmax alignments...")
    
    def process_sample(data):
        _ = model(data['text'], data['audio'], data['vision'])
        
        # Because we passed average_attn_weights=False in the architecture, shape is [batch_size, num_heads, seq_len, seq_len]
        # We squeeze batch_size (which is 1)
        audio_attn = model.last_audio_attn_weights.squeeze(0).cpu() 
        vision_attn = model.last_vision_attn_weights.squeeze(0).cpu()
        
        # Average across all 16 heads to get the [50, 50] grid
        if audio_attn.dim() == 3:
            audio_attn = audio_attn.mean(dim=0)
            vision_attn = vision_attn.mean(dim=0)
            
        # For each Text Timestep (row/query), find the Audio/Vision Timestep (column/key) with the highest attention score
        audio_argmax = torch.argmax(audio_attn, dim=1).numpy()
        vision_argmax = torch.argmax(vision_attn, dim=1).numpy()
        
        return audio_argmax, vision_argmax, data['label'].item()

    with torch.no_grad():
        pos_audio_argmax, pos_vision_argmax, pos_label = process_sample(pos_data)
        neg_audio_argmax, neg_vision_argmax, neg_label = process_sample(neg_data)
        
    print("Generating alignment line plots...")
    os.makedirs("visualizations", exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    def plot_alignment(ax, audio_align, vision_align, title):
        x = np.arange(len(audio_align))
        
        ax.plot(x, audio_align, marker='o', markersize=5, linestyle='-', color='crimson', label='Most-Attended Audio Timestep', alpha=0.8)
        ax.plot(x, vision_align, marker='s', markersize=5, linestyle='-', color='dodgerblue', label='Most-Attended Vision Timestep', alpha=0.8)
        
        # Diagonal reference line indicating perfect synchronous alignment (y = x)
        ax.plot(x, x, linestyle='--', color='gray', label='Perfect Synchrony (y=x)', alpha=0.6)
        
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel("Text Word Position", fontsize=11, fontweight='bold')
        ax.set_ylabel("Most-Attended Timestep", fontsize=11, fontweight='bold')
        
        # Ensure axis scales correctly represent a square matrix representation
        ax.set_xlim([0, len(audio_align)-1])
        ax.set_ylim([0, len(audio_align)-1])
        
        ax.legend(fontsize=10, loc='upper left')
        ax.grid(True, linestyle=':', alpha=0.5)

    plot_alignment(axes[0], pos_audio_argmax, pos_vision_argmax, f"Cross-Modal Alignment (Positive Sample, True: {pos_label:.2f})")
    plot_alignment(axes[1], neg_audio_argmax, neg_vision_argmax, f"Cross-Modal Alignment (Negative Sample, True: {neg_label:.2f})")
    
    plt.tight_layout(pad=3.0)
    out_path = "visualizations/cross_modal_alignment.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nPlot successfully saved to: {out_path}")

if __name__ == "__main__":
    main()
