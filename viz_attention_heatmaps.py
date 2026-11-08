import os
import sys
import torch
import matplotlib.pyplot as plt
import seaborn as sns

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
        
    print("\nExtracting attention weights...")
    
    with torch.no_grad():
        # Process positive sample
        _ = model(pos_data['text'], pos_data['audio'], pos_data['vision'])
        # Because we passed average_attn_weights=False, shape is [batch_size, num_heads, seq_len, seq_len]
        # We squeeze batch_size (which is 1)
        pos_audio_attn = model.last_audio_attn_weights.squeeze(0).cpu() 
        pos_vision_attn = model.last_vision_attn_weights.squeeze(0).cpu()
        pos_label = pos_data['label'].item()
        
        # Process negative sample
        _ = model(neg_data['text'], neg_data['audio'], neg_data['vision'])
        neg_audio_attn = model.last_audio_attn_weights.squeeze(0).cpu()
        neg_vision_attn = model.last_vision_attn_weights.squeeze(0).cpu()
        neg_label = neg_data['label'].item()
        
    # Average across all 16 heads
    # If the shape is [num_heads, 50, 50], taking mean(dim=0) results in [50, 50]
    if pos_audio_attn.dim() == 3:
        pos_audio_attn = pos_audio_attn.mean(dim=0)
        pos_vision_attn = pos_vision_attn.mean(dim=0)
        neg_audio_attn = neg_audio_attn.mean(dim=0)
        neg_vision_attn = neg_vision_attn.mean(dim=0)
        
    print("Generating heatmaps...")
    os.makedirs("visualizations", exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    def plot_heatmap(ax, data, title):
        sns.heatmap(data.numpy(), ax=ax, cmap="viridis", cbar=True)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel("Audio/Vision Timestep", fontsize=11, fontweight='bold')
        ax.set_ylabel("Text Timestep", fontsize=11, fontweight='bold')
        
    plot_heatmap(axes[0, 0], pos_audio_attn, f"Text-to-Audio Cross-Attention (Positive Sample, True: {pos_label:.2f})")
    plot_heatmap(axes[0, 1], pos_vision_attn, f"Text-to-Vision Cross-Attention (Positive Sample, True: {pos_label:.2f})")
    
    plot_heatmap(axes[1, 0], neg_audio_attn, f"Text-to-Audio Cross-Attention (Negative Sample, True: {neg_label:.2f})")
    plot_heatmap(axes[1, 1], neg_vision_attn, f"Text-to-Vision Cross-Attention (Negative Sample, True: {neg_label:.2f})")
    
    plt.tight_layout(pad=3.0)
    out_path = "visualizations/attention_heatmaps.png"
    plt.savefig(out_path, dpi=300)
    print(f"\nPlot successfully saved to: {out_path}")

if __name__ == "__main__":
    main()
