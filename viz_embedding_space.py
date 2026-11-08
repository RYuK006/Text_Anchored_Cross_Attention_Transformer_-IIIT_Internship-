import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

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
    
    print("\n" + "="*50)
    print("Loading Best Model (H=16)")
    print("="*50)
    
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
        print(f"Error: Could not find checkpoint at '{model_path}'")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    
    print("\nExtracting fused embeddings from the test set...")
    
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Extracting"):
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            vision = batch['vision'].to(device)
            labels = batch['label'].to(device)
            
            # Use the new return_embeddings flag
            embeddings = model(text, audio, vision, return_embeddings=True)
            
            if len(labels.shape) > 1 and labels.shape[-1] == 1:
                labels = labels.squeeze(-1)
                
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            
    embeddings_arr = np.concatenate(all_embeddings, axis=0)
    labels_arr = np.concatenate(all_labels, axis=0)
    
    print(f"Extracted embeddings shape: {embeddings_arr.shape}")
    print(f"Extracted labels shape: {labels_arr.shape}")
    
    # Subsample to 1000 points if the test set is too large
    if len(embeddings_arr) > 1000:
        print("Subsampling to 1000 random points for t-SNE visualization...")
        # Use a fixed seed for reproducibility
        np.random.seed(42)
        indices = np.random.choice(len(embeddings_arr), 1000, replace=False)
        embeddings_arr = embeddings_arr[indices]
        labels_arr = labels_arr[indices]
        
    print("\nRunning t-SNE dimensionality reduction (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings_arr)
    
    print("Generating scatter plot...")
    os.makedirs("visualizations", exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # Scatter plot with RdYlGn colormap
    # Labels represent sentiment score from negative (red) to positive (green)
    scatter = plt.scatter(
        embeddings_2d[:, 0], 
        embeddings_2d[:, 1], 
        c=labels_arr, 
        cmap='RdYlGn', 
        alpha=0.7, 
        edgecolors='k',
        s=40
    )
    
    cbar = plt.colorbar(scatter)
    cbar.set_label("Sentiment Score", fontsize=12)
    
    plt.title("t-SNE Visualization of Fused Multimodal Embeddings (Colored by Sentiment)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("t-SNE Component 1", fontsize=12, fontweight='bold')
    plt.ylabel("t-SNE Component 2", fontsize=12, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    out_path = "visualizations/embedding_space_tsne.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    
    print(f"\nPlot successfully saved to: {out_path}")

if __name__ == "__main__":
    main()
