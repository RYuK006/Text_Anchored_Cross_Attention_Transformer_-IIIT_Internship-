import os
import sys
import torch

# Ensure current directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.dataloader import get_dataloaders
from src.models.transformer_fusion import TextAnchoredCrossAttentionTransformer
from src.training.train import train_model

def main():
    dataset = "mosei"  
    batch_size = 32
    epochs = 50
    lr = 8e-5
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
    
    print("\n" + "="*50)
    print("Starting Focused Best Experiment: Text-Anchored Cross-Attention (16 Heads)")
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
    
    metrics = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        save_path="data/best_model_final.pt",
        patience=10
    )
    
    print("\n" + "="*50)
    print("FINAL BEST EXPERIMENT METRICS:")
    print("="*50)
    print(f"Best model was saved from epoch {metrics['best_epoch']} with validation accuracy {metrics['best_valid_acc']*100:.2f}%")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  Binary Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Weighted F1:  {metrics['f1']:.4f}")
    print(f"  Pearson r:  {metrics['corr']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
