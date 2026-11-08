import os
import sys
import torch
import numpy as np
from tqdm import tqdm

# Ensure current directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.dataloader import get_dataloaders
from src.models.transformer_fusion import TextAnchoredCrossAttentionTransformer
from src.training.train import calculate_metrics

def main():
    dataset = "mosei"  
    batch_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading data for {dataset}...")
    try:
        _, _, test_loader = get_dataloaders(dataset_name=dataset, data_dir="data", batch_size=batch_size)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return

    sample_batch = next(iter(test_loader))
    text_dim = sample_batch['text'].shape[-1]
    audio_dim = sample_batch['audio'].shape[-1]
    vision_dim = sample_batch['vision'].shape[-1]
    
    print("\n" + "="*50)
    print("Initializing Ensemble Models")
    print("="*50)

    model_configs = [
        {"path": "data/best_model_final.pt", "num_heads": 16, "name": "Model 1 (Final Best, H=16)"},
        {"path": "data/tf_h16.pt", "num_heads": 16, "name": "Model 2 (H=16)"},
        {"path": "data/tf_h8.pt", "num_heads": 8, "name": "Model 3 (H=8)"}
    ]

    models = []
    for cfg in model_configs:
        model = TextAnchoredCrossAttentionTransformer(
            text_dim=text_dim, 
            audio_dim=audio_dim, 
            vision_dim=vision_dim, 
            embed_dim=128, 
            num_heads=cfg["num_heads"], 
            hidden_dim=128, 
            dropout=0.2
        )
        
        # We explicitly set weights_only=True to silence the PyTorch warning
        model.load_state_dict(torch.load(cfg["path"], map_location=device, weights_only=True))
        model.to(device)
        model.eval()
        
        models.append({"model": model, "name": cfg["name"]})
        print(f"Successfully loaded {cfg['name']} from {cfg['path']}")

    print("\n" + "="*50)
    print("Running Inference on Test Set")
    print("="*50)

    # Store individual predictions and true labels
    all_preds = {m["name"]: [] for m in models}
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Ensemble Inference"):
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            vision = batch['vision'].to(device)
            labels = batch['label'].to(device)
            
            # Squeeze labels if necessary
            if len(labels.shape) > 1 and labels.shape[-1] == 1:
                labels = labels.squeeze(-1)
                
            all_labels.extend(labels.cpu().numpy())
            
            # Get predictions from each model
            for m_dict in models:
                outputs = m_dict["model"](text, audio, vision)
                if outputs.shape[-1] == 1:
                    outputs = outputs.squeeze(-1)
                all_preds[m_dict["name"]].extend(outputs.cpu().numpy())

    all_labels = np.array(all_labels)

    print("\n" + "="*50)
    print("INDIVIDUAL MODEL METRICS:")
    print("="*50)
    
    ensemble_preds_sum = np.zeros_like(all_labels, dtype=float)

    for m_dict in models:
        preds = np.array(all_preds[m_dict["name"]])
        ensemble_preds_sum += preds
        
        # Use our standard evaluation function
        metrics = calculate_metrics(preds, all_labels)
        print(f"{m_dict['name']:<28} Accuracy: {metrics['accuracy']*100:.2f}% | F1: {metrics['f1']:.4f}")

    # Soft ensemble: Average the raw regression predictions before thresholding
    ensemble_preds_avg = ensemble_preds_sum / len(models)
    
    # Calculate final metrics for the ensemble
    ensemble_metrics = calculate_metrics(ensemble_preds_avg, all_labels)

    print("\n" + "="*50)
    print("FINAL ENSEMBLE METRICS (Soft Average):")
    print("="*50)
    print(f"  MAE:  {ensemble_metrics['mae']:.4f}")
    print(f"  Binary Accuracy:  {ensemble_metrics['accuracy']:.4f} ({ensemble_metrics['accuracy']*100:.2f}%)")
    print(f"  Weighted F1:  {ensemble_metrics['f1']:.4f}")
    print(f"  Pearson r:  {ensemble_metrics['corr']:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
