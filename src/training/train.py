import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr
from tqdm import tqdm

# Add parent directory to path to allow importing src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data.dataloader import get_dataloaders
from src.models.baseline_models import EarlyFusionModel, LateFusionModel
from src.models.transformer_fusion import TextAnchoredCrossAttentionTransformer

def calculate_metrics(predictions, labels):
    """
    Calculates research-standard evaluation metrics for CMU-MOSI:
    - Mean Absolute Error (MAE): Lower is better
    - Pearson Correlation (r): Higher is better
    - Binary Accuracy (Acc-2): Sentiment classification (Positive > 0 vs Negative <= 0)
    - F1 Score: Binary F1 score
    """
    mae = np.mean(np.abs(predictions - labels))
    
    # Pearson Correlation
    try:
        corr, _ = pearsonr(predictions, labels)
    except Exception:
        corr = 0.0
        
    # Binary Classification Metrics (checking sign of sentiment score)
    binary_preds = (predictions >= 0).astype(int)
    binary_labels = (labels >= 0).astype(int)
    
    acc = accuracy_score(binary_labels, binary_preds)
    f1 = f1_score(binary_labels, binary_preds, average='binary', zero_division=0)
    
    return {
        "mae": mae,
        "corr": corr,
        "accuracy": acc,
        "f1": f1
    }

def train_model(model, train_loader, valid_loader, test_loader, epochs=20, lr=1e-3, device="cpu", save_path="best_model.pt", patience=7):
    print(f"Training on device: {device}")
    model.to(device)
    
    # Loss function and Optimizer
    criterion = nn.L1Loss() # MAE loss is standard for sentiment regression
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    
    best_valid_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        # 1. Training Phase
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            vision = batch['vision'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(text, audio, vision)
            
            # Squeeze output and labels to match shape
            if outputs.shape[-1] == 1:
                outputs = outputs.squeeze(-1)
            if len(labels.shape) > 1 and labels.shape[-1] == 1:
                labels = labels.squeeze(-1)
                
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * text.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # 2. Validation Phase
        model.eval()
        valid_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in valid_loader:
                text = batch['text'].to(device)
                audio = batch['audio'].to(device)
                vision = batch['vision'].to(device)
                labels = batch['label'].to(device)
                
                outputs = model(text, audio, vision)
                if outputs.shape[-1] == 1:
                    outputs = outputs.squeeze(-1)
                if len(labels.shape) > 1 and labels.shape[-1] == 1:
                    labels = labels.squeeze(-1)
                    
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * text.size(0)
                
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        valid_loss /= len(valid_loader.dataset)
        scheduler.step(valid_loss)
        
        # Calculate validation metrics
        val_metrics = calculate_metrics(np.array(all_preds), np.array(all_labels))
        print(f"Epoch {epoch} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Valid Loss: {valid_loss:.4f}")
        print(f"  Valid Acc: {val_metrics['accuracy']:.4f} | Valid F1: {val_metrics['f1']:.4f} | Valid MAE: {val_metrics['mae']:.4f} | Valid Corr: {val_metrics['corr']:.4f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_valid_acc:
            best_valid_acc = val_metrics['accuracy']
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
            print(f"  => Saved new best model checkpoint to {save_path}")
        else:
            patience_counter += 1
            
        print(f"  Current LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        if patience_counter >= patience:
            print("Early stopping triggered")
            break
            
    # 3. Test Phase (Load best model and evaluate)
    print("\n=== FINAL TEST EVALUATION ===")
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path))
        print(f"Loaded best model checkpoint from {save_path}")
        
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            text = batch['text'].to(device)
            audio = batch['audio'].to(device)
            vision = batch['vision'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(text, audio, vision)
            if outputs.shape[-1] == 1:
                outputs = outputs.squeeze(-1)
            if len(labels.shape) > 1 and labels.shape[-1] == 1:
                labels = labels.squeeze(-1)
                
            test_preds.extend(outputs.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
            
    test_metrics = calculate_metrics(np.array(test_preds), np.array(test_labels))
    print("Test Set Metrics:")
    print(f"  MAE:  {test_metrics['mae']:.4f} (Lower is better)")
    print(f"  Corr: {test_metrics['corr']:.4f} (Higher is better)")
    print(f"  Acc:  {test_metrics['accuracy']:.4f} (Higher is better)")
    print(f"  F1:   {test_metrics['f1']:.4f} (Higher is better)")
    
    test_metrics['best_epoch'] = best_epoch
    test_metrics['best_valid_acc'] = best_valid_acc
    
    return test_metrics

def main():
    # Configuration
    dataset = "mosei"
    batch_size = 32
    epochs = 20
    lr = 5e-4
    model_type = "transformer" # Choice: "early", "late", "transformer"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Multimodal Experiment: Dataset={dataset.upper()} | Model={model_type.upper()}")
    
    # Load Data Loaders
    try:
        train_loader, valid_loader, test_loader = get_dataloaders(
            dataset_name=dataset, 
            data_dir="data", 
            batch_size=batch_size
        )
    except FileNotFoundError as e:
        print(f"\n[Warning] {e}")
        print("Please place the downloaded 'mosi_data.pkl' file in the 'data/' directory to run the training.")
        return
        
    # Get input feature dimensions based on the dataset structure
    # CMU-MOSI shapes are usually: Text=300 (GloVe) or 768 (BERT), Audio=5, Vision=20 (Facet) or 47 (OpenFace)
    # Let's inspect a single batch to get dimensions automatically
    sample_batch = next(iter(train_loader))
    text_dim = sample_batch['text'].shape[-1]
    audio_dim = sample_batch['audio'].shape[-1]
    vision_dim = sample_batch['vision'].shape[-1]
    print(f"Auto-detected feature dimensions: Text={text_dim}, Audio={audio_dim}, Vision={vision_dim}")
    
    # Initialize Model
    if model_type == "early":
        model = EarlyFusionModel(text_dim, audio_dim, vision_dim, hidden_dim=64, dropout=0.3)
    elif model_type == "late":
        model = LateFusionModel(text_dim, audio_dim, vision_dim, hidden_dim=64, dropout=0.3)
    else:
        model = TextAnchoredCrossAttentionTransformer(
            text_dim=text_dim, 
            audio_dim=audio_dim, 
            vision_dim=vision_dim, 
            embed_dim=128, 
            num_heads=4, 
            hidden_dim=128, 
            dropout=0.2
        )
        
    train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        save_path=f"{model_type}_best_model.pt"
    )

if __name__ == "__main__":
    main()
