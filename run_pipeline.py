import argparse
import os
import sys

# Ensure current directory is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="Multimodal Sentiment Analysis Research Pipeline")
    parser.add_argument(
        "--download", 
        action="store_true", 
        help="Download pre-aligned CMU-MOSI and CMU-MOSEI datasets from Dropbox"
    )
    parser.add_argument(
        "--train", 
        action="store_true", 
        help="Run model training and evaluation"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="transformer", 
        choices=["early", "late", "transformer"],
        help="Model architecture to train (early, late, transformer)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=10, 
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Batch size for data loaders"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="mosi", 
        choices=["mosi", "mosei", "iemocap"],
        help="Dataset to train on (mosi, mosei, iemocap)"
    )
    
    args = parser.parse_args()
    
    if not args.download and not args.train:
        parser.print_help()
        print("\nExample usage:")
        print("  python run_pipeline.py --download")
        print("  python run_pipeline.py --train --model transformer --epochs 10")
        return

    # 1. Download Dataset
    if args.download:
        print("\n>>> Launching Dataset Downloader...")
        from src.data.download_data import download_and_extract
        download_and_extract()
        
    # 2. Train Model
    if args.train:
        print("\n>>> Launching Model Training Pipeline...")
        import torch
        from src.data.dataloader import get_dataloaders
        from src.models.baseline_models import EarlyFusionModel, LateFusionModel
        from src.models.transformer_fusion import TextAnchoredCrossAttentionTransformer
        from src.training.train import train_model
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device} | Dataset: {args.dataset.upper()} | Model: {args.model.upper()}")
        
        # Ingest Dataloaders
        try:
            train_loader, valid_loader, test_loader = get_dataloaders(
                dataset_name=args.dataset,
                data_dir="data",
                batch_size=args.batch_size
            )
        except FileNotFoundError as e:
            print(f"\n[Error] {e}")
            print("Please run 'python run_pipeline.py --download' first to retrieve the data files.")
            sys.exit(1)
            
        # Detect dimensions automatically
        sample_batch = next(iter(train_loader))
        text_dim = sample_batch['text'].shape[-1]
        audio_dim = sample_batch['audio'].shape[-1]
        vision_dim = sample_batch['vision'].shape[-1]
        print(f"Detected dimensions: Text={text_dim}, Audio={audio_dim}, Vision={vision_dim}")
        
        # Initialize selected model architecture
        if args.model == "early":
            model = EarlyFusionModel(text_dim, audio_dim, vision_dim, hidden_dim=64, dropout=0.3)
        elif args.model == "late":
            model = LateFusionModel(text_dim, audio_dim, vision_dim, hidden_dim=64, dropout=0.3)
        else:
            model = TextAnchoredCrossAttentionTransformer(
                text_dim=text_dim,
                audio_dim=audio_dim,
                vision_dim=vision_dim,
                embed_dim=64,
                num_heads=4,
                hidden_dim=64,
                dropout=0.3
            )
            
        # Train and validate
        train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            lr=5e-4 if args.model == "transformer" else 1e-3,
            device=device,
            save_path=f"data/{args.dataset}_{args.model}_best.pt"
        )

if __name__ == "__main__":
    main()
