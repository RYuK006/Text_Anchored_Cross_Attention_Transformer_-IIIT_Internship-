import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class MultimodalDataset(Dataset):
    """
    A standard PyTorch Dataset class for loading preprocessed, pre-aligned 
    multimodal sentiment and emotion datasets (CMU-MOSI, CMU-MOSEI, IEMOCAP).
    """
    def __init__(self, data_split):
        self.vision = torch.nan_to_num(torch.tensor(data_split['vision'], dtype=torch.float32), posinf=0.0, neginf=0.0)
        self.audio = torch.nan_to_num(torch.tensor(data_split['audio'], dtype=torch.float32), posinf=0.0, neginf=0.0)
        self.text = torch.nan_to_num(torch.tensor(data_split['text'], dtype=torch.float32), posinf=0.0, neginf=0.0)
        
        # Check label shapes and format them
        labels = data_split['labels']
        if isinstance(labels, list):
            labels = np.array(labels)
            
        # Squeeze dimensions if necessary (e.g. [num_samples, 1, 1] -> [num_samples, 1])
        if len(labels.shape) == 3:
            labels = labels.squeeze(-1)
            
        self.labels = torch.tensor(labels, dtype=torch.float32)
        
        # Check matching dataset lengths
        assert len(self.vision) == len(self.audio) == len(self.text) == len(self.labels), \
            "Modality sequences and labels must have the same length!"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'text': self.text[idx],
            'audio': self.audio[idx],
            'vision': self.vision[idx],
            'label': self.labels[idx]
        }

def get_dataloaders(dataset_name="mosi", data_dir="data", batch_size=32, num_workers=0):
    """
    Loads the dataset from pickle file and returns PyTorch DataLoader objects 
    for the train, validation, and test splits.
    
    Args:
        dataset_name (str): Name of the dataset ("mosi", "mosei", or "iemocap")
        data_dir (str): Base data directory path
        batch_size (int): Batch size for the data loaders
        num_workers (int): Number of worker processes for loading data
        
    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    file_mapping = {
        "mosi": "mosi_data.pkl",
        "mosei": "mosei_senti_data.pkl",
        "iemocap": "iemocap_data.pkl"
    }
    
    if dataset_name.lower() not in file_mapping:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Choose from 'mosi', 'mosei', 'iemocap'.")
        
    file_path = os.path.join(data_dir, file_mapping[dataset_name.lower()])
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Dataset file {file_path} not found! Please run the dataset downloader script first."
        )
        
    print(f"Loading {dataset_name.upper()} dataset from {file_path}...")
    with open(file_path, "rb") as f:
        # Load pickle file with compatibility settings for python3
        data = pickle.load(f, encoding="latin1")
        
    print("Dataset splits loaded successfully:")
    for split in ["train", "valid", "test"]:
        num_samples = len(data[split]["labels"])
        text_shape = np.shape(data[split]["text"])
        audio_shape = np.shape(data[split]["audio"])
        vision_shape = np.shape(data[split]["vision"])
        print(f"  {split.capitalize()} set size: {num_samples} samples | Shapes: Text={text_shape[1:]}, Audio={audio_shape[1:]}, Vision={vision_shape[1:]}")
        
    # Create Dataset objects
    train_dataset = MultimodalDataset(data['train'])
    valid_dataset = MultimodalDataset(data['valid'])
    test_dataset = MultimodalDataset(data['test'])
    
    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    # Test file execution
    try:
        train, valid, test = get_dataloaders(dataset_name="mosi", data_dir="data", batch_size=16)
        print("Data loaders initialized successfully!")
    except Exception as e:
        print(f"Test run notice: {e}")
