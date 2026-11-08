import os
import pickle
import numpy as np
import warnings



def main():
    file_path = os.path.join(os.path.dirname(__file__), 'data', 'mosei_senti_data.pkl')
    
    print(f"Loading data from: {file_path}")
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
        
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f"Error loading file: {e}")
        return
        
    print("\n" + "="*50)
    print("TOP-LEVEL KEYS:")
    print("="*50)
    for k in data.keys():
        print(f"- {k}")
        
    print("\n" + "="*50)
    print("ANALYZING SPLITS (train, valid, test):")
    print("="*50)
    
    keywords = ['bert', 'roberta', 'glove', 'text']
    found_keywords = set()
    
    for split in ['train', 'valid', 'test']:
        if split in data:
            print(f"\n[{split.upper()} SPLIT]")
            split_data = data[split]
            for mod_key, feature_data in split_data.items():
                
                # Check for keywords
                lower_key = str(mod_key).lower()
                for kw in keywords:
                    if kw in lower_key:
                        found_keywords.add((split, mod_key))
                
                # Print shapes
                if isinstance(feature_data, np.ndarray):
                    print(f"  Modality: '{mod_key}' -> Full Array Shape: {feature_data.shape}")
                elif isinstance(feature_data, list) and len(feature_data) > 0:
                    first_sample = feature_data[0]
                    if isinstance(first_sample, np.ndarray):
                        print(f"  Modality: '{mod_key}' -> First Sample Shape: {first_sample.shape} | Total Samples: {len(feature_data)}")
                    else:
                        print(f"  Modality: '{mod_key}' -> Type: List of {type(first_sample)} | Total Samples: {len(feature_data)}")
                else:
                    print(f"  Modality: '{mod_key}' -> Type: {type(feature_data)}")
        else:
            print(f"\n[{split.upper()} SPLIT]")
            print(f"  Split '{split}' not found in top-level dictionary.")
            
    print("\n" + "="*50)
    print("KEYWORD MATCHES (bert, roberta, glove, text):")
    print("="*50)
    if not found_keywords:
        print("No matching text/embedding keywords found in any modality keys.")
    else:
        # Sort by split then key
        for split, k in sorted(list(found_keywords)):
            print(f"*** PROMINENT KEY FOUND: '{k}' (in '{split}' split) ***")

if __name__ == "__main__":
    main()
