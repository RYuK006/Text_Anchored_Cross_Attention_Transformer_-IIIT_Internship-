import os
import requests
import zipfile
from tqdm import tqdm

def download_from_huggingface(repo_id="Aaron006/Text-Anchored-Transformer-Data", filename="mosei_senti_data.pkl", local_dir="data"):
    """
    Downloads preprocessed features directly from private Hugging Face dataset repo.
    Requires huggingface_hub installed and user authenticated.
    """
    try:
        from huggingface_hub import hf_hub_download
        print(f"Downloading {filename} from Hugging Face ({repo_id})...")
        path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, repo_type="dataset")
        print(f"Successfully downloaded {filename} to {path}")
        return path
    except Exception as e:
        print(f"Hugging Face download failed: {e}. Falling back to Dropbox...")
        return None

def download_and_extract(data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data")
    
    url = "https://www.dropbox.com/sh/hyzpgx1hp9nj37s/AADfY2s7gD_MkR76m03KS0K1a/Archive.zip?dl=1"
    zip_path = os.path.join(data_dir, "Archive.zip")

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    print("=== MULTIMODAL SENTIMENT ANALYSIS DATASET DOWNLOADER ===")
    print(f"Target Directory: {data_dir}")
    print(f"Downloading from: {url}")
    print("This archive contains preprocessed mosi_data.pkl, mosei_senti_data.pkl, and iemocap_data.pkl.")
    print("The total download size is approximately 3.2 GB. Let's download it...")

    # Streaming download with progress bar
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(zip_path, "wb") as f, tqdm(
            desc="Downloading Archive.zip",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024 * 1024):
                f.write(data)
                bar.update(len(data))
                
        print("\nDownload completed successfully!")
    except Exception as e:
        print(f"\nError during download: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return

    # Extracting the files
    print("\nExtracting archive...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            print("Files in archive:")
            for name in file_list:
                print(f"  - {name}")
            
            for file in file_list:
                print(f"Extracting {file}...")
                zip_ref.extract(file, data_dir)
                
        print("\nExtraction completed successfully!")
    except Exception as e:
        print(f"Error during extraction: {e}")
    finally:
        # Clean up the large zip file to save disk space
        if os.path.exists(zip_path):
            print("Cleaning up Archive.zip to free up disk space...")
            os.remove(zip_path)
            print("Cleanup completed.")

if __name__ == "__main__":
    # Try Hugging Face first, fall back to Dropbox archive
    hf_res = download_from_huggingface()
    if not hf_res:
        download_and_extract()
