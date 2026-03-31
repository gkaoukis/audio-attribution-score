import os
import zipfile
from huggingface_hub import snapshot_download

def download_and_unzip():
    repo_id = "Octavian97/Echoes"
    local_dir = "echoes"

    # 1. Download the dataset from Hugging Face
    print(f"Downloading dataset '{repo_id}' into '{local_dir}/'...")
    # snapshot_download automatically handles large files and resumes interrupted downloads
    snapshot_download(
        repo_id=repo_id, 
        repo_type="dataset", 
        local_dir=local_dir
    )
    print("Download complete!\n")

    # 2. Find and unzip any .zip files inside the downloaded folder
    print("Searching for .zip files to extract...")
    zip_found = False
    
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            if file.endswith(".zip"):
                zip_found = True
                zip_path = os.path.join(root, file)
                
                print(f"Unzipping: {zip_path}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # Extract in the same directory where the zip is located
                    zip_ref.extractall(root)
                print(f"Successfully extracted: {file}")
                
    if not zip_found:
        print("No .zip files were found in the dataset.")

if __name__ == "__main__":
    download_and_unzip()