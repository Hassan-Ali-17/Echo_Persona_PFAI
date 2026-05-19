import os
import urllib.request
import zipfile
import shutil

# Zenodo link for RAVDESS audio only (Speech) - approx 1 GB
RAVDESS_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
DOWNLOAD_PATH = "data/raw/ravdess.zip"
EXTRACT_PATH = "data/raw/ravdess"

def download_progress(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    if percent % 10 == 0:
        print(f"Downloading RAVDESS dataset: {percent}%", end='\r')

def setup_dataset():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/features", exist_ok=True)

    if not os.path.exists(DOWNLOAD_PATH):
        print("Starting download of RAVDESS dataset (~1 GB). This may take a few minutes...")
        urllib.request.urlretrieve(RAVDESS_URL, DOWNLOAD_PATH, reporthook=download_progress)
        print("\nDownload complete.")
    else:
        print("RAVDESS zip file already exists.")

    if not os.path.exists(EXTRACT_PATH):
        print("Extracting files...")
        with zipfile.ZipFile(DOWNLOAD_PATH, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_PATH)
        print("Extraction complete.")
        
        # Flatten directory structure
        print("Organizing files...")
        for root, dirs, files in os.walk(EXTRACT_PATH):
            for file in files:
                if file.endswith(".wav"):
                    source_path = os.path.join(root, file)
                    target_path = os.path.join("data/raw", file)
                    shutil.move(source_path, target_path)
                    
        # Remove the extracted folder structure (it's now empty)
        shutil.rmtree(EXTRACT_PATH)
        print("Files successfully organized in data/raw/")
    else:
        print("Dataset already extracted.")

if __name__ == "__main__":
    setup_dataset()
