"""
Download competition data from Kaggle using Kaggle API

Competition: ACM ICAIF '25 AI Agentic Retrieval Grand Challenge
Downloads and extracts 4 JSONL files to data/ directory
"""

import os
import zipfile
from pathlib import Path
from dotenv import load_dotenv

# Setup project paths (auto-finds root via pyproject.toml)
from src.utils.project_paths import PROJECT_ROOT
from config import PATHS

# Load environment variables
load_dotenv()

# Import kaggle after loading env
from kaggle.api.kaggle_api_extended import KaggleApi


def download_competition_data():
    """Download and extract competition data from Kaggle"""
    
    # Competition name
    competition = "acm-icaif-25-ai-agentic-retrieval-grand-challenge"
    
    # Verify credentials
    username = os.getenv('KAGGLE_USERNAME')
    key = os.getenv('KAGGLE_KEY')
    
    if not username or not key:
        print("Error: KAGGLE_USERNAME and KAGGLE_KEY must be set in .env file")
        return False
    
    print(f"Kaggle Username: {username}")
    print(f"Competition: {competition}")
    
    # Initialize and authenticate API
    api = KaggleApi()
    api.authenticate()
    
    # Setup data path
    data_path = PATHS['raw_data']
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Download competition files
    print(f"\nDownloading to: {data_path}")
    api.competition_download_files(competition, path=data_path)
    print("Download complete")
    
    # Unzip the downloaded file
    zip_file = data_path / f"{competition}.zip"
    if zip_file.exists():
        print(f"\nUnzipping: {zip_file.name}")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        print("Extraction complete")
        
        # Remove zip file
        zip_file.unlink()
        print(f"Removed: {zip_file.name}")
    else:
        print(f"Warning: zip file not found at {zip_file}")
    
    # List downloaded files
    print(f"\n{'='*50}")
    print("Downloaded files:")
    jsonl_files = sorted(data_path.glob("*.jsonl"))
    for f in jsonl_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")
    print(f"{'='*50}\n")
    
    return True


if __name__ == "__main__":
    download_competition_data()
