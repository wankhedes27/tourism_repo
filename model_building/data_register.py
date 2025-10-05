from huggingface_hub import HfApi, create_repo
from huggingface_hub.errors import RepositoryNotFoundError, HfHubHTTPError
import os
import sys

# --- 🔐 Load your HF token safely ---
token = os.getenv("HF_TOKEN")
if not token:
    sys.exit("❌ HF_TOKEN not found. Please set it in your Colab before running this script.")

# --- Initialize API client ---
api = HfApi(token=token)

# --- Repo details ---
repo_id = "wankhedes27/tourism-repo"
repo_type = "dataset"

# --- Step 1: Check or create the dataset repo ---
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"✅ Dataset repo '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"📦 Dataset repo '{repo_id}' not found. Creating it...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"✅ Dataset repo '{repo_id}' created.")

# --- Step 2: Upload folder contents ---
try:
    api.upload_folder(
        folder_path="tourism_project/data",  # adjust if needed
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print("✅ Data folder uploaded successfully.")
except Exception as e:
    print(f"❌ Upload failed: {e}")
