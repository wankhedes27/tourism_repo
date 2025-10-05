from huggingface_hub import HfApi
import os

# -----------------------------
# Authentication and Constants
# -----------------------------
api = HfApi(token=os.getenv("HF_TOKEN"))
#api = HfApi(token=os.getenv("HF_TOKEN"))
api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id="wankhedes27/tourism-project",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
