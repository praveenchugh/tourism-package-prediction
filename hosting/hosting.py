"""
Hugging Face Space Deployment Utility
-------------------------------------
Uploads the local Streamlit deployment folder to a Hugging Face Space.

Assumptions:
- Authentication is already configured via `get_hf_client()`.
- The target Space repository already exists or will be created separately.
"""

# ============================================================
# Initialize authenticated Hugging Face client
# ============================================================
api = HfApi(token=os.environ["HF_HUB_TOKEN"])


# ============================================================
# Configuration
# ============================================================
LOCAL_DEPLOYMENT_FOLDER = "tourism_project/deployment"
SPACE_REPO_ID = "praveenchugh/tourism-package-prediction"
SPACE_SUBFOLDER_PATH = ""  # Optional path inside the Space repository


# ============================================================
# Upload deployment files to Hugging Face Space
# ============================================================
api.upload_folder(
    folder_path=LOCAL_DEPLOYMENT_FOLDER,  # Local directory containing Streamlit app and dependencies
    repo_id=SPACE_REPO_ID,                # Target Hugging Face Space repository
    repo_type="space",                    # Specifies repository type as Space
    path_in_repo=SPACE_SUBFOLDER_PATH,    # Upload location inside the repo (root if empty)
)

print("Deployment folder uploaded successfully to Hugging Face Space.")

