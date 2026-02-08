"""
Hugging Face Space Deployment Utility
-------------------------------------
Uploads the local Streamlit deployment folder to a Hugging Face Space.

Authentication:
- Uses HF token from environment variable `HF_TOKEN`.
"""

# ============================================================
# Imports
# ============================================================
import os

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError


# ============================================================
# Configuration
# ============================================================
HF_TOKEN_ENV = "HF_TOKEN"

LOCAL_DEPLOYMENT_FOLDER = "deployment"
SPACE_REPO_ID = "praveenchugh/tourism-package-prediction"
SPACE_SUBFOLDER_PATH = ""  # Optional subfolder inside Space repo


# ============================================================
# Authentication
# ============================================================
def get_hf_client() -> HfApi:
    """
    Create an authenticated Hugging Face client using environment token.
    """
    token = os.getenv(HF_TOKEN_ENV)

    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable not set. "
            "Configure it locally or in GitHub Actions secrets."
        )

    return HfApi(token=token)


# ============================================================
# Repository utilities
# ============================================================
def ensure_space_exists(api: HfApi, repo_id: str) -> None:
    """
    Ensure the Hugging Face Space repository exists.
    Creates it if missing.
    """
    try:
        api.repo_info(repo_id=repo_id, repo_type="space")
        print(f"Space repository already exists: {repo_id}")

    except RepositoryNotFoundError:
        print(f"Space not found. Creating: {repo_id}")
        create_repo(repo_id=repo_id, repo_type="space", private=False)
        print("Space created successfully.")


# ============================================================
# Deployment
# ============================================================
def upload_space(api: HfApi) -> None:
    """
    Upload local Streamlit deployment folder to Hugging Face Space.
    """
    if not os.path.exists(LOCAL_DEPLOYMENT_FOLDER):
        raise FileNotFoundError(
            f"Deployment folder not found: {LOCAL_DEPLOYMENT_FOLDER}"
        )

    print(f"Uploading deployment folder to Space: {SPACE_REPO_ID}")

    api.upload_folder(
        folder_path=LOCAL_DEPLOYMENT_FOLDER,
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        path_in_repo=SPACE_SUBFOLDER_PATH,
    )

    print("Deployment upload completed successfully.")


# ============================================================
# Main execution
# ============================================================
def main() -> None:
    """End-to-end Space deployment."""
    api = get_hf_client()
    ensure_space_exists(api, SPACE_REPO_ID)
    upload_space(api)


if __name__ == "__main__":
    main()
