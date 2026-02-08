
"""
Hugging Face Authentication & Dataset Upload Utilities
-----------------------------------------------------
This module:

1. Stores and retrieves the Hugging Face access token securely
   using macOS Keychain via `keyring`.
2. Creates an authenticated Hugging Face API client.
3. Ensures a repository exists on the Hugging Face Hub.
4. Uploads a local dataset folder to the repository.
"""

# ============================================================
# Constants
# ============================================================
HF_SERVICE_NAME = "huggingface"
HF_USERNAME = "gl_access_token_travel_project"

REPO_ID = "praveenchugh/tourism-package-prediction-dataset"
REPO_TYPE = "dataset"
DATA_PATH = "tourism_project/data"


# ============================================================
# Token Setup (Run once manually if needed)
# ============================================================
def store_hf_token(token: str) -> None:
    """
    Store Hugging Face token securely in macOS Keychain.

    Parameters
    ----------
    token : str
        Hugging Face personal access token.
    """
    keyring.set_password(HF_SERVICE_NAME, HF_USERNAME, token)
    print("✅ Hugging Face token stored securely in Keychain.")


# ============================================================
# Authentication Helpers
# ============================================================
def configure_huggingface_auth() -> None:
    """
    Load token from Keychain and set HF_HUB_TOKEN
    environment variable for downstream libraries.
    """
    token = keyring.get_password(HF_SERVICE_NAME, HF_USERNAME)

    if not token:
        raise RuntimeError("❌ Hugging Face token not found in Keychain.")

    os.environ["HF_HUB_TOKEN"] = token
    print("🔐 Hugging Face authentication configured via environment variable.")


def get_hf_client() -> HfApi:
    """
    Create an authenticated Hugging Face API client.

    Returns
    -------
    HfApi
        Authenticated Hugging Face client.
    """
    token = keyring.get_password(HF_SERVICE_NAME, HF_USERNAME)

    if not token:
        raise RuntimeError(
            "❌ Unable to retrieve Hugging Face token from macOS Keychain."
        )

    print("✅ Hugging Face client initialized.")
    return HfApi(token=token)


# ============================================================
# Repository Utilities
# ============================================================
def ensure_repo_exists(api: HfApi, repo_id: str, repo_type: str) -> None:
    """
    Ensure the specified Hugging Face repository exists.
    Creates the repo if it does not exist.

    Parameters
    ----------
    api : HfApi
        Authenticated Hugging Face client.
    repo_id : str
        Repository identifier (e.g., 'username/repo-name').
    repo_type : str
        Repository type ('dataset', 'model', or 'space').
    """
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"✔ Repository already exists: {repo_id}")

    except RepositoryNotFoundError:
        print(f"➕ Repository not found. Creating: {repo_id}")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"✅ Repository created successfully: {repo_id}")


# ============================================================
# Dataset Upload
# ============================================================
def upload_dataset(api: HfApi, repo_id: str, data_path: str) -> None:
    """
    Upload a local dataset folder to Hugging Face Hub.

    Parameters
    ----------
    api : HfApi
        Authenticated Hugging Face client.
    repo_id : str
        Dataset repository ID.
    data_path : str
        Local folder path containing dataset files.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Dataset path not found: {data_path}")

    print(f"📤 Uploading dataset from '{data_path}' to '{repo_id}'...")
    api.upload_folder(folder_path=data_path, repo_id=repo_id, repo_type="dataset")
    print("🎉 Dataset upload completed successfully.")


# ============================================================
# Main Execution
# ============================================================
def main() -> None:
    """
    End-to-end workflow:
    1. Configure authentication
    2. Initialize HF client
    3. Ensure repo exists
    4. Upload dataset
    """
    configure_huggingface_auth()
    hf_api = get_hf_client()

    ensure_repo_exists(hf_api, REPO_ID, REPO_TYPE)
    upload_dataset(hf_api, REPO_ID, DATA_PATH)


if __name__ == "__main__":
    main()
