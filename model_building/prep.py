
"""
Data Preparation Module
-----------------------
Loads dataset from Hugging Face, performs train/test split,
and uploads prepared splits back to the dataset repository.
"""

# ============================================================
# Hugging Face Authentication
# ============================================================

HF_TOKEN = keyring.get_password("huggingface", "gl_access_token_travel_project")
if not HF_TOKEN:
    raise RuntimeError("Hugging Face token not found in macOS Keychain.")

os.environ["HF_HUB_TOKEN"] = HF_TOKEN

# ============================================================
# Configuration
# ============================================================
DATASET_REPO = "praveenchugh/tourism-package-prediction-dataset"
DATASET_FILENAME = "tourism.csv"
DATASET_PATH = f"hf://datasets/{DATASET_REPO}/{DATASET_FILENAME}"

TARGET_COL = "ProdTaken"
TEST_SIZE = 0.2
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "Age", "CityTier", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "NumberOfTrips", "NumberOfChildrenVisiting", "MonthlyIncome",
    "PitchSatisfactionScore", "NumberOfFollowups", "DurationOfPitch",
]

CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "MaritalStatus",
    "Designation", "ProductPitched", "Passport", "OwnCar",
]


# ============================================================
# Functions
# ============================================================
def get_hf_client() -> HfApi:
    """Return an authenticated Hugging Face client."""
    return HfApi(token=os.environ["HF_HUB_TOKEN"])


def load_dataset() -> pd.DataFrame:
    """Load dataset from Hugging Face storage."""
    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded successfully.")
    return df


def prepare_data(df: pd.DataFrame):
    """Split dataset into training and testing sets and save locally."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COL]

    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print("Data split completed and saved locally.")
    return Xtrain, Xtest, ytrain.squeeze(), ytest.squeeze()


def upload_dataset_splits(api: HfApi):
    """Upload prepared dataset splits to the Hugging Face dataset repository."""
    for file in ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]:
        api.upload_file(
            path_or_fileobj=file,
            path_in_repo=file,
            repo_id=DATASET_REPO,
            repo_type="dataset",
        )

    print("Dataset splits uploaded to Hugging Face.")


# ============================================================
# Entry Point
# ============================================================
def main():
    api = get_hf_client()
    df = load_dataset()
    prepare_data(df)
    upload_dataset_splits(api)


if __name__ == "__main__":
    main()
