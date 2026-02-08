"""
Streamlit Application: Travel Package Purchase Prediction
---------------------------------------------------------
This app:

1. Downloads the trained ML model from Hugging Face.
2. Collects customer demographic and interaction inputs.
3. Predicts probability of travel package purchase.
4. Displays prediction results in a clean UI.
"""

# ============================================================
# Imports
# ============================================================
import pandas as pd
import streamlit as st
import joblib
from huggingface_hub import hf_hub_download


# ============================================================
# Model Loading
# ============================================================
@st.cache_resource
def load_trained_model():
    """
    Download and load the trained model from Hugging Face.

    Returns
    -------
    model : sklearn-compatible model
        Loaded classification model with predict_proba support.
    """
    model_path = hf_hub_download(
        repo_id="praveenchugh/tourism-package-prediction-model",
        filename="best_mlops_tourism_model_v1.joblib",
    )
    return joblib.load(model_path)


model = load_trained_model()


# ============================================================
# Application Header
# ============================================================
st.title("Customer Travel Package Purchase Predictor")

st.markdown(
    """
    This internal tool estimates the likelihood that a customer will purchase
    a travel package based on demographic attributes and interaction history.

    Provide the required details below and click **Run Prediction**.
    """
)


# ============================================================
# Utility Functions
# ============================================================
def yes_no_to_int(value: str) -> int:
    """
    Convert Yes/No selection to binary integer.

    Parameters
    ----------
    value : str
        User selection ("Yes" or "No").

    Returns
    -------
    int
        1 if Yes, otherwise 0.
    """
    return 1 if value == "Yes" else 0


def collect_customer_inputs() -> pd.DataFrame:
    """
    Render Streamlit input widgets and return user inputs
    formatted as a single-row pandas DataFrame.
    """

    # ---------------- Demographic Information ----------------
    age = st.number_input("Age", min_value=18, max_value=100, value=30)

    contact_type = st.selectbox(
        "Contact Type",
        ["Company Invited", "Self Inquiry"],
    )

    city_tier = st.selectbox("City Tier", [1, 2, 3])

    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Freelancer", "Small Business", "Large Business"],
    )

    gender = st.selectbox("Gender", ["Male", "Female"])

    visitors = st.number_input(
        "Number of Persons Visiting", min_value=1, max_value=10, value=2
    )

    property_star = st.selectbox("Preferred Property Star", [1, 2, 3, 4, 5])

    marital_status = st.selectbox(
        "Marital Status",
        ["Single", "Married", "Divorced"],
    )

    yearly_trips = st.number_input(
        "Trips per Year", min_value=0, max_value=50, value=2
    )

    passport = yes_no_to_int(st.selectbox("Passport Available?", ["Yes", "No"]))
    car_owner = yes_no_to_int(st.selectbox("Owns a Car?", ["Yes", "No"]))

    children = st.number_input(
        "Children Visiting", min_value=0, max_value=5, value=0
    )

    designation = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager", "VP"],
    )

    income = st.number_input(
        "Monthly Income", min_value=5_000, max_value=500_000, value=50_000
    )

    # ---------------- Interaction Information ----------------
    satisfaction = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)

    product = st.selectbox(
        "Product Offered",
        ["Basic", "Standard", "Deluxe", "Super Deluxe"],
    )

    followups = st.number_input(
        "Number of Follow-ups", min_value=0, max_value=20, value=2
    )

    pitch_duration = st.number_input(
        "Pitch Duration (minutes)", min_value=1, max_value=120, value=15
    )

    # ---------------- Assemble DataFrame ----------------
    return pd.DataFrame(
        [
            {
                "Age": age,
                "TypeofContact": contact_type,
                "CityTier": city_tier,
                "Occupation": occupation,
                "Gender": gender,
                "NumberOfPersonVisiting": visitors,
                "PreferredPropertyStar": property_star,
                "MaritalStatus": marital_status,
                "NumberOfTrips": yearly_trips,
                "Passport": passport,
                "OwnCar": car_owner,
                "NumberOfChildrenVisiting": children,
                "Designation": designation,
                "MonthlyIncome": income,
                "PitchSatisfactionScore": satisfaction,
                "ProductPitched": product,
                "NumberOfFollowups": followups,
                "DurationOfPitch": pitch_duration,
            }
        ]
    )


# ============================================================
# Prediction Section
# ============================================================
PREDICTION_THRESHOLD = 0.5

user_input_df = collect_customer_inputs()

if st.button("Run Prediction"):
    """
    Execute prediction when user clicks the button.
    """

    # Predict probability of positive class
    purchase_probability = model.predict_proba(user_input_df)[0, 1]

    # Convert probability to binary decision
    predicted_label = int(purchase_probability >= PREDICTION_THRESHOLD)

    # ---------------- Display Results ----------------
    st.subheader("Prediction Result")
    st.write(f"Purchase Probability: {purchase_probability:.2%}")

    if predicted_label == 1:
        st.success("Customer is likely to purchase the travel package.")
    else:
        st.error("Customer is unlikely to purchase the travel package.")
