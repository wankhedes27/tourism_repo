import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# -----------------------------
# Load the Tourism Lead Prediction Model
# -----------------------------
model_path = hf_hub_download(repo_id="wankhedes27/tourism-model", filename="best_tourism_model_v1.joblib")
model = joblib.load(model_path)

# -----------------------------
# Streamlit App Interface
# -----------------------------
st.title("Tourism Lead Conversion Prediction App")
st.write("This internal tool helps tourism agents predict whether a potential customer is likely to purchase a travel package based on their details.")
st.write("Please enter the customer details below to generate a prediction.")

# -----------------------------
# Collect user input
# -----------------------------
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.selectbox("City Tier", [1, 2, 3])
DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0, value=15)
Occupation = st.selectbox("Occupation", ["Salaried", "Self Employed", "Small Business", "Large Business"])
Gender = st.selectbox("Gender", ["Male", "Female"])
NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", min_value=1, value=2)
NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, value=1)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
PreferredPropertyStar = st.selectbox("Preferred Property Star Rating", [3, 4, 5])
MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
NumberOfTrips = st.number_input("Number of Trips Taken per Year", min_value=0, value=2)
Passport = st.selectbox("Has Passport?", ["Yes", "No"])
PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)
OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, value=0)
Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
MonthlyIncome = st.number_input("Monthly Income (in ₹)", min_value=0.0, value=50000.0)
Age = st.number_input("Customer Age", min_value=18, max_value=80, value=30)

# -----------------------------
# Prepare input for model
# -----------------------------
input_data = pd.DataFrame([{
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': 1 if Passport == "Yes" else 0,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': 1 if OwnCar == "Yes" else 0,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
    'Age': Age
}])

# -----------------------------
# Make prediction
# -----------------------------
classification_threshold = 0.45

if st.button("Predict Lead Conversion"):
    probability = model.predict_proba(input_data)[0, 1]
    prediction = int(probability >= classification_threshold)
    result = "interested in purchasing a tour package" if prediction == 1 else "not interested at the moment"

    st.subheader("Prediction Result")
    st.write(f"The customer is likely **{result}**.")
    st.write(f"Prediction Confidence: {probability:.2%}")
