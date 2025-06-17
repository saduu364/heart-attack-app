import streamlit as st
import numpy as np
import joblib

# Load model and preprocessor
model = joblib.load("final_ensemble_model.pkl")
preprocessor = joblib.load("preprocessor (2).pkl")

# App title
st.set_page_config(page_title="Heart Attack Prediction", layout="centered")
st.title("💓 Heart Attack Risk Prediction App")
st.write("Enter the medical details below to assess heart attack risk.")

# Function to collect input
def get_user_input():
    age = st.slider("Age", 20, 80, 50)
    sex = st.selectbox("Sex", [0, 1])  # 0 = Female, 1 = Male
    chest_pain_type = st.slider("Chest Pain Type", 0.0, 3.0, 1.0)
    resting_blood_pressure = st.slider("Resting Blood Pressure", 80.0, 200.0, 120.0)
    serum_cholesterol = st.slider("Serum Cholesterol", 100.0, 600.0, 250.0)
    fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.slider("Rest ECG", 0.0, 2.0, 1.0)
    thalach = st.slider("Max Heart Rate Achieved", 70.0, 210.0, 150.0)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    slope = st.slider("Slope of ST Segment", 0.0, 2.0, 1.0)
    ca = st.slider("Number of Major Vessels (0–3)", 0.0, 3.0, 0.0)
    thal = st.slider("Thalassemia", 0.0, 3.0, 1.0)

    features = np.array([[age, sex, chest_pain_type, resting_blood_pressure,
                          serum_cholesterol, fasting_blood_sugar, restecg,
                          thalach, exang, oldpeak, slope, ca, thal]])
    return features

# Get input
input_data = get_user_input()

# Predict
if st.button("Predict"):
    processed_input = preprocessor.transform(input_data)
    prediction = model.predict(processed_input)[0]
    probability = model.predict_proba(processed_input)[0][1]

    st.subheader("🔍 Prediction")
    st.write("**Risk of Heart Attack:**", "🚨 Yes" if prediction == 1 else "✅ No")
    st.progress(int(probability * 100))
    st.write(f"**Probability of Heart Attack:** {probability:.2%}")
