
import streamlit as st
import numpy as np
import joblib

# Load the pipeline (preprocessor + model)
pipeline = joblib.load("final_heart_attack_model.pkl")

st.title("💓 Heart Attack Risk Prediction App")
st.markdown("Fill in the patient's health details below:")

# Input fields
age = st.slider("Age", 18, 100, 45)
sex = st.selectbox("Sex", ["Male", "Female"])
chest_pain_type = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
resting_blood_pressure = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
serum_cholesterol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", [0, 1])
restecg = st.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1 = yes, 0 = no)", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", [0, 1, 2, 3, 4])
thal = st.selectbox("Thalassemia (1 = normal, 2 = fixed defect, 3 = reversible defect)", [0, 1, 2, 3])

# Convert categorical variables
sex_val = 1 if sex == "Male" else 0

# Arrange input in the required order
input_data = np.array([[age, sex_val, chest_pain_type, resting_blood_pressure,
                        serum_cholesterol, fasting_blood_sugar, restecg, thalach,
                        exang, oldpeak, slope, ca, thal]])

# Predict
if st.button("Predict Heart Attack Risk"):
    prediction = pipeline.predict(input_data)[0]
    probability = pipeline.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease! Probability: {probability:.2f}")
    else:
        st.success(f"✅ Low Risk of Heart Disease. Probability: {probability:.2f}")
