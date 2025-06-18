
import streamlit as st
import pandas as pd
import joblib

# Load full pipeline
pipeline = joblib.load("pipeline_model.pkl")

st.title("💓 Heart Attack Risk Prediction (Using Full Pipeline)")
st.write("Enter patient's details below:")

# User input fields
age = st.slider("Age", 20, 90, 50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
chest_pain_type = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
resting_blood_pressure = st.number_input("Resting Blood Pressure", 80, 200, 120)
serum_cholesterol = st.number_input("Serum Cholesterol", 100, 600, 200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 250, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, step=0.1)
slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

# Prediction
if st.button("Predict"):
    input_df = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "chest_pain_type": chest_pain_type,
        "resting_blood_pressure": resting_blood_pressure,
        "serum_cholesterol": serum_cholesterol,
        "fasting_blood_sugar": fasting_blood_sugar,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    prediction = pipeline.predict(input_df)[0]
    probability = pipeline.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Attack ({probability:.2%})")
    else:
        st.success(f"✅ Low Risk of Heart Attack ({probability:.2%})")
