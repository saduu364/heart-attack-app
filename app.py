
import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessor
model = joblib.load("final_ensemble_model.pkl")
preprocessor = joblib.load("final_preprocessor.pkl")

# App title
st.title("💓 Heart Attack Risk Prediction App (Ensemble Model)")
st.write("Enter the patient's details below:")

# Input form
age = st.slider("Age", 20, 90, 50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
chest_pain_type = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
resting_blood_pressure = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
serum_cholesterol = st.number_input("Serum Cholesterol", min_value=100, max_value=600, value=200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=250, value=150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0, step=0.1)
slope = st.selectbox("Slope of ST Segment", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (0-3)", [0, 1, 2, 3])

# Predict Button
if st.button("Predict Heart Attack Risk"):
    input_data = pd.DataFrame([{
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
    transformed_input = preprocessor.transform(input_data)
    prediction = model.predict(transformed_input)[0]
    probability = model.predict_proba(transformed_input)[0][1]
    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Attack ({probability:.2%})")
    else:
        st.success(f"✅ Low Risk of Heart Attack ({probability:.2%})")
