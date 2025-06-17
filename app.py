import streamlit as st
import pickle
import numpy as np

# Load model and preprocessor using pickle
with open('preprocessor.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

with open('knn_best_tuned_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Heart Attack Risk Predictor", layout="centered")
st.title("❤️ Heart Attack Risk Prediction App")

st.write("Enter patient details below:")

# Define user input fields
age = st.slider("Age", 20, 100, 50)
sex = st.selectbox("Sex (0 = female, 1 = male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
trestbps = st.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl? (1 = True, 0 = False)", [0, 1])
restecg = st.selectbox("Resting ECG (0–2)", [0, 1, 2])
thalach = st.slider("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1 = yes, 0 = no)", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", value=1.0, step=0.1)
slope = st.selectbox("Slope of ST Segment (0–2)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (1 = normal, 2 = fixed defect, 3 = reversable defect)", [1, 2, 3])

# Collect input into a feature vector
input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach,
                        exang, oldpeak, slope, ca, thal]])

# Predict button
if st.button("Predict"):
    # Preprocess the input
    input_transformed = preprocessor.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_transformed)

    # Display result
    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Attack!")
    else:
        st.success("✅ Low Risk of Heart Attack.")
