import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load the model and preprocessor
@st.cache_resource
def load_model_and_preprocessor():
    with open("knn_best_tuned_model (1).pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("preprocessor (2).pkl", "rb") as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)
    return model, preprocessor

model, preprocessor = load_model_and_preprocessor()

# Try to infer feature names from preprocessor (assuming it's a Pipeline or ColumnTransformer)
try:
    # If it's a Pipeline
    if hasattr(preprocessor, 'named_transformers_'):
        feature_names = preprocessor.feature_names_in_
    elif hasattr(preprocessor, 'feature_names_in_'):
        feature_names = preprocessor.feature_names_in_
    else:
        st.error("Could not determine feature names from preprocessor.")
        st.stop()
except Exception as e:
    st.error(f"Error retrieving feature names: {e}")
    st.stop()

st.title("🔍 KNN Model Predictor - Manual Input")

st.markdown("Please enter the values for each feature below:")

# Collect feature values from the user
input_data = {}

with st.form("feature_form"):
    for feature in feature_names:
        # You can customize widget types per feature based on your data schema
        value = st.text_input(f"{feature}", "")
        input_data[feature] = value
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Convert numeric columns to float if possible
        for col in input_df.columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col])
            except ValueError:
                pass  # Leave as is (for categorical columns)

        # Preprocess input
        processed_input = preprocessor.transform(input_df)

        # Predict
        prediction = model.predict(processed_input)[0]

        st.success(f"✅ Prediction: {prediction}")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
