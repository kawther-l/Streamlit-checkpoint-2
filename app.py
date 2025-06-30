# Triggering redeploy
import streamlit as st
import pandas as pd
import joblib
from utils.preprocessing import preprocess_input  # your custom preprocessing
import gdown
import os
import numpy as np
import joblib

@st.cache_data
def preprocess_cached(df):
    return preprocess_input(df)

@st.cache_resource
def load_model():
    model_path = 'model/financial_model.pkl'
    # Load the model
    return joblib.load(model_path)


# Load once, cached
model = load_model()
model_columns = joblib.load('model/model_columns.pkl')
print(model_columns)

st.title("🔍 Financial Inclusion Prediction App")
st.write("Please fill out the form to predict whether the person is likely to have a bank account.")

# 1. User input form
with st.form("prediction_form"):
    year = st.selectbox("Year of Survey", [2016, 2017, 2018])
    country = st.selectbox("Country", ["Kenya", "Rwanda", "Tanzania", "Uganda"])
    location_type = st.selectbox("Location Type", ["Rural", "Urban"])
    cellphone_access = st.selectbox("Cellphone Access", ["Yes", "No"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age of Respondent", min_value=16, max_value=100, value=30)
    household_size = st.slider("Household Size", min_value=1, max_value=20, value=5)
    relationship_with_head = st.selectbox("Relationship with Head", [
        "Head of Household", "Spouse", "Child", "Parent", "Other relative", "Other non-relatives"
    ])
    marital_status = st.selectbox("Marital Status", [
        "Married/Living together", "Single/Never Married", "Divorced/Separated", "Widowed", "Don't know"
    ])
    education_level = st.selectbox("Education Level", [
        "No formal education", "Primary education", "Secondary education",
        "Tertiary education", "Vocational/Specialised training", "Other/Don't know/RTA"
    ])
    job_type = st.selectbox("Job Type", [
        "Self employed", "Government Dependent", "Formally employed Private",
        "Formally employed Government", "Informally employed", "Farming and Fishing",
        "Remittance Dependent", "No Income", "Don't know/Refuse to answer", "Other Income"
    ])

    submit = st.form_submit_button("Predict")

# 2. Prediction logic
@st.cache_resource
def load_label_encoder():
    return joblib.load('model/le_education_level.pkl')

le_education = load_label_encoder()
if submit:
    edu_level_encoded = le_education.transform([education_level])[0]
    input_dict = {
        "year":year,
        "age_of_respondent": age,
        "household_size": household_size,
        "country_" + country: 1,
        "location_type_" + location_type: 1,
        "cellphone_access_" + cellphone_access: 1,
        "gender_of_respondent_" + gender: 1,
        "relationship_with_head_" + relationship_with_head: 1,
        "marital_status_" + marital_status: 1,
        "job_type_" + job_type: 1,
        "education_level_encoded": edu_level_encoded,
    }

    # Create a blank input vector with all model columns
    input_vector = pd.DataFrame([np.zeros(len(model_columns))], columns=model_columns)

    # Fill in only the values present in the form
    for col, val in input_dict.items():
        if col in input_vector.columns:
            input_vector[col] = val

    # Predict
    prediction = model.predict(input_vector)[0]
    proba = model.predict_proba(input_vector)[0][1]

    if prediction == 1:
        st.success(f"✅ This person is likely to have a bank account. (Probability: {proba:.2%})")
    else:
        st.warning(f"⚠️ This person is unlikely to have a bank account. (Probability: {proba:.2%})")


