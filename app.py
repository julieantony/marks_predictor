import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load the pre-trained Linear Regression model
# (Make sure 'linear_regression_model.pkl' is uploaded to the same folder on GitHub)
model = joblib.load('linear_regression_model.pkl')

# 2. Set up the app page configuration
st.set_page_config(page_title="Student Marks Predictor", page_icon="🎓", layout="centered")

# App Header
st.title("🎓 Student Marks Prediction App")
st.write("This application predicts a student's final marks based on their daily routine and academic history using a Linear Regression model.")

st.markdown("---")

# 3. Input Section Header
st.header("📋 Enter Student Information:")

# Creating input elements for the 5 features
study_hours = st.number_input("Study Hours per Day", min_value=0.0, max_value=24.0, value=5.0, step=0.1)
attendance = st.number_input("Attendance Percentage (%)", min_value=0.0, max_value=100.0, value=85.0, step=1.0)
sleep_hours = st.number_input("Sleep Hours per Night", min_value=0.0, max_value=24.0, value=7.0, step=0.1)
previous_score = st.number_input("Previous Exam Score", min_value=0.0, max_value=100.0, value=75.0, step=1.0)
internet_hours = st.number_input("Internet Usage Hours per Day", min_value=0.0, max_value=24.0, value=2.0, step=0.1)

st.markdown("---")

# 4. Prediction Trigger Button
if st.button("🔮 Predict Student Marks"):
    
    # FIX: Convert inputs into a pandas DataFrame with the EXACT matching column names 
    # that your scikit-learn Linear Regression model was trained on.
    input_df = pd.DataFrame([{
        'Study_Hours': study_hours,
        'Attendance_Percentage': attendance,
        'Sleep_Hours': sleep_hours,
        'Previous_Score': previous_score,
        'Internet_Hours': internet_hours
    }])
    
    # Perform prediction using the structured DataFrame
    prediction = model.predict(input_df)[0]
    
    # Since exam marks are bounded, clip the prediction so it stays between 0 and 100
    final_score = max(0.0, min(100.0, prediction))
    
    # Display the final prediction result beautifully
    st.success(f"🎯 The predicted marks for this student are: **{final_score:.2f} / 100**")