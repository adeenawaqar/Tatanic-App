import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np

st.title("Titanic Survival Prediction App")
st.write("Enter passenger details below to predict survival:")

# --- User Inputs ---
pclass = st.selectbox("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd):", [1, 2, 3])
sex = st.selectbox("Sex:", ["Male", "Female"])
age = st.number_input("Age:", min_value=0, max_value=100, value=30)
sibsp = st.number_input("Number of Siblings/Spouses aboard:", min_value=0, max_value=10, value=0)
parch = st.number_input("Number of Parents/Children aboard:", min_value=0, max_value=10, value=0)
fare = st.number_input("Fare:", min_value=0.0, max_value=1000.0, value=32.0)
embarked = st.selectbox("Port of Embarkation (0=Cherbourg, 1=Queenstown, 2=Southampton):", [0, 1, 2])

# Encode sex
sex_encoded = 1 if sex == "Male" else 0

# Create DataFrame for prediction
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex_encoded],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked]
})

# --- Simple Logistic Regression Model Definition ---
# Example: dummy coefficients for demonstration (replace with real model if needed)
# Features order: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
model = LogisticRegression()
model.coef_ = np.array([[ -1.0, 1.5, -0.03, -0.2, -0.1, 0.01, 0.3]])
model.intercept_ = np.array([0.5])
model.classes_ = np.array([0, 1])

# --- Prediction ---
if st.button("Predict Survival"):
    # Logistic function manually
    z = np.dot(input_data.values, model.coef_.T) + model.intercept_
    prob = 1 / (1 + np.exp(-z))
    prediction = (prob >= 0.5).astype(int)[0][0]
    
    if prediction == 1:
        st.success(f"The passenger **survived** with probability {prob[0][0]:.2f}")
    else:
        st.error(f"The passenger **did not survive** with probability {1-prob[0][0]:.2f}")
