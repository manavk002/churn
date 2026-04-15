import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_ann(input_dim):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.3))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

pipeline = joblib.load("churn_full_pipeline.pkl")

st.title("Customer Churn Prediction App")

credit_score = st.number_input("Credit Score", 300, 900, 600)
age = st.number_input("Age", 18, 100, 30)
tenure = st.number_input("Tenure", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
num_products = st.number_input("Number of Products", 1, 4, 1)
salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])

has_card = st.selectbox("Has Credit Card", [0, 1])
is_active = st.selectbox("Is Active Member", [0, 1])

if st.button("Predict"):
    
    input_df = pd.DataFrame([{
        'CreditScore': credit_score,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'EstimatedSalary': salary,
        'Geography': geography,
        'Gender': gender,
        'HasCrCard': has_card,
        'IsActiveMember': is_active
    }])
    
    prediction = pipeline.predict(input_df)[0]

    if prediction == 1:
        st.error(" Customer will CHURN")
    else:
        st.success(" Customer will NOT churn")