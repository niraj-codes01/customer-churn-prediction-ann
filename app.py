import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle


model = tf.keras.models.load_model('model.h5')

with open('onehot_encoder_geo.pkl','rb') as file:
    onehot_encoder_geo= pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

## Streamlit app
st.title("Customer Churn Prediction")

credit_score = st.number_input(
    "Credit Score",
    min_value=300,
    max_value=900,
    value=650
)

geography = st.selectbox(
    "Geography",
    ["France", "Germany", "Spain"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=100,
    value=35
)

tenure = st.number_input(
    "Tenure (Years)",
    min_value=0,
    max_value=10,
    value=5
)

balance = st.number_input(
    "Balance",
    min_value=0.0,
    value=60000.0,
    step=1000.0
)

num_of_products = st.number_input(
    "Number of Products",
    min_value=1,
    max_value=4,
    value=2
)

has_cr_card = st.selectbox(
    "Has Credit Card",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

is_active_member = st.selectbox(
    "Is Active Member",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

estimated_salary = st.number_input(
    "Estimated Salary",
    min_value=0.0,
    value=50000.0,
    step=1000.0
)

input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

if st.button("Predict Churn"):

    # Encode Gender
    input_data["Gender"] = label_encoder_gender.transform(input_data["Gender"])

    # One-Hot Encode Geography
    geo_encoded = onehot_encoder_geo.transform(input_data[["Geography"]]).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(),
        index=input_data.index
    )

    # Merge encoded columns
    input_data = input_data.drop("Geography", axis=1)
    input_data = pd.concat([input_data, geo_encoded_df], axis=1)

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)
    probability = prediction[0][0]

    st.subheader("Prediction Result")
    st.write(f"**Churn Probability:** {probability:.2%}")

    if probability > 0.5:
        st.error("⚠️ Customer is likely to churn.")
    else:
        st.success("✅ Customer is likely to stay.")

