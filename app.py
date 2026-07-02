import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import pandas as pd
import pickle

# ---------------------- Page Configuration ---------------------- #
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide"
)

# ---------------------- Load Model ---------------------- #
model = load_model("model.h5")

with open("onehot_encoder_geo.pkl", "rb") as file:
    onehot_encoder_geo = pickle.load(file)

with open("label_encoder_gender.pkl", "rb") as file:
    label_encoder_gender = pickle.load(file)

with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# ---------------------- Custom CSS ---------------------- #
st.markdown("""
<style>

.main-title{
    text-align:center;
    color:#1565C0;
    font-size:42px;
    font-weight:bold;
}

.sub-title{
    text-align:center;
    color:gray;
    font-size:18px;
}

.stButton > button{
    width:100%;
    background-color:#1565C0;
    color:white;
    height:50px;
    border-radius:10px;
    font-size:18px;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ---------------------- Sidebar ---------------------- #

with st.sidebar:
    st.title("🏦 Customer Churn")
    st.markdown("---")
    st.write("### About")
    st.write(
        """
        Predict whether a customer is likely to leave the bank
        using an Artificial Neural Network.
        """
    )

    st.markdown("---")
    st.write("**Model:** ANN")
    st.write("**Framework:** TensorFlow/Keras")
    st.write("**Frontend:** Streamlit")

# ---------------------- Title ---------------------- #

st.markdown(
    "<h1 class='main-title'>🏦 Bank Customer Churn Prediction</h1>",
    unsafe_allow_html=True,
)

st.markdown(
    "<p class='sub-title'>Artificial Neural Network powered prediction system</p>",
    unsafe_allow_html=True,
)

st.markdown("---")

# ---------------------- Input Layout ---------------------- #

col1, col2 = st.columns(2)

with col1:

    credit_score = st.slider(
        "Credit Score",
        300,
        900,
        650
    )

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )

    age = st.slider(
        "Age",
        18,
        100,
        35
    )

    tenure = st.slider(
        "Tenure",
        0,
        10,
        5
    )

with col2:

    balance = st.number_input(
        "Balance",
        min_value=0.0,
        value=60000.0,
        step=1000.0
    )

    num_products = st.slider(
        "Number of Products",
        1,
        4,
        2
    )

    has_card = st.selectbox(
        "Has Credit Card?",
        ["Yes", "No"]
    )

    active_member = st.selectbox(
        "Is Active Member?",
        ["Yes", "No"]
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

st.markdown("---")

# ---------------------- Predict Button ---------------------- #

if st.button("Predict Customer Churn"):

    has_card = 1 if has_card == "Yes" else 0
    active_member = 1 if active_member == "Yes" else 0

    input_data = pd.DataFrame({

        "CreditScore":[credit_score],
        "Geography":[geography],
        "Gender":[gender],
        "Age":[age],
        "Tenure":[tenure],
        "Balance":[balance],
        "NumOfProducts":[num_products],
        "HasCrCard":[has_card],
        "IsActiveMember":[active_member],
        "EstimatedSalary":[estimated_salary]

    })

    # Encode Gender
    input_data["Gender"] = label_encoder_gender.transform(
        input_data["Gender"]
    )

    # Encode Geography
    geo_encoded = onehot_encoder_geo.transform(
        input_data[["Geography"]]
    ).toarray()

    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(),
        index=input_data.index
    )

    input_data = input_data.drop("Geography", axis=1)

    input_data = pd.concat(
        [input_data, geo_encoded_df],
        axis=1
    )

    # Scale
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    probability = prediction[0][0]

    st.markdown("## Prediction Result")

    st.progress(float(probability))

    st.metric(
        label="Churn Probability",
        value=f"{probability:.2%}"
    )

    if probability > 0.5:

        st.error("⚠️ Customer is likely to churn.")

    else:

        st.success("✅ Customer is likely to stay.")

    st.markdown("---")

    st.subheader("Customer Details")

    st.dataframe(
        pd.DataFrame({
            "Feature":[
                "Credit Score",
                "Geography",
                "Gender",
                "Age",
                "Tenure",
                "Balance",
                "Products",
                "Credit Card",
                "Active Member",
                "Estimated Salary"
            ],
            "Value":[
                credit_score,
                geography,
                gender,
                age,
                tenure,
                balance,
                num_products,
                "Yes" if has_card else "No",
                "Yes" if active_member else "No",
                estimated_salary
            ]
        }),
        use_container_width=True
    )

st.markdown("---")

st.caption(
    "Developed using TensorFlow, Scikit-learn and Streamlit"
)