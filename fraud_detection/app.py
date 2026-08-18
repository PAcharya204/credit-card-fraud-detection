import streamlit as st
import pandas as pd
import joblib


# --------------------------------------------------
# Load trained model and scaler
# --------------------------------------------------

model = joblib.load("fraud_detection_model.pkl")
scaler = joblib.load("scaler.pkl")


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)


# --------------------------------------------------
# Title and description
# --------------------------------------------------

st.title("💳 Credit Card Fraud Detection")

st.write(
    "Enter the transaction details below to predict "
    "whether the transaction is genuine or fraudulent."
)

st.info(
    "This application is a machine-learning demonstration "
    "based on the features used in the training dataset."
)


# --------------------------------------------------
# Transaction details
# --------------------------------------------------

st.subheader("Transaction Details")


# Distance from home
distance_from_home = st.number_input(
    "Distance of the transaction from your home",
    min_value=0.0,
    value=10.0,
    help=(
        "Enter the distance using the same units as the dataset. "
        "The dataset does not specify whether this value represents "
        "kilometres or metres."
    )
)


# Distance from previous transaction
distance_from_last_transaction = st.number_input(
    "Distance from your previous transaction",
    min_value=0.0,
    value=5.0,
    help=(
        "Enter the distance using the same units as the dataset. "
        "The dataset does not specify whether this value represents "
        "kilometres or metres."
    )
)


# Purchase price ratio
ratio_to_median_purchase_price = st.number_input(
    "Purchase price compared with the typical purchase price",
    min_value=0.0,
    value=1.0,
    help=(
        "This is the ratio of the transaction purchase price "
        "to the median purchase price represented in the dataset. "
        "For example, 1 means the transaction is equal to the "
        "median purchase price, while 2 means it is twice the median."
    )
)


# Repeat retailer
repeat_retailer = st.selectbox(
    "Have you purchased from this retailer before?",
    ["Yes", "No"]
)


# Chip
used_chip = st.selectbox(
    "Was a chip used for this transaction?",
    ["Yes", "No"]
)


# PIN
used_pin_number = st.selectbox(
    "Was a PIN used for this transaction?",
    ["Yes", "No"]
)


# Online order
online_order = st.selectbox(
    "Was this an online order?",
    ["Yes", "No"]
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Predict Transaction", use_container_width=True):

    # Convert Yes/No values to the 0/1 values
    # expected by the trained model

    repeat_retailer_value = 1 if repeat_retailer == "Yes" else 0
    used_chip_value = 1 if used_chip == "Yes" else 0
    used_pin_number_value = 1 if used_pin_number == "Yes" else 0
    online_order_value = 1 if online_order == "Yes" else 0


    # Create input DataFrame
    # IMPORTANT: Keep the same feature order as training

    input_data = pd.DataFrame([[
        distance_from_home,
        distance_from_last_transaction,
        ratio_to_median_purchase_price,
        repeat_retailer_value,
        used_chip_value,
        used_pin_number_value,
        online_order_value
    ]], columns=[
        "distance_from_home",
        "distance_from_last_transaction",
        "ratio_to_median_purchase_price",
        "repeat_retailer",
        "used_chip",
        "used_pin_number",
        "online_order"
    ])


    # Apply the same scaler used during model training

    input_scaled = scaler.transform(input_data)


    # Make prediction

    prediction = model.predict(input_scaled)[0]


    # Display result

    st.subheader("Prediction")

    if prediction == 1:

        st.error(
            "🚨 Fraudulent Transaction"
        )

        st.write(
            "The model predicts that this transaction may be fraudulent."
        )

    else:

        st.success(
            "✅ Genuine Transaction"
        )

        st.write(
            "The model predicts that this transaction is likely to be genuine."
        )