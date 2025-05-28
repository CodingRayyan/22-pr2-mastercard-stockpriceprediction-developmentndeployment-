import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import streamlit as st
from tensorflow.keras.models import load_model

from utils.preprocessing import split_sequence  # Ensure this is correctly implemented

# Load dataset
dataset = pd.read_csv("Mastercard_stock_history.csv")
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset.set_index('Date', inplace=True)

# Parameters
tstart = 2016
tend = 2020
n_steps = 60
features = 1

# Split data
def train_test_split(dataset, tstart, tend):
    train = dataset.loc[f"{tstart}":f"{tend}", "High"].values
    test = dataset.loc[f"{tend+1}":, "High"].values
    return train, test

training_set, test_set = train_test_split(dataset, tstart, tend)

# Load scaler
scaler_path = os.path.join("model", "stckmark_scaler.save")
sc = joblib.load(scaler_path)

# Prepare input for prediction
dataset_total = dataset["High"]
inputs = dataset_total[len(dataset_total) - len(test_set) - n_steps :].values
inputs = inputs.reshape(-1, 1)
inputs_scaled = sc.transform(inputs)

# Create sequences
X_test, y_test = split_sequence(inputs_scaled, n_steps)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], features))

# Streamlit App UI
st.set_page_config(page_title="Stock Price Predictor", layout="centered")
st.title("📈 MasterCard Stock Price Prediction")
st.markdown("Choose a model to view predicted vs actual prices.")

model_choice = st.selectbox("Select Model", ["LSTM", "GRU"])

if model_choice:
    model_file = "model_lstm.keras" if model_choice == "LSTM" else "model_gru.keras"
    model_path = os.path.join("model", model_file)
    model = load_model(model_path)

    predicted = model.predict(X_test)
    predicted = sc.inverse_transform(predicted)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(test_set, label="Real", color="gray")
    ax.plot(predicted, label="Predicted", color="red")
    ax.set_title(f"{model_choice} Prediction")
    ax.set_xlabel("Time")
    ax.set_ylabel("Stock Price")
    ax.legend()
    ax.grid()

    st.pyplot(fig)

    rmse = np.sqrt(np.mean((test_set - predicted.flatten())**2))
    st.success(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

st.markdown("---")
st.subheader("📤 Predict the Next Stock Price")
st.markdown("Enter the last 60 stock prices (comma-separated):")

user_input = st.text_area("Input Sequence", placeholder="e.g. 362.5, 365.1, 368.7, ... (60 values)")

if st.button("Predict Next Price"):
    try:
        # Parse and validate input
        input_list = [float(x.strip()) for x in user_input.split(",")]
        if len(input_list) != 60:
            st.error("Please enter exactly 60 values.")
        else:
            input_array = np.array(input_list).reshape(-1, 1)
            input_scaled = sc.transform(input_array)  # use same scaler
            input_scaled = input_scaled.reshape(1, n_steps, features)

            prediction = model.predict(input_scaled)
            predicted_price = sc.inverse_transform(prediction)[0][0]

            st.success(f"📈 Predicted Next Stock Price: ${predicted_price:.3f}")
    except Exception as e:
        st.error(f"Invalid input! Make sure you provide 60 valid float numbers.\n\nError: {e}")
