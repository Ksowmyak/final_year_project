import streamlit as st
import json
import bcrypt
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load trained LSTM model and scaler
lstm_model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")

# Feature names
feature_names = ["Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
                 "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm",
                 "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", "RISK_MM"]

# File to store user data
USER_FILE = "users.json"

# Load or initialize users
def load_users():
    try:
        with open(USER_FILE, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_users(users):
    with open(USER_FILE, "w") as file:
        json.dump(users, file, indent=4)

users = load_users()

# User authentication
def login_user(username, password):
    if username in users and bcrypt.checkpw(password.encode(), users[username]["password"].encode()):
        return True
    return False

def register_user(username, password):
    if username in users:
        return False
    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    users[username] = {"password": hashed_pw}
    save_users(users)
    return True

# App
st.set_page_config(page_title="Rain Prediction App", layout="centered")
st.title("🌦️ Rain Prediction using LSTM")

# Session state
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# Sidebar menu
menu = ["Login", "Register"] if not st.session_state.logged_in else ["Predict", "Logout"]
choice = st.sidebar.selectbox("Menu", menu)

# Login
if choice == "Login":
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login_user(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Welcome back, {username}!")
        else:
            st.error("Invalid username or password.")

# Register
elif choice == "Register":
    st.subheader("Register")
    new_user = st.text_input("New Username")
    new_pass = st.text_input("New Password", type="password")
    if st.button("Register"):
        if register_user(new_user, new_pass):
            st.success("Registration successful! You can now log in.")
        else:
            st.error("Username already exists.")

# Logout
elif choice == "Logout":
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.success("You have been logged out.")

# Predict
elif choice == "Predict" and st.session_state.logged_in:
    st.subheader("Rainfall Prediction")
    inputs = []
    for feature in feature_names:
        val = st.number_input(f"{feature}", value=0.0, format="%.2f")
        inputs.append(val)

    if st.button("Predict"):
        try:
            user_array = np.array(inputs).reshape(1, -1)
            user_scaled = scaler.transform(user_array)
            user_scaled_reshaped = user_scaled.reshape((1, 1, len(feature_names)))
            prediction_prob = lstm_model.predict(user_scaled_reshaped)[0][0]
            prediction = "🌧️ Rain Expected" if prediction_prob > 0.5 else "☀️ No Rain"
            st.success(f"Prediction: {prediction}")
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
