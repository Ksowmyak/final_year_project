import streamlit as st
import json
import bcrypt
import numpy as np
import joblib
import xgboost as xgb
from tensorflow.keras.models import load_model
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load scaler
scaler = joblib.load("scaler.pkl")

# Feature names
feature_names = ["Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
                 "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm",
                 "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", "RISK_MM"]

# Load user data
USER_FILE = "users.json"
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

# Page config
st.set_page_config(page_title="Multi-Model Rain Prediction", layout="centered")
st.title("🌦️ Rain Prediction with Multiple Models")

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

    model_choice = st.selectbox("Select Model", [
        "ANN (LSTM)", "Logistic Regression", "Decision Tree",
        "Random Forest", "Naive Bayes", "SVM", "XGBoost"
    ])

    inputs = []
    for feature in feature_names:
        val = st.number_input(f"{feature}", value=0.0, format="%.2f")
        inputs.append(val)

    if st.button("Predict"):
        try:
            input_array = np.array(inputs).reshape(1, -1)
            scaled_input = scaler.transform(input_array)

            if model_choice == "ANN (LSTM)":
                lstm_model = load_model("lstm_model.h5")
                lstm_input = scaled_input.reshape((1, 1, scaled_input.shape[1]))
                prob = lstm_model.predict(lstm_input)[0][0]

            elif model_choice == "Logistic Regression":
                lr = joblib.load("lr_model.pkl")
                prob = lr.predict_proba(scaled_input)[0][1]

            elif model_choice == "Decision Tree":
                dt = joblib.load("decision_tree_model.pkl")
                prob = dt.predict_proba(scaled_input)[0][1]

            elif model_choice == "Random Forest":
                rf = joblib.load("random_forest_model.pkl")
                prob = rf.predict_proba(scaled_input)[0][1]

            elif model_choice == "Naive Bayes":
                nb = joblib.load("gaussian_nb_model.pkl")
                prob = nb.predict_proba(scaled_input)[0][1]

            elif model_choice == "SVM":
                svm = joblib.load("svm_model.pkl")
                prob = svm.decision_function(scaled_input)
                prob = 1 / (1 + np.exp(-prob))  # sigmoid approximation

            elif model_choice == "XGBoost":
                xgb_model = xgb.Booster()
                xgb_model.load_model("xgb_model.json")
                dmatrix = xgb.DMatrix(scaled_input)
                prob = xgb_model.predict(dmatrix)[0]

            prediction = "🌧️ Rain Expected" if prob > 0.5 else "☀️ No Rain"
            st.success(f"Prediction: {prediction} (Probability: {prob:.2f})")

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
