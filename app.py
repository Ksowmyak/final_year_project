from flask import Flask, render_template, request, redirect, session, flash, url_for
import json
import bcrypt
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Change this for production

# Load trained LSTM model and scaler
lstm_model = load_model("lstm_model.h5")
scaler = joblib.load("scaler.pkl")  # Load pre-trained scaler

# Define feature names
feature_names = ["Location", "MinTemp", "MaxTemp", "Rainfall", "Evaporation", "Sunshine",
                 "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "Humidity9am", "Humidity3pm",
                 "Pressure9am", "Pressure3pm", "Cloud9am", "Cloud3pm", "Temp9am", "Temp3pm", "RISK_MM"]

# File to store user data
USER_FILE = "users.json"

# Load or create user data
try:
    with open(USER_FILE, "r") as file:
        users = json.load(file)
except (FileNotFoundError, json.JSONDecodeError):
    users = {}

# Save user data
def save_users():
    with open(USER_FILE, "w") as file:
        json.dump(users, file, indent=4)

# Route for home (redirects to login)
@app.route("/")
def home():
    if "user" in session:
        return redirect(url_for("predict"))
    return redirect(url_for("login"))

# Login Page
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"].encode("utf-8")

        if username in users and bcrypt.checkpw(password, users[username]["password"].encode("utf-8")):
            session["user"] = username
            return redirect(url_for("predict"))
        else:
            flash("Invalid username or password!", "danger")

    return render_template("login.html")

# Register Page
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"].encode("utf-8")

        if username in users:
            flash("Username already exists!", "danger")
        else:
            hashed_pw = bcrypt.hashpw(password, bcrypt.gensalt()).decode("utf-8")
            users[username] = {"password": hashed_pw}
            save_users()
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for("login"))

    return render_template("register.html")

# Logout
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# Prediction Page
@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    prediction = None
    if request.method == "POST":
        try:
            user_input = [float(request.form[feature]) for feature in feature_names]

            user_array = np.array(user_input).reshape(1, -1)
            user_scaled = scaler.transform(user_array)
            user_scaled_reshaped = user_scaled.reshape((1, 1, len(feature_names)))

            prediction_prob = lstm_model.predict(user_scaled_reshaped)[0][0]
            prediction = "Rain Expected" if prediction_prob > 0.5 else "No Rain"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template("predict.html", prediction=prediction, username=session["user"])

if __name__ == "__main__":
    app.run(debug=True)
