import os
import json
import numpy as np
from flask import *
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import time
import mysql.connector

# Flask App Initialization
app = Flask(__name__)
app.secret_key = "login-page"

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="akshita@2002",
    database="plant_disease_db"
)
cursor = db.cursor()
# Load Model & Labels
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "plant_disease_best_model.keras")
LABELS_PATH = os.path.join(MODEL_DIR, "class_labels.json")

model = load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    class_labels = json.load(f)
    class_labels = {v: k for k, v in class_labels.items()}  # Reverse mapping

# Ensure Upload Directory Exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Function to Format Disease Name
def format_disease_name(name):
    """Format disease name for better display."""
    return name.replace("_", " ").replace("__", " ").title()

# Function to Predict Disease
def predict_disease(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Update to 224x224
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    disease_name = class_labels.get(class_index, "Unknown Disease")
    confidence = round(float(predictions[0][class_index]) * 100, 2)

    return disease_name, confidence

# Updated Cure Suggestions
cure_suggestions = {
    # Tomato Diseases
    "Tomato_Bacterial_Spot": "Use copper-based fungicides and remove infected leaves.",
    "Tomato_Early_Blight": "Apply fungicides like chlorothalonil and remove infected plants.",
    "Tomato_Late_Blight": "Destroy infected plants and use fungicides like mancozeb.",
    "Tomato_Leaf_Mold": "Improve air circulation and use sulfur-based fungicides.",
    "Tomato_Septoria_Leaf_Spot": "Remove infected leaves, avoid overhead watering, and apply fungicides.",
    "Tomato_Healthy": "The plant appears healthy. No action needed.",

    # Pepper Diseases
    "Pepper__bell___Bacterial_spot": "Use copper-based fungicides and remove infected leaves to prevent spreading.",
    "Pepper__bell___healthy": "Your pepper plant is healthy! No treatment needed.",

    # Potato Diseases
    "Potato___Early_blight": "Use fungicides like chlorothalonil and ensure proper crop rotation.",
    "Potato___Late_Blight": "Destroy infected plants, use fungicides, and avoid waterlogged soil.",
    "Potato___Healthy": "The plant is in good condition. Maintain proper watering and sunlight exposure."
}

# Flask Routes
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if "username" not in session:
        return redirect("/login")

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            disease, confidence = predict_disease(filepath)
            cure = cure_suggestions.get(disease, "Consult an expert.")

            return render_template("result.html", 
                                   disease=disease, 
                                   confidence=confidence, 
                                   cure=cure, 
                                   image_url=filepath)

    return render_template("index.html")
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]  # Plain text password

        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        
        if user:
            return "Email already exists. Try logging in."

        cursor.execute("INSERT INTO users (username, email, password) VALUES (%s, %s, %s)", 
                       (username, email, password))
        db.commit()
        return redirect("/login")

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        cursor.execute("SELECT * FROM users WHERE email = %s AND password = %s", (email, password))
        user = cursor.fetchone()

        if user:
            session["username"] = user[1]  # Store username in session
            return redirect("/")
        else:
            return "Invalid credentials. Try again."

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect("/login")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Save File with Unique Name
    filename = f"{int(time.time())}_{file.filename}"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(file_path)

    # Predict Disease
    disease_name, confidence = predict_disease(file_path)

    # Find Cure Suggestion
    cure_suggestion = cure_suggestions.get(disease_name, "No cure suggestion available. Please consult an expert.")

    # ✅ Pass `image_url` to the template
    return render_template(
        "result.html",
        disease=disease_name,
        confidence=confidence,
        cure=cure_suggestion,
        image_url=f"uploads/{filename}"  # Correct path for static folder
    )

