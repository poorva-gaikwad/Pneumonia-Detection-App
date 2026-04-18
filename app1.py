# Standard library
import base64
import json
import logging
from torchvision import transforms
from PIL import Image
import os
import pickle
import re
import smtplib
import tempfile
import uuid
import imghdr
from openai import OpenAI
from collections import Counter
from datetime import datetime
from email.message import EmailMessage
from functools import wraps
from io import BytesIO
from PIL import Image, ImageStat, ImageOps, ImageFilter
import numpy as np
import logging
# SQLAlchemy
from sqlalchemy import or_
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import joinedload

# Logging
import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
from flask_login import LoginManager

# Third-party libraries
import cv2
import google.generativeai as genai
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from PIL import Image, ImageStat
from dotenv import load_dotenv
from keras.models import Model, load_model
from keras.preprocessing import image as keras_image
from torchvision import transforms
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

# Flask core
from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    make_response,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    session,
    url_for,
)

# Flask extensions
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    get_jwt_identity,
    jwt_required,
)
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_talisman import Talisman

# Google API
from google.api_core.exceptions import ResourceExhausted

# Paths & Config
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
ASTHMA_MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")  # or .h5

# Ensure required folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

# Environment variables
load_dotenv()

# Flask App Setup
app = Flask(__name__)
app.secret_key = os.getenv("APP_SECRET")

app.config.update({
    "SQLALCHEMY_DATABASE_URI": os.getenv("DATABASE_URL"),  # Postgres
    "SQLALCHEMY_TRACK_MODIFICATIONS": False,
    "UPLOAD_FOLDER": UPLOAD_FOLDER,
    "MAX_CONTENT_LENGTH": 32 * 1024 * 1024,  # 32MB
})

# Security & Authentication
app.config["RECAPTCHA_SITE_KEY"] = os.getenv("RECAPTCHA_SITE_KEY")
app.config["RECAPTCHA_SECRET_KEY"] = os.getenv("RECAPTCHA_SECRET_KEY")
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")
app.config["JWT_TOKEN_LOCATION"] = ["headers"]  # add "cookies" if needed
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = 3600  # 1 hour

jwt = JWTManager(app)


# Logging Setup
if not app.debug:
    handler = ConcurrentRotatingFileHandler(
        "app.log", maxBytes=5*1024*1024, backupCount=5
    )
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]"
    )
    handler.setFormatter(formatter)
    app.logger.addHandler(handler)

# Console logging (basic config)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Login Manager Setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login."""
    try:
        return db.session.get(User, int(user_id))
    except Exception as e:
        logger.error(f"Error loading user {user_id}: {e}")
        return None

# openai Model Setup
# Initialize Hugging Face client once
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

db = SQLAlchemy(app)
migrate = Migrate(app, db)

class PatientHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    # Link to the User who owns this record
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)

    # Relationship back to User
    user = db.relationship("User", backref=db.backref("histories", lazy=True))

    # Patient linkage (not unique, so multiple records allowed)
    username = db.Column(db.String(100), nullable=True)
    patient_id = db.Column(db.String(50), nullable=False)

    # Timestamp (no timezone)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Patient details
    patient_name = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    weight = db.Column(db.Float, nullable=True)
    height = db.Column(db.Float, nullable=True)
    blood_pressure = db.Column(db.String(20), nullable=True)
    allergies = db.Column(db.String(200), nullable=True)
    body_temperature = db.Column(db.Float, nullable=True)
    symptom_start = db.Column(db.Date, nullable=True)
    symptoms = db.Column(JSONB, nullable=True)
    risks = db.Column(JSONB, nullable=True)

    # Prediction outputs
    image_filename = db.Column(db.String(200), nullable=True)
    heatcam_filename = db.Column(db.String(200), nullable=True)
    prediction_label = db.Column(db.String(50), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    presence_percentage = db.Column(db.Float, nullable=True)

    def __repr__(self):
        return f"<PatientHistory {self.id}: {self.patient_name}>"

    # Helper to format timestamp (time only)
    def formatted_time(self):
        return self.timestamp.strftime("%I:%M %p")  # e.g. "10:30 AM"

    # Helper to format full date + time
    def formatted_datetime(self):
        return self.timestamp.strftime(
            "%d-%b-%Y %I:%M %p"
        )  # e.g. "13-Dec-2025 10:30 AM"



class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default="patient")  # patient, doctor, admin
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)

    def __repr__(self):
        return f"<User {self.username}>"


class SystemSettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    scan_timeout = db.Column(db.String(10), nullable=True)
    default_role = db.Column(db.String(20), nullable=True)
    notifications = db.Column(db.String(10), nullable=True)
    max_upload_size = db.Column(db.String(10), nullable=True)
    admin_email = db.Column(db.String(120), nullable=True)
    maintenance_mode = db.Column(db.String(10), nullable=True)

    def __repr__(self):
        return f"<SystemSettings {self.id}>"

class AsthmaRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_history_id = db.Column(db.Integer, db.ForeignKey("patient_history.id"), nullable=False)
    patient_history = db.relationship("PatientHistory", backref=db.backref("asthma_records", lazy=True))

    prediction_label = db.Column(db.String(50), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    pef = db.Column(db.Float, nullable=True)
    feno = db.Column(db.Float, nullable=True)
    er_visits = db.Column(db.Integer, nullable=True)


class TBRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_history_id = db.Column(db.Integer, db.ForeignKey("patient_history.id"), nullable=False)
    patient_history = db.relationship("PatientHistory", backref=db.backref("tb_records", lazy=True))

    image_filename = db.Column(db.String(200), nullable=True)
    prediction_label = db.Column(db.String(50), nullable=True)
    confidence = db.Column(db.Float, nullable=True)


class LungCancerRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patient_history_id = db.Column(db.Integer, db.ForeignKey("patient_history.id"), nullable=False)
    patient_history = db.relationship("PatientHistory", backref=db.backref("lung_cancer_records", lazy=True))

    image_filename = db.Column(db.String(200), nullable=True)
    prediction_label = db.Column(db.String(50), nullable=True)
    confidence = db.Column(db.Float, nullable=True)
    stage = db.Column(db.String(20), nullable=True)  # TNM staging

class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.String(255), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    is_read = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f"<Notification {self.id}: {self.message[:30]}>"

    def formatted_time(self):
        return self.timestamp.strftime("%I:%M %p")

    def formatted_datetime(self):
        return self.timestamp.strftime("%d-%b-%Y %I:%M %p")


with app.app_context():
    db.create_all()
csp = {
    "default-src": ["'self'"],
    "script-src": [
        "'self'",
        "https://www.google.com/recaptcha/api.js",
        "https://www.gstatic.com/recaptcha/",
        "https://cdn.jsdelivr.net",
        "'unsafe-inline'",
    ],
    "frame-src": [
        "'self'",
        "https://www.google.com/recaptcha/",
        "https://recaptcha.google.com/recaptcha/"
    ],
    "style-src": [
        "'self'",
        "'unsafe-inline'",
        "https://fonts.googleapis.com",
        "https://cdn.jsdelivr.net",
    ],
    "font-src": [
        "'self'",
        "https://fonts.gstatic.com"
    ],
    "img-src": [
        "'self'",
        "data:",
        "https://www.gstatic.com/recaptcha/"
    ],
    "connect-src": [
        "'self'",
        "https://www.google.com/recaptcha/",
        "https://www.gstatic.com/recaptcha/",
        "https://fonts.googleapis.com",
        "https://fonts.gstatic.com",
        "https://cdn.jsdelivr.net"
    ],
}

Talisman(
    app,
    content_security_policy=csp,
    force_https=True,
    strict_transport_security=True,
    strict_transport_security_preload=True,
    strict_transport_security_max_age=31536000,
    session_cookie_secure=True,
    session_cookie_samesite="Lax",
)


# Pneumonia models (TensorFlow / Keras)
try:
    model_step1 = load_model(os.path.join(BASE_DIR, "models", "pneumonia", "modelDense.h5"))
    print("✅ DenseNet model loaded successfully")
except Exception as e:
    model_step1 = None
    print(f"❌ Error loading DenseNet model: {e}")

try:
    model_step2 = load_model(os.path.join(BASE_DIR, "models", "pneumonia", "modelVGG16.h5"))
    print("✅ VGG16 model loaded successfully")
except Exception as e:
    model_step2 = None
    print(f"❌ Error loading VGG16 model: {e}")

# Pneumonia YOLO model
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "pneumonia", "pytorch_yolov8_model.pt")
yolo_model = None
try:
    if os.path.exists(YOLO_MODEL_PATH):
        yolo_model = YOLO(YOLO_MODEL_PATH)
        print(f"✅ YOLO model loaded from {YOLO_MODEL_PATH}")
    else:
        app.logger.warning(f"⚠️ YOLO model not found at {YOLO_MODEL_PATH}")
except Exception as e:
    yolo_model = None
    app.logger.error("❌ Failed to load YOLO model: %s", e)


# TB model (TensorFlow / Keras)
try:
    tb_model = load_model(os.path.join(BASE_DIR, "models", "tb", "best_model_epoch.keras"))
    print("✅ TB model loaded successfully")
except Exception as e:
    tb_model = None
    print(f"❌ Error loading TB model: {e}")

# Lung cancer model (TensorFlow / Keras)



# Model Loading - Asthma
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASTHMA_MODEL_PATH = os.path.join(BASE_DIR, "models", "asthma", "model.pkl")

asthma_model = None
try:
    if os.path.exists(ASTHMA_MODEL_PATH):
        with open(ASTHMA_MODEL_PATH, "rb") as f:
            asthma_model = pickle.load(f)
        print(f"✅ Asthma model loaded successfully from {ASTHMA_MODEL_PATH}")
    else:
        print(f"⚠️ Asthma model not found at {ASTHMA_MODEL_PATH}")
except Exception as e:
    asthma_model = None
    print(f"❌ Failed to load Asthma model: {e}")

# Helper Functions

def login_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not current_user.is_authenticated:
            flash("Please log in first.", "warning")
            return redirect(url_for("login"))
        return fn(*args, **kwargs)
    return wrapper

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def ensure_bool_text(form, prefix_list) -> str:
    return ",".join(f for f in prefix_list if form.get(f) == "on")

EMAIL_REGEX = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")

def is_valid_email(email: str) -> bool:
    if not email:
        return False
    return bool(EMAIL_REGEX.match(email.strip()))

# Security Helpers

def verify_captcha(response_token: str) -> bool:
    """Verify Google reCAPTCHA response token."""
    secret = app.config["RECAPTCHA_SECRET_KEY"]
    payload = {"secret": secret, "response": response_token}
    r = requests.post("https://www.google.com/recaptcha/api/siteverify", data=payload)
    return r.json().get("success", False)


def is_strong_password(pwd: str) -> bool:
    """Check if password meets strength requirements."""
    return (
        len(pwd) >= 8
        and re.search(r"[A-Z]", pwd)
        and re.search(r"[a-z]", pwd)
        and re.search(r"[0-9]", pwd)
        and re.search(r"[^A-Za-z0-9]", pwd)
    )


def is_admin_user():
    """Check if current user is authenticated admin."""
    try:
        return current_user.is_authenticated and current_user.role == "admin"
    except:
        return False


def get_rate_limit_key():
    """Custom key function that exempts admins."""
    if is_admin_user():
        return None  # Skip rate limiting for admins
    return get_remote_address()


def is_xray_image(image_path: str) -> bool:
    """Simple check: only basic heuristics for X-ray images."""
    try:
        # Allow only common formats
        if not image_path.lower().endswith((".jpg", ".jpeg", ".png")):
            return False

        img = Image.open(image_path).convert("L")
        stat = ImageStat.Stat(img)

        brightness = stat.mean[0]
        contrast = stat.stddev[0]

        w, h = img.size
        aspect_ratio = h / w

        # Simple thresholds
        if contrast > 3 and 10 < brightness < 245 and 0.3 < aspect_ratio < 3.0:
            return True
        return False

    except Exception:
        return False

# ======================================================
# Asthma Prediction
# ======================================================
EXPECTED_COLUMNS = [
    "Age", "BMI", "Family_History", "Air_Pollution_Level", "Physical_Activity_Level",
    "Occupation_Type", "Medication_Adherence", "Number_of_ER_Visits", "Peak_Expiratory_Flow",
    "FeNO_Level", "Gender_Female", "Gender_Male", "Gender_Other",
    "Smoking_Status_Current", "Smoking_Status_Former", "Smoking_Status_Never",
    "Allergies_Dust", "Allergies_Multiple", "Allergies_Pets", "Allergies_Pollen",
    "Comorbidities_Both", "Comorbidities_Diabetes", "Comorbidities_Hypertension",
]

def predict_asthma(form_data):
    if asthma_model is None:
        return "❌ Prediction failed: Model not available"
    try:
        def safe_float(val):
            try: return float(val)
            except: return 0.0

        def safe_int(val):
            try: return int(val)
            except: return 0

        air_pollution_map = {"Low": 0, "Moderate": 1, "High": 2}
        physical_activity_map = {"Sedentary": 0, "Moderate": 1, "Active": 2}
        occupation_type_map = {"Indoor": 0, "Outdoor": 1}

        data = pd.DataFrame([{
            "Age": safe_float(form_data.get("age")),
            "BMI": safe_float(form_data.get("bmi")),
            "Family_History": safe_int(form_data.get("family_history")),
            "Air_Pollution_Level": air_pollution_map.get(form_data.get("pollution"), 0),
            "Physical_Activity_Level": physical_activity_map.get(form_data.get("activity"), 0),
            "Occupation_Type": occupation_type_map.get(form_data.get("occupation"), 0),
            "Medication_Adherence": safe_int(form_data.get("medication")),
            "Number_of_ER_Visits": safe_int(form_data.get("er_visits")),
            "Peak_Expiratory_Flow": safe_float(form_data.get("pef")),
            "FeNO_Level": safe_float(form_data.get("feno")),
            "Gender": form_data.get("gender", "Female"),
            "Smoking_Status": form_data.get("smoking", "Never"),
            "Allergies": form_data.get("allergies", "None"),
            "Comorbidities": form_data.get("comorbidities", "None"),
        }])

        data = pd.get_dummies(data).reindex(columns=EXPECTED_COLUMNS, fill_value=0)
        raw_pred = asthma_model.predict(data)
        pred_value = float(raw_pred[0]) if hasattr(raw_pred, "__getitem__") else float(raw_pred)
        return "✅ No Asthma" if pred_value < 0.5 else "😷 Asthma Suspected"
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return f"❌ Prediction failed: {e}"


# ======================================================
# Pneumonia Classification Pipeline
# ======================================================
def predict(img_path):
    if not model_step1 or not model_step2:
        return "Model not available", 0.0, {}
    try:
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_array = np.array(img, dtype="float32") / 255.0
        img_input = np.expand_dims(img_array, axis=0)

        step1_output = model_step1.predict(img_input)[:, [0, 2]]
        step1_labels = ["Bacterial-Viral", "Normal"]

        class_idx = np.argmax(step1_output)
        step1_label = step1_labels[class_idx]
        confidence = float(step1_output[0][class_idx])

        probabilities = {label: round(float(prob) * 100, 2)
                         for label, prob in zip(step1_labels, step1_output[0])}

        if step1_label == "Bacterial-Viral":
            step2_output = model_step2.predict(img_input)
            viral_score = float(step2_output[0][0])
            step2_label = "Viral" if viral_score > 0.5 else "Bacterial"
            confidence = viral_score if step2_label == "Viral" else 1 - viral_score
            return step2_label, round(confidence * 100, 2), probabilities

        return step1_label, round(confidence * 100, 2), probabilities
    except Exception as e:
        print("Prediction error:", e)
        return "Error", 0.0, {}


# ======================================================
# YOLO Detection Pipeline
# ======================================================
def run_yolo_presence_estimate(image_path, confidence_threshold=0.01):
    if yolo_model is None:
        app.logger.warning("YOLO model not available")
        return 0.0
    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        img_area = width * height

        # Run YOLO prediction
        results = yolo_model.predict(img, conf=confidence_threshold)
        boxes = results[0].boxes
        app.logger.info("YOLO detected %d boxes", len(boxes))

        # Find pneumonia class id
        pneumonia_class_id = next((k for k, v in yolo_model.names.items()
                                   if v.lower() == "pneumonia"), None)
        if pneumonia_class_id is None:
            app.logger.warning("No 'pneumonia' class found in YOLO model.names")
            return 0.0

        # Filter pneumonia boxes above threshold
        filtered = [
            box for box in boxes
            if int(box.cls[0]) == pneumonia_class_id and float(box.conf[0]) > confidence_threshold
        ]
        if not filtered:
            app.logger.info("No pneumonia boxes above threshold %.2f", confidence_threshold)
            return 0.0

        pneumonia_area = 0.0
        for box in filtered:
            # Convert tensors to floats
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            conf = float(box.conf[0])
            area = max(0.0, (x2 - x1) * (y2 - y1))
            pneumonia_area += area * conf
            app.logger.info("Box class=pneumonia conf=%.2f area=%.2f", conf, area)

        presence_percentage = (pneumonia_area / img_area) * 100.0
        return round(presence_percentage, 2)

    except Exception as e:
        app.logger.error("YOLO presence estimate error: %s", e)
        return 0.0



# ======================================================
# TB Classification Pipeline
# ======================================================
def preprocess_xray(file, img_size=(64, 64)):
    img = Image.open(file).convert("L").resize(img_size)
    arr = np.array(img, dtype="float32") / 255.0
    arr = np.expand_dims(arr, axis=-1)
    return np.expand_dims(arr, axis=0)

def run_tb_inference(img_file):
    if not tb_model:
        return "Model not available", 0.0, {}
    try:
        xray = preprocess_xray(img_file)
        score = float(tb_model.predict(xray)[0][0])
        labels = ["Normal", "Tuberculosis"]
        class_idx = 1 if score >= 0.5 else 0
        label = labels[class_idx]
        confidence = score if label == "Tuberculosis" else 1 - score
        probabilities = {"Normal": round((1 - score) * 100, 2),
                         "Tuberculosis": round(score * 100, 2)}
        return label, round(confidence * 100, 2), probabilities
    except Exception as e:
        print("Prediction error:", e)
        return "Error", 0.0, {}


# ======================================================
# Lung Cancer Classification Pipeline
# ======================================================
# Preprocessing pipeline for InceptionV4 (expects 299x299 RGB)
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # adjust if trained differently
])



# ======================================================
# Grad-CAM Visualization
# ======================================================
def generate_gradcam(img_path):
    try:
        img = Image.open(img_path).convert("RGB").resize((224, 224))
        img_array = np.array(img, dtype="float32")
        img_array_expanded = np.expand_dims(img_array, axis=0) / 255.0

        # Step 1 prediction (classification)
        step1_prediction = model_step1.predict(img_array_expanded)
        class_idx = np.argmax(step1_prediction)

        # Choose last conv layer from step2 model
        last_conv_layer_name = "block3_conv1"  # adjust to your architecture
        try:
            last_conv_layer = model_step2.get_layer(last_conv_layer_name)
        except:
            print(f"Layer '{last_conv_layer_name}' not found in model_step2.")
            return None

        heatmap_model = Model(
            [model_step2.inputs], [last_conv_layer.output, model_step2.output]
        )

        # Grad-CAM++ logic
        with tf.GradientTape() as tape:
            conv_outputs, predictions = heatmap_model(img_array_expanded)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)

        # Derivatives for Grad-CAM++
        first_derivative = grads
        second_derivative = tf.square(grads)
        third_derivative = tf.pow(grads, 3)

        global_sum = tf.reduce_sum(conv_outputs, axis=(0, 1, 2))

        alpha_num = second_derivative
        alpha_denom = 2 * second_derivative + third_derivative * global_sum
        alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones_like(alpha_denom))

        alphas = alpha_num / alpha_denom
        weights = tf.reduce_sum(alphas * tf.nn.relu(first_derivative), axis=(0, 1, 2))

        conv_outputs = conv_outputs[0].numpy()
        for i in range(weights.shape[0]):
            conv_outputs[:, :, i] *= weights[i]

        heatmap = np.mean(conv_outputs, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-10
        heatmap_resized = cv2.resize(heatmap, (224, 224))

        # Overlay heatmap
        heatmap_resized = np.uint8(255 * heatmap_resized)
        heatmap_colormap = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        original_img = np.uint8(255 * img_array / np.max(img_array))
        superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap_colormap, 0.4, 0)

        output_filename = f"gradcampp_{uuid.uuid4().hex}.png"
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)
        cv2.imwrite(output_path, superimposed_img)
        return output_filename

    except Exception as e:
        print(f"Grad-CAM++ error for {img_path}: {e}")
        return None

# Role-Based Access Control

def admin_required(f):
    """Decorator to restrict access to admin users."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated:
            flash("Please log in to access admin features.", "warning")
            return redirect(url_for("login"))
        if current_user.role != "admin":
            return redirect(url_for("unauthorized"))
        return f(*args, **kwargs)
    return decorated_function


def role_required(required_role: str):
    """Decorator to restrict access to a specific role."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated or current_user.role != required_role:
                flash("Access denied.", "danger")
                return redirect(url_for("home"))
            return fn(*args, **kwargs)
        return wrapper
    return decorator


# Routes

@app.route("/")
def index():
    """Redirect authenticated users to home, otherwise show welcome page."""
    if current_user.is_authenticated:
        return redirect(url_for("home"))
    return render_template("welcome.html")


@app.route("/welcome")
def welcome():
    """Render welcome page."""
    return render_template("welcome.html")


# Rate Limiting
limiter = Limiter(
    key_func=get_rate_limit_key,
    app=app,
    default_limits=[
        f"{os.getenv('RATELIMIT_DEFAULT_PER_DAY', '200')} per day",
        f"{os.getenv('RATELIMIT_DEFAULT_PER_HOUR', '50')} per hour",
    ],
)
# User Registration
@limiter.limit(f"{os.getenv('RATELIMIT_REGISTER_PER_MINUTE', '3')} per minute", exempt_when=lambda: is_admin_user())
@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    if request.method == "POST":
        uname = request.form.get("username", "").strip().lower()
        email = request.form.get("email", "").strip().lower()
        pwd = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        role = request.form.get("role", "patient")

        # Validation checks
        if not uname or not email or not pwd or not confirm:
            error = "Please fill all fields."
            return render_template("register.html", error=error)

        if not is_valid_email(email):
            error = "Invalid email address."
            return render_template("register.html", error=error)

        if pwd != confirm:
            error = "Passwords do not match."
            return render_template("register.html", error=error)

        if not is_strong_password(pwd):
            error = "Password must be strong (8+ chars, upper/lowercase, number, symbol)."
            return render_template("register.html", error=error)

        if User.query.filter((User.username == uname) | (User.email == email)).first():
            error = "Username or email already exists."
            return render_template("register.html", error=error)

        # Create new user (admin, doctor, or patient — all allowed)
        user = User(
            username=uname,
            email=email,
            password_hash=generate_password_hash(pwd, method="scrypt"),
            role=role,
        )
        db.session.add(user)
        db.session.commit()

        return redirect(url_for("login"))

    return render_template("register.html", error=error)




# User Login
@limiter.limit(f"{os.getenv('RATELIMIT_LOGIN_PER_MINUTE', '5')} per minute", exempt_when=lambda: is_admin_user())
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uname = request.form.get("username", "").strip().lower()
        pwd = request.form.get("password", "")

        # Optional CAPTCHA enforcement
        if os.getenv("ENABLE_CAPTCHA", "false").lower() == "true":
            captcha_response = request.form.get("g-recaptcha-response")
            if not captcha_response or not verify_captcha(captcha_response):
                flash("CAPTCHA verification failed.", "danger")
                return redirect(url_for("login"))

        # Validate credentials
        user = User.query.filter_by(username=uname).first()
        if not user or not check_password_hash(user.password_hash, pwd):
            flash("Invalid username or password.", "danger")
            return redirect(url_for("login"))

        # Login user
        login_user(user)
        flash(f"Welcome back, {user.username}!", "success")
        print("Logged in role:", user.role)
        return redirect(url_for("home"))

    return render_template("login.html")


# API Login (JWT)
@limiter.limit("10 per minute")
@app.route("/api/login", methods=["POST"])
def api_login():
    uname = request.json.get("username", "").strip().lower()
    pwd = request.json.get("password", "")

    user = User.query.filter_by(username=uname).first()
    if not user or not check_password_hash(user.password_hash, pwd):
        return {"message": "Invalid credentials"}, 401

    token = create_access_token(
        identity={"id": user.id, "role": user.role, "username": user.username}
    )
    return {"access_token": token}, 200


# API - Current User
@app.route("/api/me", methods=["GET"])
@jwt_required()
def me():
    identity = get_jwt_identity()
    return {"user": identity}, 200


# Rate Limit Error Handler
@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify(error="Too many requests. Please try again shortly."), 429

@app.route("/home")
@login_required
def home():
    role = current_user.role
    app.logger.info(f"{current_user.username} accessed home as {role}")

    if role == "doctor":
        return render_template("doctor_dashboard.html")
    elif role == "patient":
        return render_template("patient_dashboard.html")
    elif role == "admin":
        return redirect(url_for("admin_dashboard"))
    else:
        return redirect(url_for("logout"))

#dashboard
@app.route("/doctor_dashboard")
@login_required
def doctor_dashboard():
    if current_user.role != "doctor":
        abort(403)
    return render_template("doctor_dashboard.html")


@app.route("/patient_dashboard")
@login_required
def patient_dashboard():
    if current_user.role != "patient":
        abort(403)
    return render_template("patient_dashboard.html")

@app.route("/admin_dashboard")
@login_required
def admin_dashboard():
    if current_user.role != "admin":
        return redirect(url_for("unauthorized"))
    return render_template("admin_dashboard.html")


@app.route("/admin_user")
@login_required
def admin_user():
    if current_user.role != "admin":
        return redirect(url_for("unauthorized"))
    users = User.query.all()
    return render_template("admin_user.html", users=users)


@app.route("/change_role/<int:user_id>", methods=["POST"])
@login_required
def change_role(user_id):
    if current_user.role != "admin":
        return redirect(url_for("unauthorized"))
    new_role = request.form.get("new_role")
    user = User.query.get_or_404(user_id)
    user.role = new_role
    db.session.commit()
    flash(f"Role updated for {user.username}", "success")
    return redirect(url_for("admin_user"))


@app.route("/toggle_user_status/<int:user_id>", methods=["POST"])
@login_required
def toggle_user_status(user_id):
    if current_user.role != "admin":
        return redirect(url_for("unauthorized"))
    user = User.query.get_or_404(user_id)
    user.is_active = not user.is_active
    db.session.commit()
    flash(f"User {user.username} status updated.", "success")
    return redirect(url_for("admin_user"))


@app.route("/delete_user/<int:user_id>", methods=["POST"])
@login_required
def delete_user(user_id):
    if current_user.role != "admin":
        return redirect(url_for("unauthorized"))
    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    flash(f"User {user.username} deleted.", "success")
    return redirect(url_for("admin_user"))


@app.route("/settings", methods=["GET", "POST"])
@login_required
def settings():
    if current_user.role != "admin":
        return redirect(url_for("unauthorized"))

    settings = SystemSettings.query.first()

    if request.method == "POST":
        if not settings:
            settings = SystemSettings()

        # Convert to proper types
        settings.scan_timeout = int(request.form.get("scan_timeout", 0))
        settings.default_role = request.form.get("default_role")
        settings.notifications = request.form.get("notifications") == "true"
        settings.max_upload_size = int(request.form.get("max_upload_size", 0))
        settings.admin_email = request.form.get("admin_email")
        settings.maintenance_mode = request.form.get("maintenance_mode") == "true"

        db.session.add(settings)
        db.session.commit()
        flash("Settings updated successfully", "success")
        return redirect(url_for("settings"))

    return render_template("settings.html", settings=settings)


@app.route("/unauthorized")
def unauthorized():
    flash("You do not have permission to access that page.", "danger")
    return redirect(url_for("login"))


# Human-readable label maps
SYMPTOM_LABELS = {
    "symptoms_fever": "Fever",
    "symptoms_chills": "Chills",
    "symptoms_cough": "Cough",
    "symptoms_shortness_of_breath": "Shortness of Breath",
    "symptoms_chest_pain": "Chest Pain",
    "symptoms_diarrhea": "Diarrhea",
    "symptoms_vomiting_nausea": "Vomiting/Nausea",
}

RISK_LABELS = {
    "risk_chronic_heart_disease": "Chronic Heart Disease",
    "risk_chronic_liver_disease": "Chronic Liver Disease",
    "risk_chronic_lung_disease": "Chronic Lung Disease",
    "risk_diabetes": "Diabetes",
}
@app.route("/patient_history", methods=["GET", "POST"])
@login_required
def patient_history():
    if current_user.role not in ["doctor", "patient"]:
        abort(403)

    if request.method == "POST":
        # Collect raw checkbox values by scanning form keys
        raw_symptoms = [key for key in request.form.keys() if key in SYMPTOM_LABELS]
        raw_risks = [key for key in request.form.keys() if key in RISK_LABELS]

        # Map to human-readable labels
        symptoms = [SYMPTOM_LABELS[s] for s in raw_symptoms]
        risks = [RISK_LABELS[r] for r in raw_risks]

        # Optional heatmap upload
        heatcam = request.files.get("heatcam_image")
        heatcam_filename = None
        if heatcam and allowed_file(heatcam.filename):
            heatcam_filename = f"{uuid.uuid4().hex}_{secure_filename(heatcam.filename)}"
            heatcam.save(os.path.join(app.config["UPLOAD_FOLDER"], heatcam_filename))

        # Create new patient history entry
        entry = PatientHistory(
            user_id=current_user.id,
            patient_id=str(current_user.id),   # still link to user
            timestamp=datetime.utcnow(),
            patient_name=request.form.get("patient_name", "").strip(),
            age=request.form.get("age", "").strip(),
            gender=request.form.get("gender", "").strip(),
            weight=request.form.get("weight", "").strip(),
            height=request.form.get("height", "").strip(),
            blood_pressure=request.form.get("blood_pressure", "").strip(),
            allergies=request.form.get("allergies", "").strip(),
            body_temperature=request.form.get("body_temperature", "").strip(),
            symptom_start=request.form.get("symptom_start", "").strip(),
            symptoms=symptoms,
            risks=risks,
            heatcam_filename=heatcam_filename,
        )

        try:
            db.session.add(entry)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            app.logger.error("DB commit failed: %s", e)
            flash("Failed to save record. Try again.", "danger")
            return redirect(url_for("patient_history"))

        # ✅ Store the PatientHistory record ID in session
        session["current_patient_id"] = entry.id

        flash("✅ Patient history saved. Now upload scan.", "success")
        return redirect(url_for("scan_predict"))

    return render_template("patient_history.html")


@app.route("/scan_predict", methods=["GET", "POST"])
@login_required
def scan_predict():
    if current_user.role not in ["doctor", "patient"]:
        abort(403)

    result = confidence = heatmap_filename = None
    presence_percentage = None
    probabilities = {}

    pid = session.get("current_patient_id")
    if not pid:
        flash("Please complete the patient form first.", "warning")
        return redirect(url_for("patient_history"))

    # ✅ Fetch by PatientHistory primary key
    patient = db.session.get(PatientHistory, pid)
    if not patient:
        flash("Patient record not found. Please fill the form again.", "danger")
        return redirect(url_for("patient_history"))

    if request.method == "POST":
        f = request.files.get("xray")
        if not f or f.filename == "" or not allowed_file(f.filename):
            flash("Please upload a valid PNG or JPG X-ray.", "danger")
            return redirect(url_for("scan_predict"))

        fname = secure_filename(f.filename)
        unique_name = f"{uuid.uuid4().hex}_{fname}"
        path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
        f.save(path)

        # Validate image before prediction
        if not is_xray_image(path):
            os.remove(path)
            flash("Invalid image. Please upload a valid chest X-ray.", "danger")
            return redirect(url_for("scan_predict"))

        try:
            # Step 1: Run CNN prediction
            result, confidence, probabilities = predict(path)

            # Step 2: Generate Grad-CAM heatmap
            heatmap_filename = generate_gradcam(path)

            # Step 3: YOLO if pneumonia
            if result.lower() in ["bacterial", "viral"]:
                presence_percentage = run_yolo_presence_estimate(path)
                app.logger.info("YOLO presence estimate: %.2f%%", presence_percentage)
            else:
                presence_percentage = 0.0
                app.logger.info("YOLO skipped because CNN result = %s", result)

            # Step 4: Update patient history
            patient.image_filename = unique_name
            patient.prediction_label = result
            patient.confidence = confidence
            patient.heatcam_filename = heatmap_filename
            patient.presence_percentage = presence_percentage

            db.session.commit()
            session["current_patient_id"] = patient.id
            flash(f"Prediction: {result} ({confidence:.2f}% confidence)", "info")

        except Exception as e:
            flash(f"Prediction failed: {str(e)}", "danger")
            return redirect(url_for("scan_predict"))

    return render_template(
        "scan_predict.html",
        result=result,
        confidence=confidence,
        patient_id=pid,
        heatmap=heatmap_filename,
        probabilities=probabilities,
        presence_percentage=presence_percentage,
    )

# Asthma Prediction Route
@app.route("/predict/asthma", methods=["GET", "POST"], endpoint="predict_asthma")
def predict_asthma_route():
    if request.method == "POST":
        form_data = {key: request.form.get(key, "") for key in request.form}
        result = predict_asthma(form_data)

        # Save to DB
        visit = PatientHistory.query.filter_by(user_id=current_user.id).order_by(PatientHistory.timestamp.desc()).first()
        if visit:
            record = AsthmaRecord(
                patient_history_id=visit.id,
                prediction_label=result,
                confidence=None,  # you can pass actual confidence if available
                pef=form_data.get("pef"),
                feno=form_data.get("feno"),
                er_visits=form_data.get("er_visits")
            )
            db.session.add(record)
            db.session.commit()

        return render_template("predict_asthma.html", result=result, data=form_data)

    return render_template("predict_asthma.html")

# TB Prediction Route
@app.route("/predict/tb", methods=["GET", "POST"], endpoint="predict_tb")
@login_required
def predict_tb():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("predict_tb.html", error="No file uploaded")

        f = request.files["file"]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f.filename))
        f.save(filepath)

        # ✅ Check if the uploaded file resembles an X-ray

    # ✅ Check before inference
        if not is_xray_image(filepath):
            return render_template(
            "predict_tb.html",
            error="Uploaded image does not resemble a chest X-ray. Please upload a valid X-ray."
        )


        # Run TB inference only if image passes validation
        label, confidence, probs = run_tb_inference(filepath)

        # Save to DB
        visit = PatientHistory.query.filter_by(user_id=current_user.id).order_by(PatientHistory.timestamp.desc()).first()
        if visit:
            record = TBRecord(
                patient_history_id=visit.id,
                image_filename=f.filename,
                prediction_label=label,
                confidence=confidence
            )
            db.session.add(record)
            db.session.commit()

        return render_template(
            "predict_tb.html",
            filename=f.filename,
            result=label,
            confidence=confidence,
            probabilities=probs,
        )

    return render_template("predict_tb.html")

@app.route("/history", methods=["GET"])
@login_required
def history():
    query = request.args.get("q", "").strip()
    role = current_user.role

    # 🚫 Block admins completely
    if role == "admin":
        abort(403)

    # Patients can only view their own records
    elif role == "patient":
        base_query = PatientHistory.query.filter_by(user_id=current_user.id)

    # Doctors can only view records they created (linked by user_id)
    elif role == "doctor":
        base_query = PatientHistory.query.filter_by(user_id=current_user.id)

    else:
        abort(403)

    base_query = base_query.options(
        joinedload(PatientHistory.asthma_records),
        joinedload(PatientHistory.tb_records)
    )

    if query:
        search_filter = (
            PatientHistory.patient_name.ilike(f"%{query}%")
            | PatientHistory.prediction_label.ilike(f"%{query}%")
        )
        rows = (
            base_query.filter(search_filter)
            .order_by(PatientHistory.timestamp.desc())
            .all()
        )
    else:
        rows = base_query.order_by(PatientHistory.timestamp.desc()).all()

    # Select template based on role
    if role == "patient":
        template_name = "patient/history.html"
    elif role == "doctor":
        template_name = "doctor/history.html"
    else:
        abort(403)

    return render_template(template_name, rows=rows)



@app.route("/report/<int:scan_id>")
@login_required
def view_report(scan_id):
    # Fetch scan record
    scan = PatientHistory.query.get_or_404(scan_id)

    # Role-based access control
    role = current_user.role
    if role == "patient" and scan.user_id != current_user.id:
        abort(403)
    elif role == "doctor" and scan.user_id != current_user.id:
        abort(403)
    elif role == "admin":
        abort(403)

    # No mapping here — pass raw scan.symptoms and scan.risks
    return render_template("report.html", scan=scan)


@app.route("/delete_history/<int:history_id>", methods=["POST"])
@login_required
def delete_history(history_id):
    history = PatientHistory.query.get_or_404(history_id)

    # Authorization: patients and doctors may only delete their own records
    if current_user.role == "patient" and history.user_id != current_user.id:
        abort(403)
    if current_user.role == "doctor" and history.user_id != current_user.id:
        abort(403)
    if current_user.role == "admin":
        abort(403)

    db.session.delete(history)
    db.session.commit()
    flash("History record deleted successfully.", "success")
    return redirect(url_for("history"))


@app.route("/analytics")
@login_required
def analytics():
    role = current_user.role

    try:
        # 1. Filter scans based on role
        if role == "admin":
            scans = PatientHistory.query.filter(PatientHistory.image_filename.isnot(None)).all()
            asthma_records = AsthmaRecord.query.all()
            tb_records = TBRecord.query.all()
        elif role == "doctor":
            scans = (
                PatientHistory.query.filter_by(user_id=current_user.id)
                .filter(PatientHistory.image_filename.isnot(None))
                .all()
            )
            asthma_records = AsthmaRecord.query.join(PatientHistory).filter(
                PatientHistory.user_id == current_user.id
            ).all()
            tb_records = TBRecord.query.join(PatientHistory).filter(
                PatientHistory.user_id == current_user.id
            ).all()
        else:
            abort(403)

        total_scans = len(scans)

        # 2. Pneumonia Prediction Distribution
        prediction_counts = Counter(r.prediction_label for r in scans if r.prediction_label)
        prediction_labels, prediction_values = list(prediction_counts.keys()), list(prediction_counts.values())

        # 3. Symptom Frequency (list-based)
        symptom_counts = Counter()
        for row in scans:
            if row.symptoms and isinstance(row.symptoms, list):
                for s in row.symptoms:
                    symptom_counts[s] += 1
        symptom_labels, symptom_values = list(symptom_counts.keys()), list(symptom_counts.values())

        # 4. Risk Factor Frequency (list-based)
        risk_counts = Counter()
        for row in scans:
            if row.risks and isinstance(row.risks, list):
                for r in row.risks:
                    risk_counts[r] += 1
        risk_labels, risk_values = list(risk_counts.keys()), list(risk_counts.values())

        # 5. Age Groups
        age_groups = {"0-18": 0, "19-35": 0, "36-60": 0, "60+": 0}
        for row in scans:
            try:
                age = int(row.age)
                if age <= 18: age_groups["0-18"] += 1
                elif age <= 35: age_groups["19-35"] += 1
                elif age <= 60: age_groups["36-60"] += 1
                else: age_groups["60+"] += 1
            except (ValueError, TypeError):
                continue
        age_labels, age_values = list(age_groups.keys()), list(age_groups.values())

        # 6. Confidence Trend (last 10 pneumonia scans)
        sorted_scans = sorted(scans, key=lambda r: r.timestamp, reverse=True)
        recent_pneumonia = [r for r in sorted_scans if r.confidence is not None][:10]
        confidence_scores = [round(r.confidence, 2) for r in reversed(recent_pneumonia)]
        confidence_timestamps = [r.timestamp.strftime("%b %d") for r in reversed(recent_pneumonia)]

        # 7. Summary Stats
        conf_scores = [r.confidence for r in scans if r.confidence is not None]
        user_conf_avg = round(sum(conf_scores) / len(conf_scores), 2) if conf_scores else 0
        top_pred = Counter(r.prediction_label for r in scans if r.prediction_label)
        user_top_prediction = top_pred.most_common(1)[0][0] if top_pred else "N/A"

        # 8. Asthma Analytics
        asthma_counts = Counter(r.prediction_label for r in asthma_records if r.prediction_label)
        asthma_labels, asthma_values = list(asthma_counts.keys()), list(asthma_counts.values())
        asthma_total = len(asthma_records)

        # 9. TB Analytics
        tb_counts = Counter(r.prediction_label for r in tb_records if r.prediction_label)
        tb_labels, tb_values = list(tb_counts.keys()), list(tb_counts.values())
        tb_total = len(tb_records)

        # 10. Merge recent history
        recent_all = []
        for r in recent_pneumonia:
            recent_all.append({
                "timestamp": r.timestamp,
                "patient_name": r.patient_name,
                "prediction_label": r.prediction_label,
                "confidence": r.confidence
            })
        for a in asthma_records[-10:]:
            recent_all.append({
                "timestamp": a.patient_history.timestamp,
                "patient_name": a.patient_history.patient_name,
                "prediction_label": f"Asthma: {a.prediction_label}",
                "confidence": a.confidence or "N/A"
            })
        for t in tb_records[-10:]:
            recent_all.append({
                "timestamp": t.patient_history.timestamp,
                "patient_name": t.patient_history.patient_name,
                "prediction_label": f"TB: {t.prediction_label}",
                "confidence": t.confidence or "N/A"
            })
        recent_all = sorted(recent_all, key=lambda x: x["timestamp"], reverse=True)[:10]

        return render_template(
            "analytics.html",
            prediction_labels=prediction_labels, prediction_values=prediction_values,
            symptom_labels=symptom_labels, symptom_values=symptom_values,
            age_labels=age_labels, age_values=age_values,
            risk_labels=risk_labels, risk_values=risk_values,
            recent_scans=recent_all, total_scans=total_scans,
            confidence_scores=confidence_scores, confidence_timestamps=confidence_timestamps,
            user_total=len(scans), user_conf_avg=user_conf_avg, user_top_prediction=user_top_prediction,
            asthma_labels=asthma_labels, asthma_values=asthma_values, asthma_total=asthma_total,
            tb_labels=tb_labels, tb_values=tb_values, tb_total=tb_total,
        )

    except Exception as e:
        app.logger.error("Analytics error: %s", e)
        flash("Analytics failed to load. Please try again later.", "danger")
        return redirect(url_for("welcome"))

@app.route("/about", strict_slashes=False)
@login_required
def about():
    return render_template("about.html",role=current_user.role)

@app.route("/prevention")
@login_required
def prevention():
    if current_user.role != "patient":
        abort(403)
    return render_template("prevention.html")

@app.route("/uploads/<filename>")
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/service-worker.js")
def serve_worker():
    return send_from_directory(
        "static", "service-worker.js", mimetype="application/javascript"
    )



@app.route("/assistant", methods=["GET", "POST"])
def assistant():
    response = None
    if request.method == "POST":
        query = request.form.get("query")

        try:
            completion = client.chat.completions.create(
                model="moonshotai/Kimi-K2-Instruct-0905",
                messages=[{"role": "user", "content": query}],
                max_tokens=150
            )
            response = completion.choices[0].message.content
        except Exception as e:
            response = f"❌ Error: {e}"

    return render_template("assistant.html", response=response)


@app.route("/logout")
@login_required
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("welcome"))
@app.errorhandler(403)
def forbidden(e):
    return render_template("403.html"), 403

if __name__ == "__main__":
    app.run(debug=True)
