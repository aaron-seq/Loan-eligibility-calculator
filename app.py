"""
Enhanced Loan Eligibility Prediction System
Modern Flask application with improved architecture and best practices
"""

import os
import logging
import pickle
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, session, jsonify, flash
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from contextlib import contextmanager
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LoanEligibilityApp:
    """Main application class for loan eligibility prediction system"""

    def __init__(self):
        self.app = Flask(__name__)
        self.setup_app()
        self.setup_database()
        self.load_ml_models()
        self.register_routes()

    def setup_app(self):
        """Configure Flask application settings"""
        self.app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))
        self.app.config["DATABASE_PATH"] = os.environ.get(
            "DATABASE_PATH", "loan_prediction.db"
        )
        self.app.config["MODEL_PATH"] = "models"

    def setup_database(self):
        """Initialize SQLite database for deployment compatibility"""
        with self.get_database_connection() as conn:
            cursor = conn.cursor()

            # Create users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    mobile_number TEXT,
                    full_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            """)

            # Create admins table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS admins (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    admin_id TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create predictions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS loan_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    first_name TEXT NOT NULL,
                    last_name TEXT NOT NULL,
                    gender INTEGER,
                    marital_status INTEGER,
                    dependents_count INTEGER,
                    education_level INTEGER,
                    employment_type INTEGER,
                    property_location INTEGER,
                    credit_history REAL,
                    cibil_score INTEGER,
                    applicant_income REAL,
                    coapplicant_income REAL,
                    loan_amount REAL,
                    loan_tenure REAL,
                    prediction_result TEXT,
                    confidence_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)

            # Create default admin if not exists
            cursor.execute("SELECT COUNT(*) FROM admins WHERE admin_id = ?", ("admin",))
            if cursor.fetchone()[0] == 0:
                admin_password_hash = generate_password_hash("admin123")
                cursor.execute(
                    "INSERT INTO admins (admin_id, password_hash) VALUES (?, ?)",
                    ("admin", admin_password_hash),
                )

            conn.commit()
            logger.info("Database initialized successfully")

    @contextmanager
    def get_database_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.app.config["DATABASE_PATH"])
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def load_ml_models(self):
        """Load machine learning models with error handling"""
        try:
            model_path = os.path.join(
                self.app.config["MODEL_PATH"], "loan_eligibility_model.pkl"
            )
            scaler_path = os.path.join(
                self.app.config["MODEL_PATH"], "feature_scaler.pkl"
            )

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                with open(model_path, "rb") as f:
                    self.prediction_model = pickle.load(f)
                with open(scaler_path, "rb") as f:
                    self.feature_scaler = pickle.load(f)
                logger.info("ML models loaded successfully")
            else:
                logger.warning(
                    "ML model files not found. Prediction service will be unavailable."
                )
                self.prediction_model = None
                self.feature_scaler = None
        except Exception as e:
            logger.error(f"Failed to load ML models: {e}")
            self.prediction_model = None
            self.feature_scaler = None

    def require_login(self, f):
        """Decorator to require user authentication"""

        @wraps(f)
        def decorated_function(*args, **kwargs):
            if "user_id" not in session:
                flash("Please log in to access this page.", "warning")
                return redirect("/")
            return f(*args, **kwargs)

        return decorated_function

    def require_admin(self, f):
        """Decorator to require admin authentication"""

        @wraps(f)
        def decorated_function(*args, **kwargs):
            if "admin_id" not in session:
                flash("Admin access required.", "error")
                return redirect("/admin-login")
            return f(*args, **kwargs)

        return decorated_function

    def register_routes(self):
        """Register all application routes"""

        @self.app.route("/")
        def landing_page():
            """Display login/signup page"""
            return render_template("auth/login.html")

        @self.app.route("/signup")
        def signup_page():
            """Display user registration page"""
            return render_template("auth/signup.html")

        @self.app.route("/api/register", methods=["POST"])
        def register_user():
            """Handle user registration"""
            try:
                form_data = request.get_json() if request.is_json else request.form

                # Validate input data
                required_fields = [
                    "user_id",
                    "email",
                    "full_name",
                    "password",
                    "confirm_password",
                ]
                for field in required_fields:
                    if not form_data.get(field):
                        return jsonify(
                            {"success": False, "message": f"{field} is required"}
                        ), 400

                # Check password confirmation
                if form_data["password"] != form_data["confirm_password"]:
                    return jsonify(
                        {"success": False, "message": "Passwords do not match"}
                    ), 400

                # Hash password
                password_hash = generate_password_hash(form_data["password"])

                # Save user to database
                with self.get_database_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO users (user_id, email, mobile_number, full_name, password_hash)
                        VALUES (?, ?, ?, ?, ?)
                    """,
                        (
                            form_data["user_id"],
                            form_data["email"],
                            form_data.get("mobile_number", ""),
                            form_data["full_name"],
                            password_hash,
                        ),
                    )
                    conn.commit()

                return jsonify({"success": True, "message": "Registration successful"})

            except sqlite3.IntegrityError:
                return jsonify(
                    {"success": False, "message": "Username or email already exists"}
                ), 409
            except Exception as e:
                logger.error(f"Registration error: {e}")
                return jsonify(
                    {"success": False, "message": "Registration failed"}
                ), 500

        @self.app.route("/api/login", methods=["POST"])
        def authenticate_user():
            """Handle user login"""
            try:
                form_data = request.get_json() if request.is_json else request.form
                user_id = form_data.get("user_id")
                password = form_data.get("password")

                if not user_id or not password:
                    return jsonify(
                        {"success": False, "message": "Username and password required"}
                    ), 400

                with self.get_database_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT user_id, password_hash, full_name FROM users WHERE user_id = ? AND is_active = 1",
                        (user_id,),
                    )
                    user = cursor.fetchone()

                if user and check_password_hash(user["password_hash"], password):
                    session["user_id"] = user["user_id"]
                    session["full_name"] = user["full_name"]
                    return jsonify({"success": True, "message": "Login successful"})
                else:
                    return jsonify(
                        {"success": False, "message": "Invalid credentials"}
                    ), 401

            except Exception as e:
                logger.error(f"Login error: {e}")
                return jsonify({"success": False, "message": "Login failed"}), 500

        @self.app.route("/dashboard")
        @self.require_login
        def user_dashboard():
            """Display user dashboard with prediction form"""
            return render_template(
                "dashboard/home.html", user_name=session.get("full_name")
            )

        @self.app.route("/api/predict", methods=["POST"])
        @self.require_login
        def predict_loan_eligibility():
            """Handle loan eligibility prediction"""
            if not self.prediction_model or not self.feature_scaler:
                return jsonify(
                    {"success": False, "message": "Prediction service unavailable"}
                ), 503

            try:
                form_data = request.get_json() if request.is_json else request.form

                # Extract and validate input features
                loan_features = self.extract_loan_features(form_data)
                if not loan_features:
                    return jsonify(
                        {"success": False, "message": "Invalid input data"}
                    ), 400

                # Generate prediction
                prediction_result = self.generate_prediction(loan_features)

                # Save prediction to database
                self.save_prediction_record(
                    session["user_id"], form_data, prediction_result
                )

                return jsonify(
                    {
                        "success": True,
                        "prediction": prediction_result["decision"],
                        "confidence": prediction_result["confidence"],
                        "message": f"Loan {prediction_result['decision'].lower()}",
                    }
                )

            except Exception as e:
                logger.error(f"Prediction error: {e}")
                return jsonify({"success": False, "message": "Prediction failed"}), 500

        @self.app.route("/admin-login")
        def admin_login_page():
            """Display admin login page"""
            return render_template("admin/login.html")

        @self.app.route("/api/admin/login", methods=["POST"])
        def authenticate_admin():
            """Handle admin authentication"""
            try:
                form_data = request.get_json() if request.is_json else request.form
                admin_id = form_data.get("admin_id")
                password = form_data.get("password")

                with self.get_database_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT admin_id, password_hash FROM admins WHERE admin_id = ?",
                        (admin_id,),
                    )
                    admin = cursor.fetchone()

                if admin and check_password_hash(admin["password_hash"], password):
                    session["admin_id"] = admin["admin_id"]
                    return jsonify(
                        {"success": True, "message": "Admin login successful"}
                    )
                else:
                    return jsonify(
                        {"success": False, "message": "Invalid admin credentials"}
                    ), 401

            except Exception as e:
                logger.error(f"Admin login error: {e}")
                return jsonify({"success": False, "message": "Admin login failed"}), 500

        @self.app.route("/admin/dashboard")
        @self.require_admin
        def admin_dashboard():
            """Display admin dashboard with all predictions"""
            try:
                with self.get_database_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT lp.*, u.full_name, u.email 
                        FROM loan_predictions lp 
                        JOIN users u ON lp.user_id = u.user_id 
                        ORDER BY lp.created_at DESC
                    """)
                    predictions = cursor.fetchall()

                return render_template("admin/dashboard.html", predictions=predictions)

            except Exception as e:
                logger.error(f"Admin dashboard error: {e}")
                flash("Failed to load dashboard", "error")
                return redirect("/admin-login")

        @self.app.route("/logout")
        def logout_user():
            """Handle user logout"""
            session.clear()
            flash("Successfully logged out", "info")
            return redirect("/")

        @self.app.route("/admin/logout")
        def logout_admin():
            """Handle admin logout"""
            session.clear()
            flash("Admin logged out successfully", "info")
            return redirect("/admin-login")

        @self.app.route("/api/health")
        def health_check():
            """API health check endpoint"""
            return jsonify(
                {
                    "status": "healthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "model_available": self.prediction_model is not None,
                }
            )

    def extract_loan_features(
        self, form_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract and validate loan features from form data"""
        try:
            return {
                "first_name": form_data.get("first_name", "").strip(),
                "last_name": form_data.get("last_name", "").strip(),
                "gender": int(form_data.get("gender", 0)),
                "marital_status": int(form_data.get("married", 0)),
                "dependents_count": int(form_data.get("dependents", 0)),
                "education_level": int(form_data.get("education", 0)),
                "employment_type": int(form_data.get("self_employed", 0)),
                "property_location": int(form_data.get("property_area", 0)),
                "credit_history": float(form_data.get("credit_history", 0)),
                "cibil_score": int(form_data.get("cibil_score", 300)),
                "applicant_income": float(form_data.get("applicant_income", 0)),
                "coapplicant_income": float(form_data.get("coapplicant_income", 0)),
                "loan_amount": float(form_data.get("loan_amount", 0)),
                "loan_tenure": float(form_data.get("loan_amount_term", 0)),
            }
        except (ValueError, TypeError) as e:
            logger.error(f"Feature extraction error: {e}")
            return None

    def generate_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate loan eligibility prediction using ML model"""
        # Feature engineering (same as original model)
        total_income = features["applicant_income"] + features["coapplicant_income"]
        loan_to_income_ratio = (
            features["loan_amount"] / total_income if total_income > 0 else 0
        )
        emi_feature = (
            features["loan_amount"] / features["loan_tenure"]
            if features["loan_tenure"] > 0
            else 0
        )

        # Log transformations
        applicant_income_log = np.log(features["applicant_income"] + 1)
        coapplicant_income_log = np.log(features["coapplicant_income"] + 1)
        loan_amount_log = np.log(features["loan_amount"] + 1)
        total_income_log = np.log(total_income + 1)

        # Create feature vector in the exact order expected by the model
        # The model was trained with BOTH raw features and engineered features
        # Must match the exact column order from model training
        feature_vector = pd.DataFrame(
            {
                "ApplicantIncome": [features["applicant_income"]],
                "ApplicantIncome_log": [applicant_income_log],
                "CIBIL_Score": [features["cibil_score"]],
                "CoapplicantIncome": [features["coapplicant_income"]],
                "CoapplicantIncome_log": [coapplicant_income_log],
                "Credit_History": [features["credit_history"]],
                "Dependents_1": [1 if features["dependents_count"] == 1 else 0],
                "Dependents_2": [1 if features["dependents_count"] == 2 else 0],
                "Dependents_3+": [1 if features["dependents_count"] >= 3 else 0],
                "Education_Not Graduate": [
                    1 if features["education_level"] == 1 else 0
                ],
                "EMI_feature": [emi_feature],
                "Gender_Male": [1 if features["gender"] == 1 else 0],
                "LoanAmount": [features["loan_amount"]],
                "LoanAmount_log": [loan_amount_log],
                "Loan_Amount_Term": [features["loan_tenure"]],
                "Loan_to_Income_Ratio": [loan_to_income_ratio],
                "Married_Yes": [1 if features["marital_status"] == 1 else 0],
                "Property_Area_Semiurban": [
                    1 if features["property_location"] == 1 else 0
                ],
                "Property_Area_Urban": [1 if features["property_location"] == 2 else 0],
                "Self_Employed_Yes": [1 if features["employment_type"] == 1 else 0],
                "Total_Income": [total_income],
                "Total_Income_log": [total_income_log],
            }
        )
        # Scale features and predict
        scaled_features = self.feature_scaler.transform(feature_vector)
        prediction = self.prediction_model.predict(scaled_features)
        prediction_probability = self.prediction_model.predict_proba(scaled_features)

        decision = "APPROVED" if prediction[0] == 1 else "REJECTED"
        confidence = float(prediction_probability[0][prediction[0]] * 100)

        return {"decision": decision, "confidence": round(confidence, 2)}

    def save_prediction_record(
        self, user_id: str, form_data: Dict[str, Any], prediction: Dict[str, Any]
    ):
        """Save prediction record to database"""
        try:
            features = self.extract_loan_features(form_data)

            with self.get_database_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO loan_predictions (
                        user_id, first_name, last_name, gender, marital_status,
                        dependents_count, education_level, employment_type, property_location,
                        credit_history, cibil_score, applicant_income, coapplicant_income,
                        loan_amount, loan_tenure, prediction_result, confidence_score
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        user_id,
                        features["first_name"],
                        features["last_name"],
                        features["gender"],
                        features["marital_status"],
                        features["dependents_count"],
                        features["education_level"],
                        features["employment_type"],
                        features["property_location"],
                        features["credit_history"],
                        features["cibil_score"],
                        features["applicant_income"],
                        features["coapplicant_income"],
                        features["loan_amount"],
                        features["loan_tenure"],
                        prediction["decision"],
                        prediction["confidence"],
                    ),
                )
                conn.commit()

        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")


# Create application instance
loan_app = LoanEligibilityApp()
app = loan_app.app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
