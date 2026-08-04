"""Loan Eligibility Prediction System.

A single-module Flask app: routes, validation and the prediction path all live
here, which suits five endpoints and keeps the request flow readable top to
bottom.

Two invariants are worth knowing before editing:

1. The feature vector in generate_prediction() is reordered to match
   models/feature_columns.pkl before scaling. The scaler was fitted on a
   DataFrame, so scikit-learn rejects any other column order.
2. The app is constructed at import time (`app` at the bottom) because that is
   the WSGI entrypoint gunicorn and Vercel load. Importing this module runs
   database setup and model loading.
"""

import os
import re
import logging
import pickle
import secrets
from datetime import datetime, timedelta, timezone
from typing import Dict, Any
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, session, jsonify, flash
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from contextlib import contextmanager
from functools import wraps
from dotenv import load_dotenv

# Read .env before any os.environ lookup below. Real environment variables
# already set by the platform win; .env only fills the gaps.
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ValidationError(ValueError):
    """Raised when submitted data fails server-side validation.

    Carries a message safe to return to the client, so callers can turn it
    into a 400 without leaking internals.
    """


# Accepted range for every loan field, validated server side. The browser form
# constrains these too, but a form is not a trust boundary: every one of these
# reaches the model, and an out-of-range value produces a confident nonsense
# prediction rather than an error.
LOAN_FIELD_RANGES = {
    "gender": (int, 0, 1),
    "married": (int, 0, 1),
    "dependents": (int, 0, 5),
    "education": (int, 0, 1),
    "self_employed": (int, 0, 1),
    "property_area": (int, 0, 2),
    "credit_history": (float, 0.0, 1.0),
    "cibil_score": (int, 300, 900),
    "applicant_income": (float, 1.0, 100_000_000.0),
    "coapplicant_income": (float, 0.0, 100_000_000.0),
    "loan_amount": (float, 1.0, 50_000_000.0),
    "loan_amount_term": (float, 1.0, 480.0),
}

NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z '\-]*$")
USER_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_]{3,50}$")
MIN_PASSWORD_LENGTH = 8


def format_bound(value: float) -> str:
    """Render a validation bound readably (50000000.0 -> '50,000,000')."""
    return f"{value:,.0f}" if value == int(value) else f"{value:g}"


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
        self.is_production = os.environ.get("FLASK_ENV", "production") != "development"

        secret_key = os.environ.get("SECRET_KEY")
        if not secret_key:
            if self.is_production:
                # A per-process random key silently invalidates every session on
                # restart, and gives each worker/serverless instance a different
                # key. Failing loudly beats logging users out at random.
                raise RuntimeError(
                    "SECRET_KEY environment variable is required when FLASK_ENV is "
                    "not 'development'. Generate one with: "
                    "python -c \"import secrets; print(secrets.token_urlsafe(32))\""
                )
            secret_key = secrets.token_urlsafe(32)
            logger.warning(
                "SECRET_KEY not set; using an ephemeral development key. "
                "Sessions will not survive a restart."
            )

        self.app.secret_key = secret_key
        self.app.config.update(
            DATABASE_PATH=os.environ.get("DATABASE_PATH", "loan_prediction.db"),
            MODEL_PATH=os.environ.get("MODEL_PATH", "models"),
            SESSION_COOKIE_HTTPONLY=True,
            SESSION_COOKIE_SAMESITE="Lax",
            SESSION_COOKIE_SECURE=self.is_production,
            PERMANENT_SESSION_LIFETIME=timedelta(hours=1),
        )

    def setup_database(self):
        """Initialize SQLite database for deployment compatibility"""
        with self.get_database_connection() as conn:
            self.enable_wal(conn)
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

            conn.commit()
            self.seed_default_admin(conn)
            conn.commit()
            logger.info("Database initialized successfully")

    def seed_default_admin(self, conn) -> None:
        """Create the default admin account from ADMIN_PASSWORD, if absent.

        Fails closed: with no ADMIN_PASSWORD set, a production deployment gets
        no admin account at all rather than one with a published password.
        """
        admin_id = os.environ.get("ADMIN_USERNAME", "admin")
        password = os.environ.get("ADMIN_PASSWORD")

        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM admins WHERE admin_id = ?", (admin_id,))
        if cursor.fetchone()[0] > 0:
            return

        if not password:
            if self.is_production:
                logger.warning(
                    "ADMIN_PASSWORD not set; no admin account created. "
                    "Set ADMIN_PASSWORD and restart to enable the admin dashboard."
                )
                return
            password = "admin123"
            logger.warning(
                "ADMIN_PASSWORD not set; seeding development admin '%s' with a "
                "well-known password. Never run this configuration in production.",
                admin_id,
            )

        # OR IGNORE, not a plain INSERT: the SELECT above is a check-then-act
        # race. Every gunicorn worker imports this module and runs startup
        # concurrently, so under `--workers 2` both can observe zero admins and
        # both insert, and the loser dies with a UNIQUE constraint violation
        # before it can serve a request.
        cursor.execute(
            "INSERT OR IGNORE INTO admins (admin_id, password_hash) VALUES (?, ?)",
            (admin_id, generate_password_hash(password)),
        )
        if cursor.rowcount:
            logger.info("Admin account '%s' created", admin_id)

    @contextmanager
    def get_database_connection(self):
        """Context manager for database connections.

        Three PRAGMAs matter here:

        - WAL lets readers run concurrently with a writer. Under the default
          rollback journal, one in-flight write blocks every read, which shows
          up as "database is locked" as soon as more than one request overlaps.
        - busy_timeout makes a blocked writer wait for the lock instead of
          failing instantly.
        - foreign_keys is off by default in SQLite, so the FK that
          loan_predictions declares against users was never actually enforced.
        """
        conn = sqlite3.connect(self.app.config["DATABASE_PATH"], timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            # Both are per-connection and need no database lock, so they are
            # safe to set on every connection. busy_timeout must come first:
            # it governs how long everything after it waits for a lock.
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
        finally:
            conn.close()

    def enable_wal(self, conn) -> None:
        """Switch the database to WAL journalling, once.

        WAL is a persistent property of the database file, so this only has to
        succeed once and every later connection inherits it. It is deliberately
        not in get_database_connection: the journal-mode switch needs an
        exclusive lock, and SQLite does not reliably invoke the busy handler
        for it, so running it per-connection makes concurrent workers fail with
        "database is locked" on startup.
        """
        try:
            mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if str(mode).lower() != "wal":
                logger.warning("Could not enable WAL; journal_mode is %r", mode)
        except sqlite3.OperationalError as e:
            # Another worker is mid-switch. Whoever wins sets it for everyone.
            logger.debug("WAL not set on this worker: %s", e)

    def load_ml_models(self):
        """Load the model, scaler and the feature order they were trained on.

        feature_columns.pkl is required, not optional: the scaler was fitted on
        a DataFrame, so scikit-learn rejects any input whose columns are in a
        different order. Loading the order from the artefact keeps the app and
        models/model_trainer.py from drifting apart.
        """
        self.prediction_model = None
        self.feature_scaler = None
        self.feature_columns = None

        model_dir = self.app.config["MODEL_PATH"]
        artefacts = {
            "model": os.path.join(model_dir, "loan_eligibility_model.pkl"),
            "scaler": os.path.join(model_dir, "feature_scaler.pkl"),
            "columns": os.path.join(model_dir, "feature_columns.pkl"),
        }

        missing = [name for name, path in artefacts.items() if not os.path.exists(path)]
        if missing:
            logger.warning(
                "ML artefacts missing (%s). Prediction unavailable - run "
                "`python models/model_trainer.py` to build them.",
                ", ".join(missing),
            )
            return

        try:
            loaded = {}
            for name, path in artefacts.items():
                # pickle.load executes arbitrary code, so it is only safe on
                # trusted input. These artefacts are not shipped or downloaded:
                # they are git-ignored and produced locally by
                # models/model_trainer.py from data/train.csv in this repo.
                # Never point MODEL_PATH at a directory you do not control.
                with open(path, "rb") as f:
                    loaded[name] = pickle.load(f)

            self.prediction_model = loaded["model"]
            self.feature_scaler = loaded["scaler"]
            self.feature_columns = list(loaded["columns"])
            logger.info(
                "ML models loaded successfully (%d features)", len(self.feature_columns)
            )
        except Exception as e:
            # Most likely a scikit-learn version mismatch against the pickle.
            logger.error("Failed to load ML models: %s", e, exc_info=True)
            self.prediction_model = None
            self.feature_scaler = None
            self.feature_columns = None

    @property
    def model_available(self) -> bool:
        """True when every artefact needed to serve a prediction is loaded."""
        return None not in (
            self.prediction_model,
            self.feature_scaler,
            self.feature_columns,
        )

    @staticmethod
    def unauthenticated_response(message: str, status: int, redirect_to: str):
        """Reject an unauthenticated caller in the shape it can understand.

        Redirecting an /api/ request sends an HTML login page to a fetch() that
        is about to call response.json(), so the browser reports a parse error
        instead of "your session expired".
        """
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "message": message}), status
        flash(message, "warning")
        return redirect(redirect_to)

    def require_login(self, f):
        """Decorator to require user authentication"""

        @wraps(f)
        def decorated_function(*args, **kwargs):
            if "user_id" not in session:
                return self.unauthenticated_response(
                    "Please log in to access this page.", 401, "/"
                )
            return f(*args, **kwargs)

        return decorated_function

    def require_admin(self, f):
        """Decorator to require admin authentication"""

        @wraps(f)
        def decorated_function(*args, **kwargs):
            if "admin_id" not in session:
                return self.unauthenticated_response(
                    "Admin access required.", 403, "/admin-login"
                )
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

                user_id = str(form_data["user_id"]).strip().lower()
                if not USER_ID_PATTERN.match(user_id):
                    return jsonify(
                        {
                            "success": False,
                            "message": "Username must be 3-50 characters, letters, "
                            "numbers and underscores only",
                        }
                    ), 400

                # Check password confirmation
                if form_data["password"] != form_data["confirm_password"]:
                    return jsonify(
                        {"success": False, "message": "Passwords do not match"}
                    ), 400

                if len(form_data["password"]) < MIN_PASSWORD_LENGTH:
                    return jsonify(
                        {
                            "success": False,
                            "message": f"Password must be at least "
                            f"{MIN_PASSWORD_LENGTH} characters",
                        }
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
                            user_id,
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
                logger.error("Registration error: %s", e, exc_info=True)
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
                        (str(user_id).strip().lower(),),
                    )
                    user = cursor.fetchone()

                if user and check_password_hash(user["password_hash"], str(password)):
                    # Drop any pre-login session so a fixed cookie cannot be
                    # carried across the privilege boundary.
                    session.clear()
                    session["user_id"] = user["user_id"]
                    session["full_name"] = user["full_name"]
                    session.permanent = True
                    return jsonify({"success": True, "message": "Login successful"})
                else:
                    logger.info("Failed login attempt for user_id=%r", user_id)
                    return jsonify(
                        {"success": False, "message": "Invalid credentials"}
                    ), 401

            except Exception as e:
                logger.error("Login error: %s", e, exc_info=True)
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
            if not self.model_available:
                return jsonify(
                    {"success": False, "message": "Prediction service unavailable"}
                ), 503

            try:
                form_data = request.get_json() if request.is_json else request.form
                loan_features = self.extract_loan_features(form_data)
            except ValidationError as e:
                return jsonify({"success": False, "message": str(e)}), 400

            try:
                prediction_result = self.generate_prediction(loan_features)

                # Save prediction to database
                self.save_prediction_record(
                    session["user_id"], loan_features, prediction_result
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
                logger.error("Prediction error: %s", e, exc_info=True)
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

                if not admin_id or not password:
                    return jsonify(
                        {"success": False, "message": "Admin ID and password required"}
                    ), 400

                with self.get_database_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "SELECT admin_id, password_hash FROM admins WHERE admin_id = ?",
                        (admin_id,),
                    )
                    admin = cursor.fetchone()

                if admin and check_password_hash(admin["password_hash"], str(password)):
                    session.clear()
                    session["admin_id"] = admin["admin_id"]
                    session.permanent = True
                    return jsonify(
                        {"success": True, "message": "Admin login successful"}
                    )
                else:
                    logger.warning(
                        "Failed admin login attempt for admin_id=%r from %s",
                        admin_id,
                        request.remote_addr,
                    )
                    return jsonify(
                        {"success": False, "message": "Invalid admin credentials"}
                    ), 401

            except Exception as e:
                logger.error("Admin login error: %s", e, exc_info=True)
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
                logger.error("Admin dashboard error: %s", e, exc_info=True)
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
            """Report liveness plus the state of each dependency."""
            try:
                with self.get_database_connection() as conn:
                    conn.execute("SELECT 1")
                database_available = True
            except Exception as e:
                logger.error("Health check database probe failed: %s", e)
                database_available = False

            healthy = database_available and self.model_available
            return jsonify(
                {
                    "status": "healthy" if healthy else "degraded",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model_available": self.model_available,
                    "database_available": database_available,
                }
            ), (200 if healthy else 503)

    @staticmethod
    def validate_name(form_data: Dict[str, Any], field: str) -> str:
        """Return a stripped, validated person name."""
        value = str(form_data.get(field) or "").strip()
        if not value:
            raise ValidationError(f"{field.replace('_', ' ').title()} is required")
        if len(value) > 50:
            raise ValidationError(
                f"{field.replace('_', ' ').title()} must be 50 characters or fewer"
            )
        if not NAME_PATTERN.match(value):
            raise ValidationError(
                f"{field.replace('_', ' ').title()} may only contain letters, "
                "spaces, apostrophes and hyphens"
            )
        return value

    @staticmethod
    def validate_number(form_data: Dict[str, Any], field: str):
        """Return a numeric field coerced to its declared type and range."""
        cast, low, high = LOAN_FIELD_RANGES[field]
        raw = form_data.get(field)
        if raw is None or str(raw).strip() == "":
            raise ValidationError(f"{field.replace('_', ' ').title()} is required")
        try:
            # int("1.0") raises, but HTML selects legitimately submit "1.0" for
            # whole-number fields, so route ints through float first.
            value = cast(float(raw))
        except (TypeError, ValueError):
            raise ValidationError(
                f"{field.replace('_', ' ').title()} must be a number"
            )
        if not low <= value <= high:
            raise ValidationError(
                f"{field.replace('_', ' ').title()} must be between "
                f"{format_bound(low)} and {format_bound(high)}"
            )
        return value

    def extract_loan_features(self, form_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a loan application and map it onto the database columns.

        Raises:
            ValidationError: with a client-safe message on the first bad field.
        """
        numbers = {
            field: self.validate_number(form_data, field)
            for field in LOAN_FIELD_RANGES
        }

        total_income = numbers["applicant_income"] + numbers["coapplicant_income"]
        # Guard against applications the model has no support for: the training
        # data tops out near 25x, so beyond 50x the prediction is extrapolation.
        if numbers["loan_amount"] > total_income * 50:
            raise ValidationError(
                "Loan amount must not exceed 50 times total monthly income"
            )

        return {
            "first_name": self.validate_name(form_data, "first_name"),
            "last_name": self.validate_name(form_data, "last_name"),
            "gender": numbers["gender"],
            "marital_status": numbers["married"],
            "dependents_count": numbers["dependents"],
            "education_level": numbers["education"],
            "employment_type": numbers["self_employed"],
            "property_location": numbers["property_area"],
            "credit_history": numbers["credit_history"],
            "cibil_score": numbers["cibil_score"],
            "applicant_income": numbers["applicant_income"],
            "coapplicant_income": numbers["coapplicant_income"],
            "loan_amount": numbers["loan_amount"],
            "loan_tenure": numbers["loan_amount_term"],
        }

    def generate_prediction(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate loan eligibility prediction using ML model"""
        model, scaler, columns = (
            self.prediction_model,
            self.feature_scaler,
            self.feature_columns,
        )
        if model is None or scaler is None or columns is None:
            raise RuntimeError("Prediction requested while ML artefacts are unloaded")

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

        # Build every feature the trainer produces. Order here is irrelevant —
        # it is imposed below from feature_columns.pkl, which is the single
        # source of truth for what the scaler was fitted on.
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

        missing = set(columns) - set(feature_vector.columns)
        if missing:
            # The trainer's feature set changed without this function following.
            raise RuntimeError(
                f"Model expects features this app does not build: {sorted(missing)}. "
                "Re-run models/model_trainer.py or update generate_prediction()."
            )

        # Reorder to the exact column sequence the scaler was fitted on.
        scaled_features = scaler.transform(feature_vector[columns])
        prediction = model.predict(scaled_features)
        prediction_probability = model.predict_proba(scaled_features)

        decision = "APPROVED" if prediction[0] == 1 else "REJECTED"
        confidence = float(prediction_probability[0][prediction[0]] * 100)

        return {"decision": decision, "confidence": round(confidence, 2)}

    def save_prediction_record(
        self, user_id: str, features: Dict[str, Any], prediction: Dict[str, Any]
    ):
        """Persist a prediction. Failure is logged but never fails the request:
        the user already has their answer, and losing an audit row is less bad
        than losing the response."""
        try:
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
            logger.error("Failed to save prediction: %s", e, exc_info=True)


# Create application instance
loan_app = LoanEligibilityApp()
app = loan_app.app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
