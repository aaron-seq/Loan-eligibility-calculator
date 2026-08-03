"""Shared fixtures.

Every test runs against a throwaway SQLite file and a real Flask test client,
so the request/response cycle under test is the one that ships.
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# app.py instantiates the application at module scope (the WSGI entrypoint
# gunicorn and Vercel expect), so importing it anywhere in the suite runs the
# full startup path. Seed a safe development environment before that happens.
# setdefault, so a real environment variable still wins.
os.environ.setdefault("FLASK_ENV", "development")
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-used-in-production")
os.environ.setdefault(
    "DATABASE_PATH", os.path.join(tempfile.mkdtemp(prefix="loan-tests-"), "import.db")
)

MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models")
)
MODEL_ARTEFACTS = [
    "loan_eligibility_model.pkl",
    "feature_scaler.pkl",
    "feature_columns.pkl",
]

# Tests needing a real prediction are skipped rather than failed when the
# artefacts have not been built, so a fresh clone still gets a green run.
models_available = all(
    os.path.exists(os.path.join(MODEL_DIR, name)) for name in MODEL_ARTEFACTS
)
requires_model = pytest.mark.skipif(
    not models_available,
    reason="ML artefacts absent; run `python models/model_trainer.py`",
)


@pytest.fixture
def flask_app(tmp_path, monkeypatch):
    """A fully configured app instance backed by a temporary database."""
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("SECRET_KEY", "test-secret-key-not-used-in-production")
    monkeypatch.setenv("ADMIN_PASSWORD", "test-admin-password")
    monkeypatch.setenv("FLASK_ENV", "development")

    # Reimported per test so setup_app/setup_database observe the patched env.
    sys.modules.pop("app", None)
    import app as app_module

    instance = app_module.LoanEligibilityApp()
    instance.app.config["TESTING"] = True
    return instance


@pytest.fixture
def client(flask_app):
    """Anonymous test client."""
    return flask_app.app.test_client()


@pytest.fixture
def user_credentials():
    return {
        "user_id": "ada_lovelace",
        "email": "ada@example.com",
        "full_name": "Ada Lovelace",
        "password": "CorrectHorse1",
        "confirm_password": "CorrectHorse1",
    }


@pytest.fixture
def auth_client(client, user_credentials):
    """Test client with a registered, logged-in user session."""
    assert client.post("/api/register", json=user_credentials).status_code == 200
    response = client.post(
        "/api/login",
        json={
            "user_id": user_credentials["user_id"],
            "password": user_credentials["password"],
        },
    )
    assert response.status_code == 200
    return client


@pytest.fixture
def loan_application():
    """A well-formed application that passes validation."""
    return {
        "first_name": "Ada",
        "last_name": "Lovelace",
        "gender": 1,
        "married": 1,
        "dependents": 2,
        "education": 0,
        "self_employed": 0,
        "property_area": 2,
        "credit_history": 1.0,
        "cibil_score": 750,
        "applicant_income": 5000,
        "coapplicant_income": 1500,
        "loan_amount": 150,
        "loan_amount_term": 360,
    }
