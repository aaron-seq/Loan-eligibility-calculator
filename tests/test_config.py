"""Configuration, security defaults and the health endpoint."""

import importlib
import sys

import pytest


def _reload_app():
    sys.modules.pop("app", None)
    return importlib.import_module("app")


def test_production_without_secret_key_refuses_to_start(monkeypatch):
    """A per-process random key silently logs everyone out on restart, so an
    unset SECRET_KEY must fail loudly rather than degrade quietly.

    The app is built at module scope, so this surfaces as an ImportError-time
    failure: gunicorn and Vercel refuse to boot rather than serving with a
    throwaway key.
    """
    monkeypatch.setenv("FLASK_ENV", "production")
    monkeypatch.delenv("SECRET_KEY", raising=False)

    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        _reload_app()


def test_production_without_admin_password_creates_no_admin(monkeypatch, tmp_path):
    """Fail closed: no ADMIN_PASSWORD means no admin account at all, rather
    than one with a published default password."""
    monkeypatch.setenv("FLASK_ENV", "production")
    monkeypatch.setenv("SECRET_KEY", "a-real-secret-for-this-test")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "prod.db"))
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)

    instance = _reload_app().LoanEligibilityApp()

    with instance.get_database_connection() as conn:
        assert conn.execute("SELECT COUNT(*) c FROM admins").fetchone()["c"] == 0

    response = instance.app.test_client().post(
        "/api/admin/login", json={"admin_id": "admin", "password": "admin123"}
    )
    assert response.status_code == 401


def test_production_enables_secure_session_cookies(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASK_ENV", "production")
    monkeypatch.setenv("SECRET_KEY", "a-real-secret-for-this-test")
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "prod.db"))

    config = _reload_app().LoanEligibilityApp().app.config

    assert config["SESSION_COOKIE_SECURE"] is True
    assert config["SESSION_COOKIE_HTTPONLY"] is True
    assert config["SESSION_COOKIE_SAMESITE"] == "Lax"


def test_development_allows_an_ephemeral_secret_key(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASK_ENV", "development")
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "dev.db"))

    instance = _reload_app().LoanEligibilityApp()

    assert instance.app.secret_key
    assert instance.app.config["SESSION_COOKIE_SECURE"] is False


def test_health_reports_each_dependency(client):
    response = client.get("/api/health")
    body = response.get_json()

    assert set(body) == {
        "status",
        "timestamp",
        "model_available",
        "database_available",
    }
    assert body["database_available"] is True
    # 503 when the ML artefacts have not been built; both are valid states.
    assert response.status_code == (200 if body["model_available"] else 503)


def test_health_timestamp_is_timezone_aware(client):
    """datetime.utcnow() produced a naive timestamp that read as local time."""
    timestamp = client.get("/api/health").get_json()["timestamp"]
    assert timestamp.endswith("+00:00")


def test_prediction_unavailable_without_model(flask_app, auth_client, loan_application):
    """With artefacts unloaded the endpoint must return 503, not crash."""
    flask_app.prediction_model = None
    flask_app.feature_scaler = None
    flask_app.feature_columns = None

    response = auth_client.post("/api/predict", json=loan_application)

    assert response.status_code == 503
    assert flask_app.model_available is False
