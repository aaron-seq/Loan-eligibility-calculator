"""Configuration, security defaults and the health endpoint."""

import importlib
import os
import sqlite3
import subprocess
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


def test_concurrent_worker_startup(tmp_path):
    """Regression: `gunicorn --workers 2` could not boot.

    Real processes, because the failure is cross-process and a same-process
    loop does not reproduce it. Every worker imports app.py and runs the full
    startup path against one database file, which surfaced two races:

      UNIQUE constraint failed: admins.admin_id
          check-then-insert in seed_default_admin; two workers both saw an
          empty table and both inserted.
      database is locked
          PRAGMA journal_mode=WAL taken per-connection before busy_timeout was
          set. SQLite does not reliably run the busy handler for a
          journal-mode switch, so workers collided on the exclusive lock.

    Found by running the Docker image with two workers, not by the rest of
    this suite.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    worker = (
        f"import sys; sys.path.insert(0, r'{repo_root}');"
        "import app; app.LoanEligibilityApp()"
    )
    env = {
        **os.environ,
        "FLASK_ENV": "development",
        "SECRET_KEY": "concurrent-startup-test",
        "ADMIN_PASSWORD": "shared-admin-password",
        "DATABASE_PATH": str(tmp_path / "concurrent.db"),
    }

    processes = [
        subprocess.Popen(
            [sys.executable, "-c", worker],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        for _ in range(4)
    ]

    failures = []
    for process in processes:
        _, stderr = process.communicate(timeout=120)
        if process.returncode != 0:
            failures.append(stderr.decode(errors="replace").strip().splitlines()[-1])

    assert not failures, "workers failed to start: " + "; ".join(failures)

    connection = sqlite3.connect(str(tmp_path / "concurrent.db"))
    try:
        admins = connection.execute(
            "SELECT COUNT(*) FROM admins WHERE admin_id = 'admin'"
        ).fetchone()[0]
    finally:
        connection.close()
    assert admins == 1


def test_database_pragmas_are_applied(flask_app):
    """WAL + busy_timeout keep overlapping requests from hitting
    'database is locked'; foreign_keys is off by default in SQLite, so the FK
    loan_predictions declares was never enforced."""
    with flask_app.get_database_connection() as conn:
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_concurrent_readers_do_not_block_a_writer(flask_app):
    """Under the default rollback journal an open read transaction blocks a
    write; WAL is what makes this pass."""
    with flask_app.get_database_connection() as reader:
        reader.execute("BEGIN")
        reader.execute("SELECT * FROM users").fetchall()

        with flask_app.get_database_connection() as writer:
            try:
                writer.execute(
                    "INSERT INTO users (user_id, email, full_name, password_hash) "
                    "VALUES (?, ?, ?, ?)",
                    ("concurrent", "c@example.com", "Concurrent", "hash"),
                )
                writer.commit()
            except sqlite3.OperationalError as e:  # pragma: no cover
                raise AssertionError(f"write blocked by open reader: {e}")


def test_orphan_prediction_is_rejected(flask_app):
    """The declared foreign key is now actually enforced."""
    with flask_app.get_database_connection() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO loan_predictions (user_id, first_name, last_name) "
                "VALUES (?, ?, ?)",
                ("no_such_user", "Ghost", "User"),
            )
            conn.commit()


def test_prediction_unavailable_without_model(flask_app, auth_client, loan_application):
    """With artefacts unloaded the endpoint must return 503, not crash."""
    flask_app.prediction_model = None
    flask_app.feature_scaler = None
    flask_app.feature_columns = None

    response = auth_client.post("/api/predict", json=loan_application)

    assert response.status_code == 503
    assert flask_app.model_available is False
