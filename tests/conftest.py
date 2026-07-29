"""Shared fixtures.

Two kinds of test app:

* ``app`` — a real model bundle loaded from ``models/``. Skipped if none has
  been trained, so a fresh checkout does not fail with a confusing error.
* ``modelless_app`` — no bundle at all, to prove the degraded path behaves.

Every test gets its own SQLite file in a tmp directory, so nothing touches a
developer's real database.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loanapp import create_app  # noqa: E402
from loanapp.config import Config  # noqa: E402
from loanapp.features import LoanApplication  # noqa: E402
from loanapp.ml.registry import bundle_path  # noqa: E402
from loanapp.security import limiter  # noqa: E402

MODEL_DIR = str(Path(__file__).resolve().parent.parent / "models")


def _config(tmp_path: Path, model_dir: str) -> Config:
    return Config(
        env="testing",
        secret_key="test-secret-key-at-least-32-characters-long",
        database_path=str(tmp_path / "test.db"),
        model_dir=model_dir,
        cookie_secure=False,
    )


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """The limiter is process-global; leaking counts between tests makes
    unrelated tests fail once a suite grows."""
    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture
def trained_model_dir() -> str:
    if not os.path.exists(bundle_path(MODEL_DIR)):
        pytest.skip("No trained model. Run: python -m loanapp.ml.train")
    return MODEL_DIR


@pytest.fixture
def app(tmp_path, trained_model_dir):
    return create_app(_config(tmp_path, trained_model_dir))


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def modelless_app(tmp_path):
    """An app with no model bundle, for testing the degraded path."""
    empty = tmp_path / "no-models"
    empty.mkdir()
    return create_app(_config(tmp_path, str(empty)))


@pytest.fixture
def assessor(app):
    return app.extensions["assessor"]


@pytest.fixture
def sample_application() -> LoanApplication:
    return LoanApplication(
        applicant_income=50_000,
        coapplicant_income=20_000,
        loan_amount=2_000_000,
        loan_term_months=240,
        credit_history=1,
        dependents=1,
        education="graduate",
        self_employed=False,
        property_area="urban",
        cibil_score=740,
    )


@pytest.fixture
def registered_client(client):
    """A client with a signed-in user, plus that user's CSRF token."""
    token = client.get("/api/csrf-token").get_json()["data"]["csrf_token"]
    response = client.post(
        "/register",
        data={
            "user_id": "testuser",
            "email": "test@example.com",
            "full_name": "Test User",
            "password": "correct-horse-battery",
            "confirm_password": "correct-horse-battery",
            "csrf_token": token,
        },
        follow_redirects=False,
    )
    assert response.status_code in (201, 302), response.data
    client.csrf = client.get("/api/csrf-token").get_json()["data"]["csrf_token"]
    return client
