"""Prediction endpoint: the feature contract, validation, and auth."""

import pickle

import pytest

from conftest import requires_model
from app import LOAN_FIELD_RANGES


@requires_model
def test_feature_vector_matches_trained_column_order(flask_app, loan_application):
    """Regression guard for the bug that made every prediction a 500.

    app.py built its feature DataFrame in alphabetical order while the scaler
    was fitted in the trainer's order. scikit-learn compares DataFrame column
    names *and* their order, so transform() raised on every call.
    """
    features = flask_app.extract_loan_features(loan_application)
    result = flask_app.generate_prediction(features)

    assert result["decision"] in {"APPROVED", "REJECTED"}
    assert 0.0 <= result["confidence"] <= 100.0


@requires_model
def test_app_builds_every_feature_the_scaler_expects(flask_app):
    """The app must produce exactly the feature set the trainer persisted."""
    # Trusted input: this artefact is produced locally by the trainer in this
    # repo, never downloaded. See the note in app.load_ml_models.
    with open("models/feature_columns.pkl", "rb") as f:
        trained_columns = set(pickle.load(f))

    assert set(flask_app.feature_columns) == trained_columns
    assert len(flask_app.feature_columns) == len(trained_columns)


@requires_model
def test_predict_returns_a_decision(auth_client, loan_application):
    response = auth_client.post("/api/predict", json=loan_application)
    body = response.get_json()

    assert response.status_code == 200
    assert body["success"] is True
    assert body["prediction"] in {"APPROVED", "REJECTED"}
    assert 0.0 <= body["confidence"] <= 100.0


@requires_model
def test_credit_history_moves_the_decision(auth_client, loan_application):
    """Credit history is the dominant signal; the model should react to it."""
    good = auth_client.post(
        "/api/predict", json={**loan_application, "credit_history": 1.0}
    ).get_json()
    bad = auth_client.post(
        "/api/predict", json={**loan_application, "credit_history": 0.0}
    ).get_json()

    assert good["prediction"] == "APPROVED"
    assert bad["prediction"] == "REJECTED"


@requires_model
def test_cibil_score_is_a_documented_synthetic_feature(flask_app, loan_application):
    """Pins the caveat documented in README 'Model honesty'.

    CIBIL_Score is fabricated by the trainer with np.random.randint, so the
    model's sensitivity to it is memorised noise. This test does not assert the
    behaviour is *good* — it asserts the README's description stays true, so
    the caveat cannot silently rot if the trainer changes.
    """
    confidences = {
        score: flask_app.generate_prediction(
            flask_app.extract_loan_features({**loan_application, "cibil_score": score})
        )["confidence"]
        for score in (300, 500, 800)
    }

    # Still noise-sensitive: if this ever collapses to a flat line, the feature
    # became inert and the README section needs rewriting.
    assert len(set(confidences.values())) > 1, (
        "CIBIL no longer affects output; update the README 'Model honesty' section"
    )

    # Credit_History remains the dominant, genuine signal.
    approved = flask_app.generate_prediction(
        flask_app.extract_loan_features({**loan_application, "credit_history": 1.0})
    )
    rejected = flask_app.generate_prediction(
        flask_app.extract_loan_features({**loan_application, "credit_history": 0.0})
    )
    assert approved["decision"] == "APPROVED"
    assert rejected["decision"] == "REJECTED"


@requires_model
def test_prediction_is_persisted(auth_client, flask_app, loan_application):
    auth_client.post("/api/predict", json=loan_application)

    with flask_app.get_database_connection() as conn:
        rows = conn.execute(
            "SELECT user_id, first_name, prediction_result FROM loan_predictions"
        ).fetchall()

    assert len(rows) == 1
    assert rows[0]["user_id"] == "ada_lovelace"
    assert rows[0]["first_name"] == "Ada"


def test_predict_requires_authentication(client, loan_application):
    """API callers get a JSON 401, not a redirect to an HTML login page that
    the dashboard's fetch() would fail to parse."""
    response = client.post("/api/predict", json=loan_application)

    assert response.status_code == 401
    assert response.get_json()["success"] is False


def test_page_routes_still_redirect_when_logged_out(client):
    """Browser navigation keeps the redirect; only /api/ returns JSON."""
    assert client.get("/dashboard").status_code == 302
    assert client.get("/admin/dashboard").status_code == 302


@pytest.mark.parametrize("field", sorted(LOAN_FIELD_RANGES))
def test_missing_required_field_is_rejected(auth_client, loan_application, field):
    """Absent fields must 400, not silently default to zero and predict."""
    incomplete = {k: v for k, v in loan_application.items() if k != field}
    response = auth_client.post("/api/predict", json=incomplete)

    assert response.status_code == 400
    assert "required" in response.get_json()["message"].lower()


@pytest.mark.parametrize(
    "field,value",
    [
        ("cibil_score", 299),
        ("cibil_score", 901),
        ("gender", 2),
        ("married", -1),
        ("dependents", 6),
        ("education", 7),
        ("property_area", 3),
        ("credit_history", 1.5),
        ("applicant_income", 0),
        ("coapplicant_income", -1),
        ("loan_amount", 0),
        ("loan_amount_term", 481),
    ],
)
def test_out_of_range_values_are_rejected(
    auth_client, loan_application, field, value
):
    response = auth_client.post(
        "/api/predict", json={**loan_application, field: value}
    )

    assert response.status_code == 400
    assert "between" in response.get_json()["message"].lower()


@pytest.mark.parametrize("value", ["", "   ", "Robert'); DROP TABLE users;--", "N4me"])
def test_invalid_names_are_rejected(auth_client, loan_application, value):
    response = auth_client.post(
        "/api/predict", json={**loan_application, "first_name": value}
    )
    assert response.status_code == 400


def test_non_numeric_value_is_rejected(auth_client, loan_application):
    response = auth_client.post(
        "/api/predict", json={**loan_application, "applicant_income": "lots"}
    )

    assert response.status_code == 400
    assert "number" in response.get_json()["message"].lower()


def test_loan_far_beyond_income_is_rejected(auth_client, loan_application):
    """Guards the model against extrapolating past its training support."""
    response = auth_client.post(
        "/api/predict",
        json={
            **loan_application,
            "applicant_income": 1000,
            "coapplicant_income": 0,
            "loan_amount": 5_000_000,
        },
    )

    assert response.status_code == 400
    assert "50 times" in response.get_json()["message"]


def test_html_select_decimal_is_accepted(auth_client, loan_application):
    """The dashboard's credit-history select submits '1.0' for an int-typed
    field; coercion must not reject it."""
    response = auth_client.post(
        "/api/predict", json={**loan_application, "dependents": "2.0"}
    )
    assert response.status_code in {200, 503}
