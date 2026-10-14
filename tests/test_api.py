"""HTTP surface: response contracts, validation, and the degraded path."""

from __future__ import annotations

import pytest

VALID = {
    "applicant_income": 50_000,
    "coapplicant_income": 10_000,
    "loan_amount": 1_500_000,
    "loan_term_months": 240,
    "credit_history": 1,
    "dependents": 1,
    "education": "graduate",
    "self_employed": False,
    "property_area": "urban",
    "cibil_score": 730,
}


class TestHealth:
    def test_reports_ok_with_a_model_loaded(self, client):
        body = client.get("/api/health").get_json()
        assert body["status"] == "ok"
        assert body["model_status"] == "ready"
        assert "version" in body["model"]

    def test_reports_degraded_without_a_model_but_still_returns_200(self, modelless_app):
        """A missing model is not a reason for an orchestrator to restart the
        process — restarting will not produce one."""
        response = modelless_app.test_client().get("/api/health")
        assert response.status_code == 200
        body = response.get_json()
        assert body["status"] == "degraded"
        assert "train" in body["model_error"]


class TestAssess:
    def test_returns_a_complete_assessment(self, client):
        data = client.post("/api/assess", json=VALID).get_json()["data"]
        assert 0 <= data["probability"] <= 1
        assert data["band"] in ("likely", "borderline", "unlikely")
        assert data["affordability"]["monthly_instalment"] > 0
        assert data["rules"]
        assert data["disclaimer"]

    def test_anonymous_results_are_not_saved(self, client):
        data = client.post("/api/assess", json=VALID).get_json()["data"]
        assert data["saved"] is False
        assert "save_hint" in data

    def test_signed_in_results_are_saved_and_retrievable(self, registered_client):
        headers = {"X-CSRF-Token": registered_client.csrf}
        data = registered_client.post(
            "/api/assess", json=VALID, headers=headers
        ).get_json()["data"]
        assert data["saved"] is True

        listing = registered_client.get("/api/applications").get_json()["data"]
        assert listing["count"] == 1

        record = registered_client.get(
            f"/api/applications/{data['application_id']}"
        ).get_json()["data"]
        assert record["payload"]["loan_amount"] == VALID["loan_amount"]

    def test_accepts_form_encoded_bodies_too(self, client):
        response = client.post("/api/assess", data=VALID)
        assert response.status_code == 200

    @pytest.mark.parametrize(
        "field,value",
        [
            ("loan_amount", -1),
            ("loan_amount", 0),
            ("loan_term_months", 3),
            ("loan_term_months", 600),
            ("credit_history", 5),
            ("cibil_score", 100),
            ("cibil_score", 1000),
            ("applicant_income", -5),
            ("property_area", "atlantis"),
            ("education", "phd"),
            ("dependents", 99),
        ],
    )
    def test_out_of_range_input_is_rejected_with_the_field_named(
        self, client, field, value
    ):
        body = client.post("/api/assess", json={**VALID, field: value}).get_json()
        assert body["error"] == "validation_failed"
        assert field in body["errors"]

    def test_missing_required_fields_are_reported(self, client):
        body = client.post("/api/assess", json={}).get_json()
        assert body["error"] == "validation_failed"
        assert "loan_amount" in body["errors"]

    def test_optional_cibil_score_may_be_omitted(self, client):
        payload = {k: v for k, v in VALID.items() if k != "cibil_score"}
        assert client.post("/api/assess", json=payload).status_code == 200

    def test_blank_cibil_score_from_a_form_is_treated_as_omitted(self, client):
        assert (
            client.post("/api/assess", data={**VALID, "cibil_score": ""}).status_code
            == 200
        )

    def test_returns_503_when_no_model_is_loaded(self, modelless_app):
        response = modelless_app.test_client().post("/api/assess", json=VALID)
        assert response.status_code == 503
        body = response.get_json()
        assert body["error"] == "model_unavailable"
        assert "loanapp.ml.train" in body["message"]  # actionable, not just "failed"


class TestCompare:
    def test_scores_multiple_scenarios_and_names_the_best(self, client):
        response = client.post(
            "/api/compare",
            json={"scenarios": [VALID, {**VALID, "loan_amount": 400_000}]},
        )
        data = response.get_json()["data"]
        assert len(data["scenarios"]) == 2
        assert data["best_index"] == 1  # the smaller loan

    def test_requires_at_least_two_scenarios(self, client):
        assert client.post("/api/compare", json={"scenarios": [VALID]}).status_code == 400

    def test_caps_the_number_of_scenarios(self, client):
        response = client.post("/api/compare", json={"scenarios": [VALID] * 6})
        assert response.status_code == 400

    def test_invalid_scenarios_are_reported_by_index(self, client):
        body = client.post(
            "/api/compare", json={"scenarios": [VALID, {**VALID, "loan_amount": -1}]}
        ).get_json()
        assert any("1" in key for key in body["errors"])


class TestModelCard:
    def test_publishes_metrics_and_the_feature_list(self, client):
        data = client.get("/api/model-card").get_json()["data"]
        assert data["version"]
        assert data["features"]
        assert "roc_auc" in data["metrics"]
        assert data["disclaimer"]


class TestApplicationLifecycle:
    def test_delete_removes_the_record(self, registered_client):
        headers = {"X-CSRF-Token": registered_client.csrf}
        created = registered_client.post(
            "/api/assess", json=VALID, headers=headers
        ).get_json()["data"]["application_id"]

        assert (
            registered_client.delete(
                f"/api/applications/{created}", headers=headers
            ).status_code
            == 200
        )
        assert registered_client.get(f"/api/applications/{created}").status_code == 404

    def test_unknown_application_returns_404(self, registered_client):
        assert registered_client.get("/api/applications/999999").status_code == 404


class TestErrorContract:
    def test_unknown_api_route_returns_json_not_html(self, client):
        response = client.get("/api/nope")
        assert response.status_code == 404
        assert response.get_json()["error"] == "not_found"

    def test_rate_limiting_returns_429_with_retry_after(self, app):
        app.config["RATE_LIMITS"]["predict"] = (2, 60)
        client = app.test_client()
        for _ in range(2):
            client.post("/api/assess", json=VALID)
        response = client.post("/api/assess", json=VALID)
        assert response.status_code == 429
        assert response.headers["Retry-After"]
