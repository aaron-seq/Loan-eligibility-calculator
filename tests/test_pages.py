"""Browser flow.

The headline regression: the previous form posted to a JSON endpoint, so
submitting it showed the applicant raw JSON instead of a result page.
"""

from __future__ import annotations

import pytest

FORM = {
    "applicant_income": "60000",
    "coapplicant_income": "0",
    "loan_amount": "1800000",
    "loan_term_months": "240",
    "credit_history": "1",
    "dependents": "1",
    "education": "graduate",
    "property_area": "urban",
    "cibil_score": "720",
}


def form_with_csrf(client, **overrides):
    client.get("/apply")
    token = client.get("/api/csrf-token").get_json()["data"]["csrf_token"]
    return {**FORM, **overrides, "csrf_token": token}


@pytest.mark.parametrize("path", ["/", "/apply", "/login", "/signup", "/how-it-works"])
def test_public_pages_render(client, path):
    response = client.get(path)
    assert response.status_code == 200
    assert b"<html" in response.data


class TestApplicationFlow:
    def test_submitting_the_form_renders_a_result_page_not_json(self, client):
        response = client.post("/apply", data=form_with_csrf(client))
        assert response.status_code == 200
        body = response.data.decode()
        assert "<html" in body
        assert "Your estimate" in body
        assert not body.strip().startswith("{")

    def test_the_result_page_shows_reasons_and_the_disclaimer(self, client):
        body = client.post("/apply", data=form_with_csrf(client)).data.decode()
        assert "Why the estimate landed here" in body
        assert "Not a loan offer" in body

    def test_the_result_page_shows_the_instalment(self, client):
        body = client.post("/apply", data=form_with_csrf(client)).data.decode()
        assert "Monthly instalment" in body

    def test_invalid_input_re_renders_the_form_with_errors(self, client):
        response = client.post("/apply", data=form_with_csrf(client, loan_amount="-5"))
        assert response.status_code == 400
        body = response.data.decode()
        assert "correct the highlighted fields" in body.lower()
        # The applicant's other answers survive, so they do not retype the form.
        assert "60000" in body

    def test_the_apply_page_warns_when_no_model_is_loaded(self, modelless_app):
        body = modelless_app.test_client().get("/").data.decode()
        assert "offline" in body.lower()


class TestSessionFlow:
    def test_register_then_see_the_dashboard(self, registered_client):
        response = registered_client.get("/dashboard")
        assert response.status_code == 200
        assert b"My applications" in response.data

    def test_dashboard_requires_sign_in(self, client):
        response = client.get("/dashboard")
        assert response.status_code == 302
        assert "/login" in response.headers["Location"]

    def test_a_saved_application_appears_in_the_dashboard(self, registered_client):
        registered_client.post("/apply", data=form_with_csrf(registered_client))
        body = registered_client.get("/dashboard").data.decode()
        assert "1,800,000" in body

    def test_re_running_prefills_the_form(self, registered_client):
        registered_client.post("/apply", data=form_with_csrf(registered_client))
        record_id = registered_client.get("/api/applications").get_json()["data"][
            "applications"
        ][0]["id"]
        body = registered_client.get(f"/apply?from={record_id}").data.decode()
        assert 'value="1800000' in body.replace(".0", "")

    def test_prefill_ignores_another_users_application(self, app, registered_client):
        from loanapp.security import hash_password

        app.extensions["user_repository"].create(
            "stranger", "s@example.com", "Stranger", hash_password("x" * 20)
        )
        foreign_id = app.extensions["application_repository"].save(
            user_id="stranger",
            label="private",
            payload={"loan_amount": 99_999_999},
            assessment={
                "probability": 0.5,
                "band": "borderline",
                "model": {"version": "v"},
            },
        )
        body = registered_client.get(f"/apply?from={foreign_id}").data.decode()
        assert "99999999" not in body

    def test_sign_out_ends_the_session(self, registered_client):
        registered_client.post("/logout", data={"csrf_token": registered_client.csrf})
        assert registered_client.get("/dashboard").status_code == 302


class TestHowItWorks:
    def test_states_the_limitations_plainly(self, client):
        body = client.get("/how-it-works").data.decode().lower()
        assert "614" in body
        assert "not a loan offer" in body

    def test_discloses_what_the_model_does_not_use(self, client):
        body = client.get("/how-it-works").data.decode().lower()
        assert "gender" in body and "marital status" in body

    def test_shows_the_live_model_metrics(self, client):
        body = client.get("/how-it-works").data.decode().lower()
        assert "roc" in body or "brier" in body


class TestAdminPages:
    def test_admin_dashboard_rejects_a_normal_user(self, registered_client):
        response = registered_client.get("/admin/")
        assert response.status_code == 302

    def test_admin_dashboard_renders_for_an_admin(self, app):
        from loanapp.security import hash_password

        app.extensions["user_repository"].create(
            "boss", "boss@example.com", "Boss", hash_password("x" * 20), is_admin=True
        )
        client = app.test_client()
        token = client.get("/api/csrf-token").get_json()["data"]["csrf_token"]
        client.post(
            "/login",
            json={"user_id": "boss", "password": "x" * 20, "csrf_token": token},
        )
        response = client.get("/admin/")
        assert response.status_code == 200
        assert b"Operations overview" in response.data


class TestErrorPages:
    def test_unknown_page_renders_html_not_a_traceback(self, client):
        response = client.get("/definitely-not-a-page")
        assert response.status_code == 404
        assert b"<html" in response.data
        assert b"Traceback" not in response.data
