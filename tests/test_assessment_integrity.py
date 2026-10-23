"""Issue #39: an unsupported estimate must stay unavailable across outputs."""

import pytest
from test_api import VALID
from test_pages import form_with_csrf

from loanapp.security import hash_password

UNSUPPORTED = {**VALID, "applicant_income": 0, "coapplicant_income": 0}


def test_anonymous_unsupported_assessment_has_no_model_claim(client):
    response = client.post("/api/assess", json=UNSUPPORTED)
    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["out_of_range"] is True
    assert data["probability"] is None
    assert data["band"] == "unavailable"
    assert data["band_label"] == "Estimate unavailable"
    assert data["factors"] == data["suggestions"] == []
    assert any(rule["code"] == "income_present" for rule in data["rules"])
    assert data["affordability"]["monthly_instalment"] > 0


@pytest.mark.parametrize(
    "scenarios,best", [([UNSUPPORTED, VALID], 1), ([UNSUPPORTED] * 2, None)]
)
def test_comparison_never_ranks_unsupported_scenarios(client, scenarios, best):
    response = client.post("/api/compare", json={"scenarios": scenarios})
    assert response.status_code == 200
    data = response.get_json()["data"]
    assert data["best_index"] == best
    first = data["scenarios"][0]
    assert first["probability"] is None
    assert first["band"] == "unavailable"
    assert first["out_of_range_reason"]


def test_unsupported_saved_result_stays_unavailable_in_every_owner_view(
    registered_client,
):
    response = registered_client.post(
        "/api/assess",
        json={**UNSUPPORTED, "label": "No income scenario"},
        headers={"X-CSRF-Token": registered_client.csrf},
    )
    assert response.status_code == 200
    result = response.get_json()["data"]
    assert result["saved"] is True
    record_id = result["application_id"]
    stored = registered_client.get(f"/api/applications/{record_id}").get_json()["data"]
    assert stored["probability"] is None
    assert stored["assessment"]["probability"] is None
    history = registered_client.get("/api/applications").get_json()["data"]
    assert history["applications"][0]["probability"] is None
    for path in ("/dashboard", f"/applications/{record_id}"):
        page = registered_client.get(path)
        assert page.status_code == 200
        assert b"Estimate unavailable" in page.data
        assert b"20%" not in page.data
    assert b"Monthly instalment" in page.data
    assert b"No income recorded" in page.data


def test_unsupported_html_result_has_no_model_band(client):
    page = client.post("/apply", data=form_with_csrf(client, applicant_income="0"))
    assert page.status_code == 200
    assert b"Estimate unavailable" in page.data
    assert b"Unlikely to be approved" not in page.data


def test_aggregates_exclude_unsupported_estimates(app, registered_client):
    headers = {"X-CSRF-Token": registered_client.csrf}
    supported = registered_client.post(
        "/api/assess", json=VALID, headers=headers
    ).get_json()["data"]
    registered_client.post("/api/assess", json=UNSUPPORTED, headers=headers)
    stats = app.extensions["application_repository"].aggregate()
    assert stats["total"] == 2
    assert stats["scored"] == 1
    assert stats["unavailable"] == 1
    assert stats["mean_probability"] == supported["probability"]
    assert sum(stats["bands"].values()) == 1


def test_all_unsupported_aggregate_has_no_mean(app, registered_client):
    registered_client.post(
        "/api/assess", json=UNSUPPORTED, headers={"X-CSRF-Token": registered_client.csrf}
    )
    stats = app.extensions["application_repository"].aggregate()
    assert stats["total"] == stats["unavailable"] == 1
    assert stats["mean_probability"] is None
    assert stats["bands"] == {}


@pytest.mark.parametrize("scenarios", [[], [UNSUPPORTED], [UNSUPPORTED, VALID]])
def test_admin_renders_empty_unavailable_and_mixed_portfolios(
    app, registered_client, scenarios
):
    for scenario in scenarios:
        assert (
            registered_client.post(
                "/api/assess",
                json=scenario,
                headers={"X-CSRF-Token": registered_client.csrf},
            ).status_code
            == 200
        )
    app.extensions["user_repository"].create(
        "portfolio_admin",
        "portfolio@example.com",
        "Portfolio Admin",
        hash_password("correct-admin-battery"),
        is_admin=True,
    )
    admin = app.test_client()
    assert (
        admin.post(
            "/login",
            json={"user_id": "portfolio_admin", "password": "correct-admin-battery"},
        ).status_code
        == 200
    )
    page = admin.get("/admin/")
    assert page.status_code == 200
    stats = admin.get("/admin/stats").get_json()["data"]
    assert stats["total"] == len(scenarios)
    if not scenarios or len(scenarios) == 1:
        assert stats["mean_probability"] is None
        assert b"No supported estimates recorded yet" in page.data
    else:
        assert stats["scored"] == stats["unavailable"] == 1
