"""Small negative cases at JSON, cookie and response privacy boundaries."""

import pytest


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/api/assess", [1]),
        ("/api/compare", "text"),
        ("/register", 7),
        ("/login", True),
    ],
)
def test_non_object_json_is_rejected_without_side_effects(app, client, path, payload):
    response = client.post(path, json=payload)
    assert response.status_code in (400, 401)
    assert response.is_json
    assert app.extensions["user_repository"].count() == 0
    assert app.extensions["application_repository"].aggregate()["total"] == 0


def test_non_object_json_cannot_crash_csrf_or_write_history(app, registered_client):
    response = registered_client.post("/api/assess", json=[1])
    assert response.status_code == 400
    assert response.get_json()["error"] == "csrf_failed"
    assert app.extensions["application_repository"].aggregate()["total"] == 0
    assert registered_client.get("/dashboard").status_code == 200


def test_invalid_body_with_valid_csrf_is_still_rejected(app, registered_client):
    response = registered_client.post(
        "/api/assess", json=[1], headers={"X-CSRF-Token": registered_client.csrf}
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "validation_failed"
    assert app.extensions["application_repository"].aggregate()["total"] == 0


@pytest.mark.parametrize(
    "path", ["/dashboard", "/api/applications", "/api/csrf-token", "/admin/stats"]
)
def test_dynamic_account_responses_are_not_cacheable(registered_client, path):
    response = registered_client.get(path)
    assert "no-store" in response.headers.get("Cache-Control", "")


def test_static_css_keeps_its_independent_cache_policy(client):
    response = client.get("/static/css/app.css")
    assert response.status_code == 200
    assert "no-store" not in response.headers.get("Cache-Control", "")
