"""Controlled users prove both allowed paths and absence of forbidden effects."""

import pytest
from test_api import VALID

from loanapp.db import connect


def register(app, user_id, **extra):
    client = app.test_client()
    token = client.get("/api/csrf-token").get_json()["data"]["csrf_token"]
    response = client.post(
        "/register",
        json={
            "user_id": user_id,
            "email": f"{user_id}@example.com",
            "full_name": "Test Actor",
            "password": "correct-horse-battery",
            "confirm_password": "correct-horse-battery",
            "csrf_token": token,
            **extra,
        },
    )
    assert response.status_code == 201
    client.csrf = client.get("/api/csrf-token").get_json()["data"]["csrf_token"]
    return client


@pytest.fixture
def actors(app):
    owner = register(app, "owner")
    other = register(app, "other")
    result = app.test_cli_runner().invoke(
        args=[
            "create-admin",
            "--user-id",
            "audit_admin",
            "--email",
            "admin@example.com",
            "--password",
            "correct-admin-battery",
        ]
    )
    assert result.exit_code == 0, result.output
    admin = app.test_client()
    assert (
        admin.post(
            "/login", json={"user_id": "audit_admin", "password": "correct-admin-battery"}
        ).status_code
        == 200
    )
    return owner, other, admin, app.test_client()


def test_actor_action_resource_matrix(app, actors):
    owner, other, admin, anonymous = actors
    label = "Owner's <b>private scenario</b>"
    created = owner.post(
        "/api/assess",
        json={**VALID, "loan_amount": 1_432_100, "label": label},
        headers={"X-CSRF-Token": owner.csrf},
    ).get_json()["data"]
    path = f"/api/applications/{created['application_id']}"
    view = f"/applications/{created['application_id']}"
    assert owner.get(path).status_code == owner.get(view).status_code == 200
    assert b"&lt;b&gt;private scenario&lt;/b&gt;" in owner.get(view).data
    for denied in (other, admin):
        assert denied.get(path).status_code == denied.get(view).status_code == 404
        assert (
            b"1432100" not in denied.get(f"/apply?from={created['application_id']}").data
        )
    assert other.get("/api/applications").get_json()["data"]["count"] == 0
    denied_delete = other.delete(path, headers={"X-CSRF-Token": other.csrf})
    assert denied_delete.status_code == 404
    assert owner.get(path).get_json()["data"]["label"] == label
    assert anonymous.get(path).status_code == 401
    assert anonymous.get(view).status_code == 302
    assert other.get("/admin/stats").status_code == 403
    assert anonymous.get("/admin/stats").status_code == 401
    assert (
        admin.get("/admin/stats").status_code == admin.get("/admin/").status_code == 200
    )
    assert owner.delete(path).status_code == 400  # missing CSRF changes nothing
    assert owner.get(path).status_code == 200
    assert owner.delete(path, headers={"X-CSRF-Token": owner.csrf}).status_code == 200
    assert owner.get(path).status_code == 404


def test_registration_cannot_assign_admin_privileges(app):
    client = register(app, "overpost", is_admin=True)
    assert app.extensions["user_repository"].find("overpost")["is_admin"] == 0
    assert client.get("/admin/stats").status_code == 403


def test_disabled_account_loses_access_on_the_next_request(app):
    client = register(app, "disabled")
    assert client.get("/dashboard").status_code == 200
    with connect(app.config["APP_CONFIG"].database_path) as conn:
        conn.execute("UPDATE users SET is_active=0 WHERE user_id=?", ("disabled",))
    assert client.get("/api/applications").status_code == 401
    with client.session_transaction() as state:
        assert "user_id" not in state
