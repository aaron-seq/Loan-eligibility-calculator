"""Registration, login, sessions and admin access control."""

import pytest


def test_register_then_login(client, user_credentials):
    assert client.post("/api/register", json=user_credentials).status_code == 200

    response = client.post(
        "/api/login",
        json={
            "user_id": user_credentials["user_id"],
            "password": user_credentials["password"],
        },
    )
    assert response.status_code == 200
    assert response.get_json()["success"] is True


def test_password_is_hashed_at_rest(client, flask_app, user_credentials):
    client.post("/api/register", json=user_credentials)

    with flask_app.get_database_connection() as conn:
        row = conn.execute(
            "SELECT password_hash FROM users WHERE user_id = ?",
            (user_credentials["user_id"],),
        ).fetchone()

    assert user_credentials["password"] not in row["password_hash"]
    assert len(row["password_hash"]) > 40


def test_duplicate_registration_conflicts(client, user_credentials):
    client.post("/api/register", json=user_credentials)
    response = client.post("/api/register", json=user_credentials)

    assert response.status_code == 409


@pytest.mark.parametrize("user_id", ["ab", "has spaces", "sym!bol", "a" * 51])
def test_invalid_usernames_are_rejected(client, user_credentials, user_id):
    response = client.post(
        "/api/register", json={**user_credentials, "user_id": user_id}
    )
    assert response.status_code == 400


def test_short_password_is_rejected(client, user_credentials):
    response = client.post(
        "/api/register",
        json={**user_credentials, "password": "short", "confirm_password": "short"},
    )

    assert response.status_code == 400
    assert "8 characters" in response.get_json()["message"]


def test_mismatched_confirmation_is_rejected(client, user_credentials):
    response = client.post(
        "/api/register",
        json={**user_credentials, "confirm_password": "DifferentPass1"},
    )

    assert response.status_code == 400
    assert "match" in response.get_json()["message"].lower()


def test_usernames_are_case_insensitive(client, user_credentials):
    """Registration lowercases the username, so login must too, or the
    account becomes unreachable from the signup form."""
    client.post("/api/register", json={**user_credentials, "user_id": "AdaLovelace"})

    response = client.post(
        "/api/login",
        json={"user_id": "ADALOVELACE", "password": user_credentials["password"]},
    )
    assert response.status_code == 200


def test_wrong_password_is_unauthorised(client, user_credentials):
    client.post("/api/register", json=user_credentials)

    response = client.post(
        "/api/login",
        json={"user_id": user_credentials["user_id"], "password": "WrongPass1"},
    )
    assert response.status_code == 401


def test_unknown_user_is_unauthorised(client):
    response = client.post(
        "/api/login", json={"user_id": "nobody_here", "password": "Whatever1"}
    )
    assert response.status_code == 401


def test_login_without_credentials_is_a_bad_request(client):
    assert client.post("/api/login", json={}).status_code == 400


def test_dashboard_requires_login(client):
    assert client.get("/dashboard").status_code == 302


def test_logout_ends_the_session(auth_client):
    assert auth_client.get("/dashboard").status_code == 200
    auth_client.get("/logout")
    assert auth_client.get("/dashboard").status_code == 302


def test_admin_login_without_credentials_is_a_bad_request(client):
    """Missing fields previously reached check_password_hash(hash, None) and
    surfaced as a 500 instead of a 400."""
    response = client.post("/api/admin/login", json={})

    assert response.status_code == 400
    assert response.get_json()["success"] is False


def test_admin_login_succeeds_with_configured_password(client):
    response = client.post(
        "/api/admin/login",
        json={"admin_id": "admin", "password": "test-admin-password"},
    )
    assert response.status_code == 200


def test_admin_login_rejects_the_old_hardcoded_password(client):
    """ADMIN_PASSWORD is set for these tests, so the former built-in default
    must not work."""
    response = client.post(
        "/api/admin/login", json={"admin_id": "admin", "password": "admin123"}
    )
    assert response.status_code == 401


def test_admin_dashboard_requires_admin_session(client):
    assert client.get("/admin/dashboard").status_code == 302


def test_user_session_cannot_reach_admin_dashboard(auth_client):
    """A logged-in ordinary user must not cross into the admin area."""
    assert auth_client.get("/admin/dashboard").status_code == 302


def test_admin_login_clears_any_prior_user_session(auth_client):
    auth_client.post(
        "/api/admin/login",
        json={"admin_id": "admin", "password": "test-admin-password"},
    )

    assert auth_client.get("/admin/dashboard").status_code == 200
    # The user half of the session must not survive the privilege change.
    assert auth_client.get("/dashboard").status_code == 302
