"""CLI-created accounts and login return-path handling."""

from __future__ import annotations

import pytest

from loanapp.security import hash_password, verify_password

ADMIN = {
    "user_id": "adminuser",
    "email": "admin@example.com",
    "full_name": "Example Administrator",
    "password": "disposable-admin-password",
}


def create_admin(app, **overrides):
    values = {**ADMIN, **overrides}
    arguments = ["create-admin"]
    for name, value in values.items():
        arguments.extend(["--" + name.replace("_", "-"), value])
    return app.test_cli_runner().invoke(args=arguments)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"user_id": " AdminUser "},
        {"password": " disposable-admin-password "},
        {"email": " Admin@Example.Com ", "full_name": " Example Administrator "},
    ],
)
def test_cli_creation_stores_normalized_values_and_allows_login(modelless_app, overrides):
    result = create_admin(modelless_app, **overrides)
    assert result.exit_code == 0, result.output

    user = modelless_app.extensions["user_repository"].find(ADMIN["user_id"])
    assert user is not None
    assert user["email"] == ADMIN["email"]
    assert user["full_name"] == ADMIN["full_name"]
    assert verify_password(user["password_hash"], ADMIN["password"])
    assert user["is_admin"] == 1

    client = modelless_app.test_client()
    response = client.post(
        "/login", json={"user_id": ADMIN["user_id"], "password": ADMIN["password"]}
    )
    assert response.status_code == 200
    assert response.get_json()["success"] is True
    assert client.get("/admin/").status_code == 200


@pytest.mark.parametrize(
    "overrides",
    [
        {"user_id": " AdminUser ", "email": "other@example.com"},
        {"user_id": "otheruser", "email": " Admin@Example.Com "},
    ],
)
def test_cli_creation_rejects_normalized_duplicate_accounts(modelless_app, overrides):
    assert create_admin(modelless_app).exit_code == 0
    result = create_admin(modelless_app, **overrides)
    assert result.exit_code == 1
    assert "already exists" in result.output
    assert modelless_app.extensions["user_repository"].count() == 1


@pytest.fixture
def login_client(modelless_app):
    modelless_app.extensions["user_repository"].create(
        ADMIN["user_id"],
        ADMIN["email"],
        ADMIN["full_name"],
        hash_password(ADMIN["password"]),
    )
    return modelless_app.test_client()


@pytest.mark.parametrize("next_value", [123, True, ["/apply"], {"path": "/apply"}])
def test_login_ignores_non_string_return_path(login_client, next_value):
    response = login_client.post(
        "/login",
        json={
            "user_id": ADMIN["user_id"],
            "password": ADMIN["password"],
            "next": next_value,
        },
    )
    assert response.status_code == 200
    assert response.get_json()["redirect_to"] == "/dashboard"
    with login_client.session_transaction() as session:
        assert session["user_id"] == ADMIN["user_id"]
    assert login_client.get("/dashboard").status_code == 200


@pytest.mark.parametrize(
    ("fields", "destination"),
    [
        ({}, "/dashboard"),
        ({"next": None}, "/dashboard"),
        ({"next": "/apply"}, "/apply"),
        ({"next": "//example.com"}, "/dashboard"),
    ],
)
def test_login_preserves_optional_and_same_site_return_paths(
    login_client, fields, destination
):
    response = login_client.post(
        "/login",
        json={
            "user_id": ADMIN["user_id"],
            "password": ADMIN["password"],
            **fields,
        },
    )
    assert response.status_code == 200
    assert response.get_json()["redirect_to"] == destination
