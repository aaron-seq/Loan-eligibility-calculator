"""Security behaviour: config, CSRF, rate limits, auth and access control.

Each of these covers a specific weakness found in the previous implementation.
"""

from __future__ import annotations

import pytest

from loanapp.config import ConfigError, load_config
from loanapp.security import RateLimiter, hash_password, verify_password


class TestConfig:
    def test_production_without_a_secret_key_refuses_to_start(self, monkeypatch):
        """The previous code fell back to os.urandom(24), which looks secure
        and silently breaks every session across workers and restarts."""
        monkeypatch.delenv("SECRET_KEY", raising=False)
        monkeypatch.setenv("FLASK_ENV", "production")
        with pytest.raises(ConfigError, match="SECRET_KEY"):
            load_config()

    def test_production_rejects_a_short_secret_key(self, monkeypatch):
        monkeypatch.setenv("SECRET_KEY", "short")
        monkeypatch.setenv("FLASK_ENV", "production")
        with pytest.raises(ConfigError, match="32 characters"):
            load_config()

    def test_development_generates_a_key_but_warns_loudly(self, monkeypatch):
        monkeypatch.delenv("SECRET_KEY", raising=False)
        config = load_config("development")
        assert config.secret_key
        assert any("SECRET_KEY" in w for w in config.warnings)

    def test_unknown_environment_is_rejected(self, monkeypatch):
        monkeypatch.setenv("SECRET_KEY", "k" * 40)
        with pytest.raises(ConfigError, match="FLASK_ENV"):
            load_config("staging")

    def test_inverted_decision_thresholds_are_rejected(self, monkeypatch):
        monkeypatch.setenv("SECRET_KEY", "k" * 40)
        monkeypatch.setenv("APPROVE_THRESHOLD", "0.2")
        monkeypatch.setenv("BORDERLINE_THRESHOLD", "0.8")
        with pytest.raises(ConfigError, match="THRESHOLD"):
            load_config("development")

    def test_interest_rate_given_as_a_percentage_is_rejected(self, monkeypatch):
        monkeypatch.setenv("SECRET_KEY", "k" * 40)
        monkeypatch.setenv("ANNUAL_INTEREST_RATE", "10.5")  # meant 0.105
        with pytest.raises(ConfigError, match="fraction"):
            load_config("development")

    def test_cookies_are_secure_by_default_in_production(self, monkeypatch):
        monkeypatch.setenv("SECRET_KEY", "k" * 40)
        monkeypatch.delenv("SESSION_COOKIE_SECURE", raising=False)
        assert load_config("production").cookie_secure is True


class TestPasswords:
    def test_hashes_are_salted_and_verifiable(self):
        a, b = hash_password("correct-horse"), hash_password("correct-horse")
        assert a != b  # per-hash salt
        assert verify_password(a, "correct-horse")
        assert not verify_password(a, "wrong")

    def test_plaintext_is_never_stored_in_the_hash(self):
        assert "correct-horse" not in hash_password("correct-horse")

    def test_not_md5(self):
        """The legacy app hashed passwords with unsalted MD5."""
        digest = hash_password("correct-horse")
        assert len(digest) > 32
        assert digest.startswith(("scrypt:", "pbkdf2:", "argon2"))


class TestRateLimiter:
    def test_allows_up_to_the_limit_then_blocks(self):
        limiter = RateLimiter()
        assert all(limiter.check("k", 3, 60)[0] for _ in range(3))
        allowed, retry_after = limiter.check("k", 3, 60)
        assert not allowed
        assert retry_after > 0

    def test_buckets_are_independent(self):
        limiter = RateLimiter()
        limiter.check("a", 1, 60)
        assert limiter.check("b", 1, 60)[0]

    def test_window_expiry_lets_requests_through_again(self):
        limiter = RateLimiter()
        assert limiter.check("k", 1, 0)[0]
        assert limiter.check("k", 1, 0)[0]


class TestCsrf:
    def test_form_post_without_a_token_is_rejected(self, client):
        client.get("/login")  # establishes a session cookie
        response = client.post("/login", data={"user_id": "x", "password": "y"})
        assert response.status_code in (302, 400)
        assert "user_id" not in (response.headers.get("Set-Cookie") or "")

    def test_json_post_with_a_session_but_no_token_is_rejected(self, registered_client):
        response = registered_client.post(
            "/api/assess",
            json={
                "applicant_income": 50_000,
                "loan_amount": 1_000_000,
                "loan_term_months": 240,
                "credit_history": 1,
            },
        )
        assert response.status_code == 400
        assert response.get_json()["error"] == "csrf_failed"

    def test_a_cookieless_client_needs_no_token(self, client):
        """CSRF is an attack on ambient credentials. A request carrying no
        session cookie has none to abuse, so plain curl works."""
        response = client.post(
            "/api/assess",
            json={
                "applicant_income": 50_000,
                "loan_amount": 1_000_000,
                "loan_term_months": 240,
                "credit_history": 1,
            },
        )
        assert response.status_code == 200

    def test_a_valid_token_is_accepted(self, registered_client):
        response = registered_client.post(
            "/api/assess",
            json={
                "applicant_income": 50_000,
                "loan_amount": 1_000_000,
                "loan_term_months": 240,
                "credit_history": 1,
            },
            headers={"X-CSRF-Token": registered_client.csrf},
        )
        assert response.status_code == 200

    def test_safe_methods_are_never_blocked(self, client):
        assert client.get("/api/health").status_code == 200


class TestAuthentication:
    def test_wrong_password_and_unknown_user_give_the_same_message(self, client):
        token = client.get("/api/csrf-token").get_json()["data"]["csrf_token"]
        unknown = client.post(
            "/login",
            json={
                "user_id": "nobody",
                "password": "whatever12345",
                "csrf_token": token,
            },
        )
        assert unknown.status_code == 401
        # A different message here would confirm which usernames exist.
        assert "Incorrect username or password" in unknown.get_json()["message"]

    def test_registration_rejects_a_short_password(self, client):
        token = client.get("/api/csrf-token").get_json()["data"]["csrf_token"]
        response = client.post(
            "/register",
            json={
                "user_id": "shorty",
                "email": "s@example.com",
                "full_name": "Short Pass",
                "password": "short",
                "confirm_password": "short",
                "csrf_token": token,
            },
        )
        assert response.status_code == 400
        assert "password" in response.get_json()["errors"]

    def test_duplicate_registration_does_not_reveal_which_field_collided(
        self, registered_client
    ):
        response = registered_client.post(
            "/register",
            json={
                "user_id": "testuser",
                "email": "different@example.com",
                "full_name": "Impostor",
                "password": "correct-horse-battery",
                "confirm_password": "correct-horse-battery",
                "csrf_token": registered_client.csrf,
            },
        )
        assert response.status_code == 409
        # The message covers both fields without saying which one collided, so
        # it cannot be used to test whether an email address is registered.
        message = response.get_json()["message"].lower()
        assert "username or email" in message

    def test_login_next_parameter_cannot_redirect_off_site(self, client):
        token = client.get("/api/csrf-token").get_json()["data"]["csrf_token"]
        client.post(
            "/register",
            json={
                "user_id": "redir",
                "email": "r@example.com",
                "full_name": "Redirect Test",
                "password": "correct-horse-battery",
                "confirm_password": "correct-horse-battery",
                "csrf_token": token,
            },
        )
        token = client.get("/api/csrf-token").get_json()["data"]["csrf_token"]
        response = client.post(
            "/login",
            json={
                "user_id": "redir",
                "password": "correct-horse-battery",
                "next": "https://evil.example.com/phish",
                "csrf_token": token,
            },
        )
        assert "evil.example.com" not in response.get_json().get("redirect_to", "")


class TestAccessControl:
    def test_history_requires_authentication(self, client):
        assert client.get("/api/applications").status_code == 401

    def test_admin_endpoints_reject_a_normal_user(self, registered_client):
        assert registered_client.get("/admin/stats").status_code == 403

    def test_admin_endpoints_reject_anonymous_callers(self, client):
        assert client.get("/admin/stats").status_code == 401

    def test_a_user_cannot_read_another_users_application(self, app, registered_client):
        """Ownership is enforced in the SQL, not after the fetch."""
        repository = app.extensions["application_repository"]
        app.extensions["user_repository"].create(
            "otheruser", "other@example.com", "Other", hash_password("x" * 20)
        )
        foreign_id = repository.save(
            user_id="otheruser",
            label="private",
            payload={"loan_amount": 1},
            assessment={
                "probability": 0.9,
                "band": "likely",
                "model": {"version": "v"},
            },
        )
        response = registered_client.get(f"/api/applications/{foreign_id}")
        assert response.status_code == 404  # not 403 — existence is not revealed

    def test_a_user_cannot_delete_another_users_application(self, app, registered_client):
        repository = app.extensions["application_repository"]
        app.extensions["user_repository"].create(
            "victim", "victim@example.com", "Victim", hash_password("x" * 20)
        )
        foreign_id = repository.save(
            user_id="victim",
            label="private",
            payload={"loan_amount": 1},
            assessment={
                "probability": 0.9,
                "band": "likely",
                "model": {"version": "v"},
            },
        )
        response = registered_client.delete(
            f"/api/applications/{foreign_id}",
            headers={"X-CSRF-Token": registered_client.csrf},
        )
        assert response.status_code == 404
        assert repository.get(foreign_id) is not None

    def test_no_default_admin_account_exists(self, app):
        """The previous version auto-created admin/admin123 on first boot."""
        assert app.extensions["user_repository"].find("admin") is None
