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


class TestReviewRegressions:
    """Each of these reproduces a defect found in code review of this branch."""

    def test_non_ascii_csrf_token_is_rejected_not_a_500(self, registered_client):
        """``hmac.compare_digest`` raises TypeError on non-ASCII ``str``, so a
        one-character token turned a 400 into an unhandled 500 on a public
        endpoint."""
        response = registered_client.post(
            "/api/assess",
            json={
                "applicant_income": 50_000,
                "loan_amount": 1_000_000,
                "loan_term_months": 240,
                "credit_history": 1,
            },
            headers={"X-CSRF-Token": "é" * 20},
        )
        assert response.status_code == 400
        assert response.get_json()["error"] == "csrf_failed"

    def test_forwarded_for_uses_the_rightmost_hop(self, app):
        """A proxy *appends* the peer it saw, so the left-most entry is
        caller-controlled. Reading it let an attacker mint a fresh rate-limit
        bucket per request by varying the header."""
        from loanapp.security import client_identifier

        app.config["TRUSTED_PROXY"] = True
        with app.test_request_context(
            "/", headers={"X-Forwarded-For": "1.1.1.1, 2.2.2.2, 3.3.3.3"}
        ):
            assert client_identifier() == "3.3.3.3"

    def test_forwarded_for_is_ignored_without_a_trusted_proxy(self, app):
        from loanapp.security import client_identifier

        app.config["TRUSTED_PROXY"] = False
        with app.test_request_context("/", headers={"X-Forwarded-For": "9.9.9.9"}):
            assert client_identifier() != "9.9.9.9"

    def test_spoofed_forwarded_for_cannot_bypass_the_rate_limit(self, app):
        """The whole point of the previous two tests, end to end."""
        app.config["TRUSTED_PROXY"] = True
        app.config["RATE_LIMITS"]["predict"] = (2, 60)
        client = app.test_client()
        payload = {
            "applicant_income": 50_000,
            "loan_amount": 1_000_000,
            "loan_term_months": 240,
            "credit_history": 1,
        }
        codes = [
            client.post(
                "/api/assess",
                json=payload,
                headers={"X-Forwarded-For": f"10.0.0.{i}, 2.2.2.2"},
            ).status_code
            for i in range(4)
        ]
        # All four vary the left-most hop; the real hop is constant, so the
        # limit must still bite.
        assert 429 in codes, codes

    def test_rate_limiter_does_not_grow_without_bound(self):
        """``_hits`` was an attacker-controlled unbounded allocation."""
        from loanapp.security import RateLimiter

        limiter = RateLimiter()
        limiter.MAX_KEYS = 50
        for i in range(500):
            limiter.check(f"key-{i}", limit=10, window_seconds=60)
        assert len(limiter._hits) <= 50

    def test_invalid_samesite_is_rejected_at_startup(self, monkeypatch):
        """Werkzeug raises while *setting* the cookie, so a typo here breaks
        every response rather than one setting."""
        from loanapp.config import ConfigError, load_config

        monkeypatch.setenv("SECRET_KEY", "k" * 40)
        monkeypatch.setenv("SESSION_COOKIE_SAMESITE", "Lx")
        with pytest.raises(ConfigError, match="SAMESITE"):
            load_config("development")

    def test_samesite_none_requires_secure(self, monkeypatch):
        """Browsers silently drop SameSite=None without Secure."""
        from loanapp.config import ConfigError, load_config

        monkeypatch.setenv("SECRET_KEY", "k" * 40)
        monkeypatch.setenv("SESSION_COOKIE_SAMESITE", "None")
        monkeypatch.setenv("SESSION_COOKIE_SECURE", "false")
        with pytest.raises(ConfigError, match="SESSION_COOKIE_SECURE"):
            load_config("development")

    def test_samesite_is_normalised(self, monkeypatch):
        from loanapp.config import load_config

        monkeypatch.setenv("SECRET_KEY", "k" * 40)
        monkeypatch.setenv("SESSION_COOKIE_SAMESITE", "strict")
        assert load_config("development").cookie_samesite == "Strict"


class TestAuthenticatedUser:
    def test_raises_a_clear_error_when_no_one_is_signed_in(self, app):
        """A view that forgets @login_required should fail at the access check,
        not as `'NoneType' object is not subscriptable` somewhere downstream."""
        from loanapp.security import authenticated_user

        with (
            app.test_request_context("/"),
            pytest.raises(RuntimeError, match="login_required"),
        ):
            authenticated_user()

    def test_returns_the_user_when_signed_in(self, registered_client, app):
        response = registered_client.get("/api/applications")
        assert response.status_code == 200  # the view resolved the user


class TestRateLimitConfig:
    """Limits were hardcoded. Behind a corporate NAT an entire office shares one
    IP, so 30 assessments an hour locked everyone out with no operator lever."""

    def test_defaults_are_unchanged(self, monkeypatch):
        from loanapp.config import load_config

        monkeypatch.setenv("SECRET_KEY", "k" * 40)
        for name in ("RATE_LIMIT_LOGIN", "RATE_LIMIT_REGISTER", "RATE_LIMIT_PREDICT"):
            monkeypatch.delenv(name, raising=False)
        config = load_config("development")
        assert config.login_rate_limit == (10, 300)
        assert config.register_rate_limit == (5, 3600)
        assert config.predict_rate_limit == (30, 3600)

    def test_each_limit_is_configurable(self, monkeypatch):
        from loanapp.config import load_config

        monkeypatch.setenv("SECRET_KEY", "k" * 40)
        monkeypatch.setenv("RATE_LIMIT_PREDICT", "500/60")
        monkeypatch.setenv("RATE_LIMIT_LOGIN", "20/600")
        config = load_config("development")
        assert config.predict_rate_limit == (500, 60)
        assert config.login_rate_limit == (20, 600)

    @pytest.mark.parametrize(
        "value", ["30", "30/", "/60", "abc/60", "30/xyz", "0/60", "30/0", "-1/60"]
    )
    def test_malformed_limits_fail_at_startup(self, monkeypatch, value):
        """A typo must not silently disable rate limiting."""
        from loanapp.config import ConfigError, load_config

        monkeypatch.setenv("SECRET_KEY", "k" * 40)
        monkeypatch.setenv("RATE_LIMIT_PREDICT", value)
        with pytest.raises(ConfigError, match="RATE_LIMIT_PREDICT"):
            load_config("development")


class TestProxyAndBooleanConfig:
    """``TRUSTED_PROXY`` was read with ``bool(os.environ.get(...))``, so
    ``TRUSTED_PROXY=false`` or ``=0`` *enabled* trust in X-Forwarded-For — the
    spoofable header the limiter otherwise ignores. ``TRUSTED_PROXY_DEPTH`` was
    documented but never read from the environment at all."""

    def _app(self, tmp_path, monkeypatch, **env):
        from loanapp import create_app
        from loanapp.config import load_config

        monkeypatch.setenv("SECRET_KEY", "k" * 40)
        monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "t.db"))
        monkeypatch.setenv("MODEL_DIR", str(tmp_path))
        for name in ("TRUSTED_PROXY", "TRUSTED_PROXY_DEPTH"):
            monkeypatch.delenv(name, raising=False)
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return create_app(load_config("testing"))

    @pytest.mark.parametrize("value", ["false", "False", "0", "no", "off", ""])
    def test_a_false_value_does_not_trust_forwarded_for(
        self, tmp_path, monkeypatch, value
    ):
        from loanapp.security import client_identifier

        app = self._app(tmp_path, monkeypatch, TRUSTED_PROXY=value)
        assert app.config["TRUSTED_PROXY"] is False
        with app.test_request_context(
            "/",
            headers={"X-Forwarded-For": "9.9.9.9"},
            environ_base={"REMOTE_ADDR": "1.2.3.4"},
        ):
            assert client_identifier() == "1.2.3.4"

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_a_true_value_trusts_forwarded_for(self, tmp_path, monkeypatch, value):
        app = self._app(tmp_path, monkeypatch, TRUSTED_PROXY=value)
        assert app.config["TRUSTED_PROXY"] is True

    def test_unset_means_untrusted(self, tmp_path, monkeypatch):
        assert self._app(tmp_path, monkeypatch).config["TRUSTED_PROXY"] is False

    def test_depth_is_read_from_the_environment(self, tmp_path, monkeypatch):
        from loanapp.security import client_identifier

        app = self._app(tmp_path, monkeypatch, TRUSTED_PROXY="1", TRUSTED_PROXY_DEPTH="2")
        with app.test_request_context(
            "/", headers={"X-Forwarded-For": "6.6.6.6, 7.7.7.7, 8.8.8.8"}
        ):
            assert client_identifier() == "7.7.7.7"

    @pytest.mark.parametrize("value", ["0", "-1", "two"])
    def test_invalid_depth_fails_at_startup(self, monkeypatch, value):
        from loanapp.config import ConfigError, load_config

        monkeypatch.setenv("SECRET_KEY", "k" * 40)
        monkeypatch.setenv("TRUSTED_PROXY_DEPTH", value)
        with pytest.raises(ConfigError, match="TRUSTED_PROXY_DEPTH"):
            load_config("development")

    @pytest.mark.parametrize("name", ["TRUSTED_PROXY", "SESSION_COOKIE_SECURE"])
    @pytest.mark.parametrize("value", ["ture", "flase", "y", "enabled"])
    def test_an_unrecognised_boolean_fails_at_startup(self, monkeypatch, name, value):
        """Unrecognised used to mean False. For SESSION_COOKIE_SECURE that is
        fail-open: a typo of "true" sent the session cookie over plain HTTP."""
        from loanapp.config import ConfigError, load_config

        monkeypatch.setenv("SECRET_KEY", "k" * 40)
        monkeypatch.setenv(name, value)
        with pytest.raises(ConfigError, match=name):
            load_config("development")
