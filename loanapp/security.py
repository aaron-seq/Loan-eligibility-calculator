"""Security primitives: CSRF, rate limiting, auth decorators.

Both CSRF and rate limiting are implemented here rather than pulled in as
dependencies. They are small enough to read in one sitting and are covered by
tests, which is worth more for a project of this size than an extra transitive
dependency tree. Their limits are documented honestly below.
"""

from __future__ import annotations

import hmac
import secrets
import time
from collections import defaultdict, deque
from functools import wraps
from typing import Callable

from flask import current_app, g, jsonify, redirect, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

CSRF_SESSION_KEY = "_csrf_token"
CSRF_FORM_FIELD = "csrf_token"
CSRF_HEADER = "X-CSRF-Token"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})


# --------------------------------------------------------------------------
# Passwords
# --------------------------------------------------------------------------
def hash_password(password: str) -> str:
    """Hash with Werkzeug's default (scrypt on modern versions)."""
    return generate_password_hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    return check_password_hash(password_hash, password)


#: Verified against this when the user does not exist, so a failed login costs
#: the same time whether or not the username is real. Without it, response
#: timing enumerates valid usernames.
_DUMMY_HASH = generate_password_hash("timing-equalisation-placeholder")


def verify_user_password(user: dict | None, password: str) -> bool:
    if user is None:
        check_password_hash(_DUMMY_HASH, password)
        return False
    return verify_password(user["password_hash"], password)


# --------------------------------------------------------------------------
# CSRF
# --------------------------------------------------------------------------
def _carries_session_cookie() -> bool:
    return current_app.config["SESSION_COOKIE_NAME"] in request.cookies


def csrf_token() -> str:
    """Return the session's CSRF token, minting one if needed."""
    token = session.get(CSRF_SESSION_KEY)
    if not token:
        token = secrets.token_urlsafe(32)
        session[CSRF_SESSION_KEY] = token
    return token


def validate_csrf() -> bool:
    expected = session.get(CSRF_SESSION_KEY)
    if not expected:
        return False
    supplied = request.headers.get(CSRF_HEADER) or request.form.get(CSRF_FORM_FIELD)
    if not supplied and request.is_json:
        supplied = (request.get_json(silent=True) or {}).get(CSRF_FORM_FIELD)
    return bool(supplied) and hmac.compare_digest(str(supplied), str(expected))


def install_csrf(app) -> None:
    """Reject unsafe requests without a valid token, and expose it to Jinja.

    Applied to every state-changing request rather than per-route, so a new
    endpoint is protected by default instead of by remembering a decorator.
    """

    @app.before_request
    def _check_csrf():
        if request.method in SAFE_METHODS:
            return None
        if getattr(
            current_app.view_functions.get(request.endpoint), "_csrf_exempt", False
        ):
            return None
        if not _carries_session_cookie():
            # CSRF is an attack on ambient credentials. A request with no
            # session cookie has none to abuse, so a plain `curl` against the
            # anonymous endpoints works without a token dance. Anything that
            # acts on behalf of a signed-in user carries the cookie and is
            # still checked below.
            return None
        if validate_csrf():
            return None
        if wants_json():
            return jsonify(
                {
                    "error": "csrf_failed",
                    "message": "Missing or invalid CSRF token. Fetch one from "
                    "GET /api/csrf-token and send it in the X-CSRF-Token header.",
                }
            ), 400
        return redirect(url_for("pages.home", error="session_expired"))

    app.jinja_env.globals["csrf_token"] = csrf_token


def csrf_exempt(view: Callable) -> Callable:
    """Opt a view out of CSRF checking. Use sparingly and say why."""
    view._csrf_exempt = True
    return view


# --------------------------------------------------------------------------
# Rate limiting
# --------------------------------------------------------------------------
class RateLimiter:
    """Fixed-window counter, in process memory.

    Limitations, stated plainly: counters are per-process, so N gunicorn
    workers allow roughly N times the configured limit, and everything resets
    on restart. That is adequate protection against casual credential stuffing
    and accidental hammering, and it is not a substitute for a shared store or
    an edge WAF. Moving to Redis is a roadmap item; the interface here is the
    one a Redis backend would implement.
    """

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        """Return ``(allowed, retry_after_seconds)``."""
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] > window_seconds:
            hits.popleft()
        if len(hits) >= limit:
            return False, int(window_seconds - (now - hits[0])) + 1
        hits.append(now)
        return True, 0

    def reset(self) -> None:
        self._hits.clear()


limiter = RateLimiter()


def client_identifier() -> str:
    """Best-effort client identity.

    ``X-Forwarded-For`` is only trusted when ``TRUSTED_PROXY`` is set, because
    a spoofable header would otherwise let a caller mint a fresh rate-limit
    bucket per request.
    """
    if current_app.config.get("TRUSTED_PROXY"):
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def rate_limit(limit_name: str) -> Callable:
    """Rate limit a view using the named limit from config."""

    def decorator(view: Callable) -> Callable:
        @wraps(view)
        def wrapper(*args, **kwargs):
            limit, window = current_app.config["RATE_LIMITS"][limit_name]
            key = f"{limit_name}:{client_identifier()}"
            allowed, retry_after = limiter.check(key, limit, window)
            if allowed:
                return view(*args, **kwargs)

            current_app.logger.warning(
                "rate limit hit: limit=%s client=%s", limit_name, key
            )
            message = f"Too many requests. Try again in {retry_after} seconds."
            if wants_json():
                response = jsonify({"error": "rate_limited", "message": message})
                response.status_code = 429
                response.headers["Retry-After"] = str(retry_after)
                return response
            return redirect(url_for("pages.home", error="rate_limited"))

        return wrapper

    return decorator


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
def json_endpoint(view: Callable) -> Callable:
    """Mark a view outside ``/api/`` as returning JSON.

    Without this, path-sniffing sends a machine-readable endpoint such as
    ``/admin/stats`` an HTML redirect when the caller is unauthorised, which is
    useless to the scraper or dashboard consuming it.
    """
    view._returns_json = True
    return view


def wants_json() -> bool:
    if request.path.startswith("/api/") or request.is_json:
        return True
    view = current_app.view_functions.get(request.endpoint)
    return bool(getattr(view, "_returns_json", False))


def current_user() -> dict | None:
    """The logged-in user, loaded once per request."""
    if "current_user" not in g:
        user_id = session.get("user_id")
        g.current_user = (
            current_app.extensions["user_repository"].find(user_id) if user_id else None
        )
    return g.current_user


def login_required(view: Callable) -> Callable:
    @wraps(view)
    def wrapper(*args, **kwargs):
        if current_user() is None:
            # Clear a stale session so a deleted or deactivated account cannot
            # keep presenting a cookie that looks logged in.
            session.clear()
            if wants_json():
                return jsonify(
                    {"error": "unauthorised", "message": "Sign in to continue."}
                ), 401
            return redirect(url_for("pages.login", next=request.path))
        return view(*args, **kwargs)

    return wrapper


def admin_required(view: Callable) -> Callable:
    @wraps(view)
    def wrapper(*args, **kwargs):
        user = current_user()
        if user is None:
            if wants_json():
                return jsonify(
                    {"error": "unauthorised", "message": "Sign in to continue."}
                ), 401
            return redirect(url_for("pages.login", next=request.path))
        if not user.get("is_admin"):
            current_app.logger.warning(
                "admin access denied for user_id=%s", user["user_id"]
            )
            if wants_json():
                return jsonify(
                    {"error": "forbidden", "message": "Administrator access required."}
                ), 403
            return redirect(url_for("pages.dashboard"))
        return view(*args, **kwargs)

    return wrapper


def start_session(user: dict) -> None:
    """Log a user in, rotating the session to prevent fixation."""
    session.clear()
    session["user_id"] = user["user_id"]
    session["full_name"] = user["full_name"]
    session.permanent = True
    csrf_token()  # mint a fresh token bound to the new session
