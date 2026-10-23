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
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar, cast

from flask import (
    Flask,
    Response,
    current_app,
    flash,
    g,
    jsonify,
    redirect,
    request,
    session,
    url_for,
)
from flask.typing import ResponseReturnValue
from werkzeug.security import check_password_hash, generate_password_hash

#: A user row as returned by UserRepository.
User = dict[str, Any]

#: Any view function. Decorators are typed to return the *same* type they
#: receive, so wrapping a view does not erase its signature.
ViewT = TypeVar("ViewT", bound=Callable[..., Any])

CSRF_SESSION_KEY = "_csrf_token"
CSRF_FORM_FIELD = "csrf_token"
CSRF_HEADER = "X-CSRF-Token"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})


def install_security_headers(app: Flask) -> None:
    """Apply the same policy to HTML, JSON, redirects, errors and static files."""

    @app.after_request
    def secure_response(response: Response) -> Response:
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; script-src 'none'; style-src 'self'; "
            "img-src 'self'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "same-origin"
        return response


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


def verify_user_password(user: User | None, password: str) -> bool:
    if user is None:
        check_password_hash(_DUMMY_HASH, password)
        return False
    return verify_password(str(user["password_hash"]), password)


# --------------------------------------------------------------------------
# CSRF
# --------------------------------------------------------------------------
def _current_view() -> Callable[..., Any] | None:
    """The view function handling this request, or None if no route matched.

    ``request.endpoint`` is None for a request that matched no route. Both the
    CSRF hook and ``wants_json`` used to look this up separately and pass that
    None straight into ``dict.get``; this is now the one place that handles it.
    """
    if request.endpoint is None:
        return None
    return current_app.view_functions.get(request.endpoint)


#: The page holding the form for each endpoint a browser posts to, for
#: endpoints other than the loan form.
_FORM_PAGES = {"auth.login": "pages.login", "auth.register": "pages.signup"}


def _back_to_form() -> ResponseReturnValue:
    """Send a browser back to the form whose post was just rejected.

    Rejected posts used to go to the loan application form whatever they
    were, so someone rate-limited while signing in landed on an unrelated
    page with their sign-in form gone. The flashed message renders on
    whichever page this lands on.
    """
    return redirect(url_for(_FORM_PAGES.get(request.endpoint or "", "pages.apply_form")))


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
    if not supplied:
        return False
    # Compare as bytes. hmac.compare_digest raises TypeError on a str
    # containing non-ASCII, so a token of "é" turned a 400 csrf_failed into an
    # unhandled 500 — a trivially reachable crash on a public endpoint.
    # errors="replace" keeps any input encodable without making two distinct
    # tokens collide, since a mismatch is the only outcome that matters here.
    try:
        supplied_bytes = str(supplied).encode("utf-8", errors="replace")
        expected_bytes = str(expected).encode("utf-8", errors="replace")
    except (UnicodeError, AttributeError):
        return False
    return hmac.compare_digest(supplied_bytes, expected_bytes)


def install_csrf(app: Flask) -> None:
    """Reject unsafe requests without a valid token, and expose it to Jinja.

    Applied to every state-changing request rather than per-route, so a new
    endpoint is protected by default instead of by remembering a decorator.
    """

    @app.before_request
    def _check_csrf() -> ResponseReturnValue | None:
        if request.method in SAFE_METHODS:
            return None
        if getattr(_current_view(), "_csrf_exempt", False):
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
        flash(
            "Your session expired before the form was submitted. "
            "Please fill it in again.",
            "error",
        )
        return _back_to_form()

    app.jinja_env.globals["csrf_token"] = csrf_token


def csrf_exempt(view: ViewT) -> ViewT:
    """Opt a view out of CSRF checking. Use sparingly and say why."""
    setattr(view, "_csrf_exempt", True)  # noqa: B010 - marker attribute
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

    #: Cap on distinct tracked keys. Without it the dict is an unbounded,
    #: attacker-controlled allocation: every new client IP (or every spoofed
    #: forwarded address) adds an entry that is never collected. At the limit
    #: we evict the stalest bucket, which can forgive a request but cannot
    #: exhaust memory.
    MAX_KEYS = 100_000

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, limit: int, window_seconds: int) -> tuple[bool, int]:
        """Return ``(allowed, retry_after_seconds)``."""
        now = time.monotonic()

        if key not in self._hits and len(self._hits) >= self.MAX_KEYS:
            self._evict_stalest(now, window_seconds)

        hits = self._hits[key]
        while hits and now - hits[0] > window_seconds:
            hits.popleft()
        if len(hits) >= limit:
            return False, int(window_seconds - (now - hits[0])) + 1
        hits.append(now)
        return True, 0

    def _evict_stalest(self, now: float, window_seconds: int) -> None:
        """Drop fully-expired buckets, or the oldest if none have expired."""
        expired = [
            k for k, v in self._hits.items() if not v or now - v[-1] > window_seconds
        ]
        if expired:
            for k in expired:
                del self._hits[k]
            return
        oldest = min(self._hits, key=lambda k: self._hits[k][-1])
        del self._hits[oldest]

    def reset(self) -> None:
        self._hits.clear()


limiter = RateLimiter()


def client_identifier() -> str:
    """Best-effort client identity for rate limiting.

    ``X-Forwarded-For`` is only consulted when ``TRUSTED_PROXY`` is set, and
    then the **right-most** hop is used, not the left-most.

    That direction matters. A standard proxy *appends* the peer it saw, so the
    list reads ``<client-supplied>, <real client>, <inner proxy>``. Taking
    ``[0]`` therefore reads a value the caller fully controls: with
    ``TRUSTED_PROXY=1`` an attacker could send a different
    ``X-Forwarded-For`` on every request and get a fresh rate-limit bucket each
    time, bypassing the limiter entirely. The last entry is the one the nearest
    trusted proxy wrote and is the furthest an unprivileged caller can reach.

    If your topology has more than one proxy in front of the app, the correct
    index is deeper; ``TRUSTED_PROXY_DEPTH`` sets how many hops to skip from
    the right.
    """
    if current_app.config.get("TRUSTED_PROXY"):
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            hops = [h.strip() for h in forwarded.split(",") if h.strip()]
            if hops:
                depth = max(1, int(current_app.config.get("TRUSTED_PROXY_DEPTH", 1)))
                return hops[-min(depth, len(hops))]
    return request.remote_addr or "unknown"


def rate_limit(limit_name: str) -> Callable[[ViewT], ViewT]:
    """Rate limit a view using the named limit from config."""

    def decorator(view: ViewT) -> ViewT:
        @wraps(view)
        def wrapper(*args: Any, **kwargs: Any) -> ResponseReturnValue:
            limit, window = current_app.config["RATE_LIMITS"][limit_name]
            key = f"{limit_name}:{client_identifier()}"
            allowed, retry_after = limiter.check(key, limit, window)
            if allowed:
                return cast(ResponseReturnValue, view(*args, **kwargs))

            current_app.logger.warning(
                "rate limit hit: limit=%s client=%s", limit_name, key
            )
            message = f"Too many requests. Try again in {retry_after} seconds."
            if wants_json():
                response = jsonify({"error": "rate_limited", "message": message})
                response.status_code = 429
                response.headers["Retry-After"] = str(retry_after)
                return response
            flash(message, "error")
            return _back_to_form()

        return cast(ViewT, wrapper)

    return decorator


# --------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------
def json_endpoint(view: ViewT) -> ViewT:
    """Mark a view outside ``/api/`` as returning JSON.

    Without this, path-sniffing sends a machine-readable endpoint such as
    ``/admin/stats`` an HTML redirect when the caller is unauthorised, which is
    useless to the scraper or dashboard consuming it.
    """
    setattr(view, "_returns_json", True)  # noqa: B010 - marker attribute
    return view


def wants_json() -> bool:
    if request.path.startswith("/api/") or request.is_json:
        return True
    return bool(getattr(_current_view(), "_returns_json", False))


def current_user() -> User | None:
    """The logged-in user, loaded once per request."""
    if "current_user" not in g:
        user_id = session.get("user_id")
        g.current_user = (
            current_app.extensions["user_repository"].find(user_id) if user_id else None
        )
    return cast(User | None, g.current_user)


def authenticated_user() -> User:
    """The logged-in user, for views behind ``@login_required``.

    Views used to index ``current_user()["user_id"]`` directly, relying on the
    decorator to have run. Forgetting the decorator then failed as
    ``TypeError: 'NoneType' object is not subscriptable`` deep inside the view,
    which reads like a data bug rather than a missing access check. This fails
    at the right place with the right message.
    """
    user = current_user()
    if user is None:
        raise RuntimeError(
            "authenticated_user() called without a signed-in user; "
            "is the view missing @login_required?"
        )
    return user


def _unauthorised() -> ResponseReturnValue:
    if wants_json():
        return jsonify({"error": "unauthorised", "message": "Sign in to continue."}), 401
    return redirect(url_for("pages.login", next=request.path))


def login_required(view: ViewT) -> ViewT:
    @wraps(view)
    def wrapper(*args: Any, **kwargs: Any) -> ResponseReturnValue:
        if current_user() is None:
            # Clear a stale session so a deleted or deactivated account cannot
            # keep presenting a cookie that looks logged in.
            session.clear()
            return _unauthorised()
        return cast(ResponseReturnValue, view(*args, **kwargs))

    return cast(ViewT, wrapper)


def admin_required(view: ViewT) -> ViewT:
    @wraps(view)
    def wrapper(*args: Any, **kwargs: Any) -> ResponseReturnValue:
        user = current_user()
        if user is None:
            return _unauthorised()
        if not user.get("is_admin"):
            current_app.logger.warning(
                "admin access denied for user_id=%s", user["user_id"]
            )
            if wants_json():
                return jsonify(
                    {"error": "forbidden", "message": "Administrator access required."}
                ), 403
            return redirect(url_for("pages.dashboard"))
        return cast(ResponseReturnValue, view(*args, **kwargs))

    return cast(ViewT, wrapper)


def start_session(user: User) -> None:
    """Log a user in, rotating the session to prevent fixation."""
    session.clear()
    session["user_id"] = user["user_id"]
    session["full_name"] = user["full_name"]
    session.permanent = True
    csrf_token()  # mint a fresh token bound to the new session
