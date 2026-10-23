"""Registration, sign-in and sign-out.

Form posts and JSON requests share the same handler and the same validation.
Only the response rendering differs, so the browser flow and the API can never
drift apart in what they accept.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from flask import (
    Blueprint,
    current_app,
    flash,
    jsonify,
    redirect,
    request,
    session,
    url_for,
)
from flask.typing import ResponseReturnValue

from ..schemas import LoginRequest, RegistrationRequest, SchemaError, parse
from ..security import (
    hash_password,
    rate_limit,
    start_session,
    verify_user_password,
    wants_json,
)

auth_bp = Blueprint("auth", __name__)

#: Deliberately identical for "no such user" and "wrong password". Telling the
#: two apart hands an attacker a username oracle.
INVALID_CREDENTIALS = "Incorrect username or password."


def _respond(
    success: bool, message: str, status: int, destination: str, **extra: Any
) -> ResponseReturnValue:
    """Render one outcome for either a JSON client or a browser.

    The parameter is ``destination`` rather than ``redirect_to`` because
    callers pass ``redirect_to=`` through ``**extra`` to put the resolved URL
    in the JSON body — sharing the name silently shadowed it and raised
    TypeError on every successful login.
    """
    if wants_json():
        return jsonify({"success": success, "message": message, **extra}), status
    flash(message, "success" if success else "error")
    return redirect(destination)


def _payload() -> dict[str, Any]:
    return request.get_json(silent=True) or request.form.to_dict()


@auth_bp.post("/register")
@rate_limit("register")
def register() -> ResponseReturnValue:
    try:
        data = parse(RegistrationRequest, _payload())
    except SchemaError as exc:
        if wants_json():
            return jsonify(
                {
                    "success": False,
                    "message": "Check the highlighted fields.",
                    "errors": exc.errors,
                }
            ), 400
        for message in exc.errors.values():
            flash(message, "error")
        return redirect(url_for("pages.signup"))

    repository = current_app.extensions["user_repository"]
    try:
        repository.create(
            user_id=data.user_id,
            email=data.email,
            full_name=data.full_name,
            password_hash=hash_password(data.password),
        )
    except sqlite3.IntegrityError:
        # Do not reveal which of the two collided — that would confirm whether
        # a given email is registered.
        return _respond(
            False,
            "That username or email is not available.",
            409,
            url_for("pages.signup"),
        )

    current_app.logger.info("registered user_id=%s", data.user_id)
    user = repository.find(data.user_id)
    start_session(user)
    return _respond(True, "Account created.", 201, url_for("pages.dashboard"))


@auth_bp.post("/login")
@rate_limit("login")
def login() -> ResponseReturnValue:
    try:
        data = parse(LoginRequest, _payload())
    except SchemaError:
        # Never echo which field was malformed on a login attempt.
        return _respond(False, INVALID_CREDENTIALS, 401, url_for("pages.login"))

    user = current_app.extensions["user_repository"].find(data.user_id)
    if not verify_user_password(user, data.password):
        current_app.logger.info("failed login for user_id=%s", data.user_id)
        return _respond(False, INVALID_CREDENTIALS, 401, url_for("pages.login"))

    start_session(user)
    current_app.logger.info("login user_id=%s", user["user_id"])

    destination = url_for("pages.dashboard")
    requested = request.args.get("next") or _payload().get("next")
    # Only accept same-site relative paths, or this becomes an open redirect.
    if requested and requested.startswith("/") and not requested.startswith("//"):
        destination = requested

    return _respond(True, "Signed in.", 200, destination, redirect_to=destination)


@auth_bp.post("/logout")
def logout() -> ResponseReturnValue:
    session.clear()
    return _respond(True, "Signed out.", 200, url_for("pages.home"))
