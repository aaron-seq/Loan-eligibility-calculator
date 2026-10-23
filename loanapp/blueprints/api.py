"""JSON API.

The API is the real interface; the HTML pages are a client of it. Response
shapes are stable and documented in ``docs/API.md``:

* success  ``{"data": ...}``
* failure  ``{"error": "<machine_code>", "message": "<human text>", ...}``

``/api/assess`` is intentionally usable without an account so the tool is
reachable to someone who just wants an answer. Saving to history requires
signing in.
"""

from __future__ import annotations

from typing import Any, cast

from flask import Blueprint, current_app, jsonify, request
from flask.typing import ResponseReturnValue

from .. import __version__
from ..assessment import DISCLAIMER, Assessor
from ..schemas import ApplicationRequest, CompareRequest, SchemaError, parse
from ..security import (
    authenticated_user,
    csrf_token,
    current_user,
    login_required,
    rate_limit,
)

api_bp = Blueprint("api", __name__)


def _assessor() -> Assessor | None:
    """The loaded assessor, if there is one."""
    return cast(Assessor | None, current_app.extensions.get("assessor"))


#: Why assessment is unavailable, keyed by model_status. Each says what an
#: operator should actually do, rather than a generic "service unavailable".
_UNAVAILABLE_MESSAGES = {
    "contract_mismatch": (
        "The deployed model does not match this build's feature or library "
        "contract, so predictions are disabled. See the server log and redeploy "
        "after retraining."
    ),
    "load_failed": (
        "The model file could not be read; it may be corrupt or truncated. "
        "Retrain with: python -m loanapp.ml.train"
    ),
}


def _model_unavailable() -> ResponseReturnValue:
    """A 503 that says why assessment is offline.

    Replaces a helper that returned an ``(assessor, error_response)`` tuple for
    every caller to unpack. That shape could not be typed precisely and hid the
    success path behind a truthiness check on the second element.
    """
    status = current_app.extensions.get("model_status")
    message = _UNAVAILABLE_MESSAGES.get(
        str(status), "No model is loaded. Train one with: python -m loanapp.ml.train"
    )
    return jsonify({"error": "model_unavailable", "message": message}), 503


def _payload() -> dict[str, Any]:
    return request.get_json(silent=True) or request.form.to_dict()


@api_bp.get("/health")
def health() -> ResponseReturnValue:
    """Liveness plus an honest statement of what works right now."""
    status = current_app.extensions.get("model_status")
    body = {
        "status": "ok" if status == "ready" else "degraded",
        "app_version": __version__,
        "model_status": status,
    }
    assessor = current_app.extensions.get("assessor")
    if assessor is not None:
        body["model"] = {
            "version": assessor.bundle.version,
            "calibrated": assessor.bundle.is_calibrated,
            "trained_at": assessor.bundle.trained_at,
            "metrics": assessor.bundle.metrics,
            # The library versions the bundle was pickled under. Published so
            # an operator can diagnose a contract_mismatch without shell access
            # to the container.
            "runtime_versions": assessor.bundle.runtime_versions,
        }
    elif current_app.extensions.get("model_error"):
        body["model_error"] = current_app.extensions["model_error"]
    # 200 either way: the process is alive and every non-prediction feature
    # works. Orchestrators should restart on a dead process, not on a missing
    # model, which restarting will not fix.
    return jsonify(body)


@api_bp.get("/csrf-token")
def get_csrf_token() -> ResponseReturnValue:
    """Issue a CSRF token for non-browser clients."""
    return jsonify({"data": {"csrf_token": csrf_token()}})


@api_bp.get("/model-card")
def model_card() -> ResponseReturnValue:
    """Publish what the model is and how well it performs."""
    assessor = _assessor()
    if assessor is None:
        return _model_unavailable()
    return jsonify(
        {
            "data": {
                "version": assessor.bundle.version,
                "calibrated": assessor.bundle.is_calibrated,
                "trained_at": assessor.bundle.trained_at,
                "metrics": assessor.bundle.metrics,
                "features": list(assessor.bundle.feature_columns),
                "metadata": assessor.bundle.metadata,
                "disclaimer": DISCLAIMER,
            }
        }
    )


@api_bp.post("/assess")
@rate_limit("predict")
def assess() -> ResponseReturnValue:
    """Assess one application.

    Open to anonymous callers. When a user is signed in the result is saved to
    their history and the response carries the new ``application_id``.
    """
    assessor = _assessor()
    if assessor is None:
        return _model_unavailable()

    try:
        data = parse(ApplicationRequest, _payload())
    except SchemaError as exc:
        return jsonify(
            {
                "error": "validation_failed",
                "message": "Check the highlighted fields.",
                "errors": exc.errors,
            }
        ), 400

    application = data.to_application()
    result = assessor.assess(application).to_dict()

    user = current_user()
    if user is not None:
        result["application_id"] = current_app.extensions["application_repository"].save(
            user_id=user["user_id"],
            label=data.label,
            payload=application.to_dict(),
            assessment=result,
        )
        result["saved"] = True
    else:
        result["saved"] = False
        result["save_hint"] = "Sign in to keep this assessment in your history."

    return jsonify({"data": result})


@api_bp.post("/compare")
@rate_limit("predict")
def compare() -> ResponseReturnValue:
    """Score 2-5 scenarios side by side.

    The what-if view exists because "what would change this?" is the question
    an applicant actually has, and answering it needs more than one score.
    """
    assessor = _assessor()
    if assessor is None:
        return _model_unavailable()

    try:
        data = parse(CompareRequest, request.get_json(silent=True) or {})
    except SchemaError as exc:
        return jsonify(
            {
                "error": "validation_failed",
                "message": "Provide 2-5 valid scenarios.",
                "errors": exc.errors,
            }
        ), 400

    results = assessor.compare([s.to_application() for s in data.scenarios])
    supported = [result for result in results if not result["out_of_range"]]
    best = max(supported, key=lambda r: r["probability"], default=None)
    return jsonify(
        {
            "data": {
                "scenarios": results,
                "best_index": best["index"] if best else None,
                "disclaimer": DISCLAIMER,
            }
        }
    )


@api_bp.get("/applications")
@login_required
def list_applications() -> ResponseReturnValue:
    user = authenticated_user()
    records = current_app.extensions["application_repository"].list_for_user(
        user["user_id"]
    )
    return jsonify({"data": {"applications": records, "count": len(records)}})


@api_bp.get("/applications/<int:application_id>")
@login_required
def get_application(application_id: int) -> ResponseReturnValue:
    user = authenticated_user()
    # Scoped by user_id in the query — a missing record and someone else's
    # record are indistinguishable from the outside.
    record = current_app.extensions["application_repository"].get(
        application_id, user_id=user["user_id"]
    )
    if record is None:
        return jsonify({"error": "not_found", "message": "No such application."}), 404
    return jsonify({"data": record})


@api_bp.delete("/applications/<int:application_id>")
@login_required
def delete_application(application_id: int) -> ResponseReturnValue:
    user = authenticated_user()
    deleted = current_app.extensions["application_repository"].delete(
        application_id, user_id=user["user_id"]
    )
    if not deleted:
        return jsonify({"error": "not_found", "message": "No such application."}), 404
    return jsonify({"data": {"deleted": application_id}})
