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

from flask import Blueprint, current_app, jsonify, request

from .. import __version__
from ..assessment import DISCLAIMER
from ..schemas import ApplicationRequest, CompareRequest, SchemaError, parse
from ..security import csrf_token, current_user, login_required, rate_limit

api_bp = Blueprint("api", __name__)


def _assessor():
    """Return the loaded assessor, or a 503 response explaining why not."""
    assessor = current_app.extensions.get("assessor")
    if assessor is not None:
        return assessor, None

    status = current_app.extensions.get("model_status")
    if status == "contract_mismatch":
        message = (
            "The deployed model does not match this build's feature contract, "
            "so predictions are disabled. See server logs and redeploy after "
            "retraining."
        )
    else:
        message = "No model is loaded. Train one with: python -m loanapp.ml.train"
    return None, (jsonify({"error": "model_unavailable", "message": message}), 503)


def _payload() -> dict:
    return request.get_json(silent=True) or request.form.to_dict()


@api_bp.get("/health")
def health():
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
        }
    elif current_app.extensions.get("model_error"):
        body["model_error"] = current_app.extensions["model_error"]
    # 200 either way: the process is alive and every non-prediction feature
    # works. Orchestrators should restart on a dead process, not on a missing
    # model, which restarting will not fix.
    return jsonify(body)


@api_bp.get("/csrf-token")
def get_csrf_token():
    """Issue a CSRF token for non-browser clients."""
    return jsonify({"data": {"csrf_token": csrf_token()}})


@api_bp.get("/model-card")
def model_card():
    """Publish what the model is and how well it performs."""
    assessor, error = _assessor()
    if error:
        return error
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
def assess():
    """Assess one application.

    Open to anonymous callers. When a user is signed in the result is saved to
    their history and the response carries the new ``application_id``.
    """
    assessor, error = _assessor()
    if error:
        return error

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
        result["application_id"] = current_app.extensions[
            "application_repository"
        ].save(
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
def compare():
    """Score 2-5 scenarios side by side.

    The what-if view exists because "what would change this?" is the question
    an applicant actually has, and answering it needs more than one score.
    """
    assessor, error = _assessor()
    if error:
        return error

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
    best = max(results, key=lambda r: r["probability"])
    return jsonify(
        {
            "data": {
                "scenarios": results,
                "best_index": best["index"],
                "disclaimer": DISCLAIMER,
            }
        }
    )


@api_bp.get("/applications")
@login_required
def list_applications():
    user = current_user()
    records = current_app.extensions["application_repository"].list_for_user(
        user["user_id"]
    )
    return jsonify({"data": {"applications": records, "count": len(records)}})


@api_bp.get("/applications/<int:application_id>")
@login_required
def get_application(application_id: int):
    user = current_user()
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
def delete_application(application_id: int):
    user = current_user()
    deleted = current_app.extensions["application_repository"].delete(
        application_id, user_id=user["user_id"]
    )
    if not deleted:
        return jsonify({"error": "not_found", "message": "No such application."}), 404
    return jsonify({"data": {"deleted": application_id}})
