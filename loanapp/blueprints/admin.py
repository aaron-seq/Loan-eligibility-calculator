"""Analyst / administrator views.

Aggregates rather than a raw dump of every row. The previous admin page
selected every prediction ever made, joined it to the users table, and rendered
the lot — which is both a scaling problem and more personal data than an
operator needs on screen to answer "how is this system behaving?".

Admin accounts are created with ``flask create-admin``. There is no default
account and no default password.
"""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, render_template

from ..security import admin_required, json_endpoint

admin_bp = Blueprint("admin", __name__)


@admin_bp.get("/")
@admin_required
def dashboard():
    repository = current_app.extensions["application_repository"]
    return render_template(
        "admin/dashboard.html",
        stats=repository.aggregate(),
        recent=repository.recent(limit=50),
        user_count=current_app.extensions["user_repository"].count(),
        model_status=current_app.extensions["model_status"],
        assessor=current_app.extensions.get("assessor"),
    )


@admin_bp.get("/stats")
@json_endpoint
@admin_required
def stats():
    """Aggregates as JSON, for dashboards or monitoring.

    Counts and rates only — no per-applicant rows, so this can be scraped
    without exposing personal data.
    """
    return jsonify(
        {
            "data": {
                **current_app.extensions["application_repository"].aggregate(),
                "users": current_app.extensions["user_repository"].count(),
                "model_status": current_app.extensions["model_status"],
            }
        }
    )
