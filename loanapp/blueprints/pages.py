"""HTML pages.

The browser flow is: landing -> apply -> result. Signing in is optional and
only buys saved history, because forcing an account before answering the
question the visitor came with is a poor trade for them.

The form posts here and gets a rendered page back. The previous version posted
a plain HTML form at a JSON endpoint, so submitting it showed the applicant a
wall of raw JSON.
"""

from __future__ import annotations

from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)

from ..assessment import DISCLAIMER
from ..schemas import ApplicationRequest, SchemaError, parse
from ..security import current_user, login_required, rate_limit

pages_bp = Blueprint("pages", __name__)


@pages_bp.app_context_processor
def inject_globals():
    return {"current_user": current_user(), "disclaimer": DISCLAIMER}


@pages_bp.get("/")
def home():
    return render_template(
        "home.html", model_status=current_app.extensions["model_status"]
    )


@pages_bp.get("/login")
def login():
    if current_user():
        return redirect(url_for("pages.dashboard"))
    return render_template("auth/login.html", next=request.args.get("next", ""))


@pages_bp.get("/signup")
def signup():
    if current_user():
        return redirect(url_for("pages.dashboard"))
    return render_template("auth/signup.html")


@pages_bp.get("/apply")
def apply_form():
    """The application form.

    ``prefill`` re-opens a saved application so the applicant can adjust one
    field and re-run it, which is the most common real interaction.
    """
    prefill = {}
    prefill_id = request.args.get("from")
    user = current_user()
    if prefill_id and user:
        record = current_app.extensions["application_repository"].get(
            int(prefill_id), user_id=user["user_id"]
        )
        if record:
            prefill = record["payload"]

    return render_template(
        "apply.html",
        prefill=prefill,
        config=current_app.config["APP_CONFIG"],
        model_status=current_app.extensions["model_status"],
    )


@pages_bp.post("/apply")
@rate_limit("predict")
def submit_application():
    assessor = current_app.extensions.get("assessor")
    if assessor is None:
        flash("The assessment model is not available on this deployment yet.", "error")
        return redirect(url_for("pages.apply_form"))

    try:
        data = parse(ApplicationRequest, request.form.to_dict())
    except SchemaError as exc:
        return (
            render_template(
                "apply.html",
                prefill=request.form.to_dict(),
                errors=exc.errors,
                config=current_app.config["APP_CONFIG"],
                model_status=current_app.extensions["model_status"],
            ),
            400,
        )

    application = data.to_application()
    assessment = assessor.assess(application)
    result = assessment.to_dict()

    saved_id = None
    user = current_user()
    if user is not None:
        saved_id = current_app.extensions["application_repository"].save(
            user_id=user["user_id"],
            label=data.label,
            payload=application.to_dict(),
            assessment=result,
        )

    return render_template(
        "result.html",
        assessment=assessment,
        application=application,
        saved_id=saved_id,
        config=current_app.config["APP_CONFIG"],
    )


@pages_bp.get("/dashboard")
@login_required
def dashboard():
    user = current_user()
    records = current_app.extensions["application_repository"].list_for_user(
        user["user_id"]
    )
    return render_template("dashboard.html", applications=records)


@pages_bp.get("/applications/<int:application_id>")
@login_required
def view_application(application_id: int):
    user = current_user()
    record = current_app.extensions["application_repository"].get(
        application_id, user_id=user["user_id"]
    )
    if record is None:
        return render_template(
            "error.html", code=404, message="No such application."
        ), 404
    return render_template("saved_application.html", record=record)


@pages_bp.get("/how-it-works")
def how_it_works():
    """What the model is, how well it performs, and what it cannot do.

    A user-facing page rather than a buried document: a tool that tells people
    something about their finances owes them a plain account of its limits.
    """
    assessor = current_app.extensions.get("assessor")
    return render_template(
        "how_it_works.html",
        bundle=assessor.bundle if assessor else None,
        config=current_app.config["APP_CONFIG"],
    )
