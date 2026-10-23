"""Application factory.

``create_app`` is the only supported way to build the app. It is what the WSGI
entrypoint (``app.py``), the CLI and the tests all call, so there is exactly
one code path that decides how the application is configured.

The model is loaded once at startup, not per request. If it is missing or its
feature contract does not match the code, the app still starts — every other
feature keeps working and ``/api/health`` reports the reason — but assessment
endpoints return 503 with an actionable message instead of a stack trace.
"""

from __future__ import annotations

import logging
import sys
from datetime import timedelta

from flask import Flask, jsonify, render_template
from flask.typing import ResponseReturnValue

from .assessment import Assessor
from .config import Config, load_config
from .db import ApplicationRepository, UserRepository, initialise
from .ml.registry import ModelContractError, ModelUnavailable, load_bundle
from .security import install_csrf, install_security_headers

__version__ = "2.1.0"


def configure_logging(config: Config) -> None:
    logging.basicConfig(
        level=logging.DEBUG if config.env == "development" else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        stream=sys.stdout,
    )


def create_app(config: Config | None = None) -> Flask:
    config = config or load_config()
    configure_logging(config)

    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config.update(
        SECRET_KEY=config.secret_key,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SECURE=config.cookie_secure,
        SESSION_COOKIE_SAMESITE=config.cookie_samesite,
        PERMANENT_SESSION_LIFETIME=timedelta(seconds=config.session_lifetime_seconds),
        MAX_CONTENT_LENGTH=256 * 1024,  # no endpoint accepts an upload
        TRUSTED_PROXY=config.trusted_proxy,
        TRUSTED_PROXY_DEPTH=config.trusted_proxy_depth,
        RATE_LIMITS={
            "login": config.login_rate_limit,
            "register": config.register_rate_limit,
            "predict": config.predict_rate_limit,
        },
        APP_CONFIG=config,
        TESTING=config.testing,
    )

    for warning in config.warnings:
        app.logger.warning(warning)

    initialise(config.database_path)
    app.extensions["user_repository"] = UserRepository(config.database_path)
    app.extensions["application_repository"] = ApplicationRepository(config.database_path)

    _load_model(app, config)
    install_csrf(app)
    install_security_headers(app)
    _register_blueprints(app)
    _register_error_handlers(app)
    _register_cli(app)

    app.jinja_env.globals.update(app_version=__version__)
    return app


def _load_model(app: Flask, config: Config) -> None:
    """Attach an :class:`Assessor`, or record why we could not."""
    app.extensions["assessor"] = None
    app.extensions["model_status"] = "unavailable"
    app.extensions["model_error"] = None

    try:
        bundle = load_bundle(config.model_dir)
    except ModelUnavailable as exc:
        app.extensions["model_error"] = str(exc)
        app.logger.warning("Model not loaded: %s", exc)
        return
    except ModelContractError as exc:
        # A stale bundle is a deployment error, not a missing-file condition.
        # Serving mis-aligned predictions is the failure mode this whole
        # design exists to prevent, so we refuse rather than degrade.
        app.extensions["model_status"] = "contract_mismatch"
        app.extensions["model_error"] = str(exc)
        app.logger.error("Model rejected: %s", exc)
        return
    except Exception as exc:
        # Anything else the loader can raise: a truncated or corrupt pickle, a
        # permission error, a joblib version it cannot read. save_bundle is not
        # atomic, so a build interrupted mid-write leaves exactly that.
        #
        # This has to be caught here. Letting it escape create_app crash-loops
        # every worker on a condition a restart cannot fix, when the degraded
        # path already exists and reports the reason on /api/health.
        app.extensions["model_status"] = "load_failed"
        app.extensions["model_error"] = (
            f"{type(exc).__name__}: {exc}. The bundle may be corrupt or "
            "truncated; retrain with: python -m loanapp.ml.train"
        )
        app.logger.exception("Model could not be loaded: %s", exc)
        return

    app.extensions["assessor"] = Assessor(bundle, config)
    app.extensions["model_status"] = "ready"
    app.logger.info(
        "Model loaded: version=%s calibrated=%s", bundle.version, bundle.is_calibrated
    )


def _register_blueprints(app: Flask) -> None:
    from .blueprints.admin import admin_bp
    from .blueprints.api import api_bp
    from .blueprints.auth import auth_bp
    from .blueprints.pages import pages_bp

    app.register_blueprint(pages_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(admin_bp, url_prefix="/admin")


def _register_error_handlers(app: Flask) -> None:
    from .security import wants_json

    @app.errorhandler(404)
    def not_found(_: Exception) -> ResponseReturnValue:
        if wants_json():
            return jsonify({"error": "not_found", "message": "No such endpoint."}), 404
        return render_template("error.html", code=404, message="Page not found."), 404

    @app.errorhandler(500)
    def server_error(exc: Exception) -> ResponseReturnValue:
        app.logger.exception("Unhandled error: %s", exc)
        if wants_json():
            # Never leak exception text to the client; the log has the detail.
            return jsonify(
                {"error": "internal_error", "message": "Something went wrong."}
            ), 500
        return render_template(
            "error.html", code=500, message="Something went wrong on our side."
        ), 500


def _register_cli(app: Flask) -> None:
    import click

    @app.cli.command("create-admin")
    @click.option("--user-id", required=True)
    @click.option("--email", required=True)
    @click.option("--full-name", default="Administrator")
    @click.password_option()
    def create_admin(user_id: str, email: str, full_name: str, password: str) -> None:
        """Create an administrator account.

        Replaces the previous behaviour of auto-creating an ``admin``/
        ``admin123`` account on first boot, which shipped a known credential to
        every deployment.
        """
        from .schemas import RegistrationRequest, SchemaError, parse
        from .security import hash_password

        try:
            parse(
                RegistrationRequest,
                {
                    "user_id": user_id,
                    "email": email,
                    "full_name": full_name,
                    "password": password,
                    "confirm_password": password,
                },
            )
        except SchemaError as exc:
            raise click.ClickException(str(exc)) from exc

        repository = app.extensions["user_repository"]
        if repository.exists(user_id.lower(), email.lower()):
            raise click.ClickException("That username or email already exists.")

        repository.create(
            user_id=user_id.lower(),
            email=email.lower(),
            full_name=full_name,
            password_hash=hash_password(password),
            is_admin=True,
        )
        click.echo(f"Created administrator {user_id!r}.")
