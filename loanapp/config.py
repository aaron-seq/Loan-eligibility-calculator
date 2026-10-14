"""Environment-driven configuration.

Design rule: **fail closed in production, be convenient in development.**
The previous implementation fell back to ``os.urandom(24)`` for the secret key,
which looks secure but silently breaks every session as soon as a second
worker (or a second serverless invocation) starts with a different key. Here a
missing ``SECRET_KEY`` is fatal in production and generates an explicitly
labelled ephemeral key in development.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

#: The only writable path on Vercel and AWS Lambda, and the reason serverless
#: is a demo-only target for this application — see docs/DEPLOYMENT.md.
#:
#: The scanners flag a hardcoded /tmp (bandit B108 / ruff S108) because of
#: symlink and predictable-name attacks on a shared multi-user /tmp. Neither
#: applies here: a serverless invocation gets a private, single-tenant,
#: ephemeral filesystem, and this is the only writable path the platform
#: offers, so it is forced rather than chosen. Always overridable via
#: DATABASE_PATH.
SERVERLESS_WRITABLE_DIR = "/tmp"  # nosec B108  # noqa: S108


class ConfigError(RuntimeError):
    """Raised when the environment cannot produce a safe configuration."""


def load_env_file(path: Path) -> bool:
    """Load ``KEY=value`` pairs from a ``.env`` file into the environment.

    Called by the process entrypoint (``app.py``), deliberately not from
    :func:`load_config`: the library reads ``os.environ`` and nothing else, so
    tests and embedders are never affected by a developer's local file.

    ``override=False`` is the important argument. A real environment variable
    always wins, so a ``.env`` left lying around can never silently replace a
    production secret. ``.dockerignore`` also keeps the file out of the image.

    Returns whether a file was found, so callers can log it if they wish.
    """
    if not path.is_file():
        return False

    from dotenv import load_dotenv  # local: only the entrypoint needs it

    load_dotenv(path, override=False)
    return True


def _bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


@dataclass
class Config:
    """Resolved application configuration."""

    env: str = "production"
    secret_key: str = ""
    database_path: str = ""
    model_dir: str = ""

    # Session / cookie hardening
    session_lifetime_seconds: int = 3600
    cookie_secure: bool = True
    cookie_samesite: str = "Lax"

    # Rate limits (requests per window, per client IP)
    login_rate_limit: tuple[int, int] = (10, 300)
    register_rate_limit: tuple[int, int] = (5, 3600)
    predict_rate_limit: tuple[int, int] = (30, 3600)

    # Lending assumptions used by the affordability calculator. These are
    # inputs to a published formula, not learned parameters — see docs/MODEL.md.
    annual_interest_rate: float = 0.105
    max_emi_to_income: float = 0.50
    min_cibil_score: int = 300
    max_cibil_score: int = 900
    cibil_caution_threshold: int = 650
    cibil_strong_threshold: int = 750

    # Decision bands over the calibrated model probability.
    approve_threshold: float = 0.65
    borderline_threshold: float = 0.40

    warnings: list[str] = field(default_factory=list)

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def testing(self) -> bool:
        return self.env == "testing"


def load_config(env: str | None = None) -> Config:
    """Build a :class:`Config` from the process environment.

    Raises :class:`ConfigError` when production is missing something that
    cannot be safely defaulted.
    """
    env = (env or os.environ.get("FLASK_ENV") or "production").strip().lower()
    if env not in ("production", "development", "testing"):
        raise ConfigError(
            f"FLASK_ENV must be production, development or testing; got {env!r}"
        )

    warnings: list[str] = []

    secret_key = os.environ.get("SECRET_KEY", "").strip()
    if not secret_key:
        if env == "production":
            raise ConfigError(
                "SECRET_KEY is required in production. Generate one with:\n"
                "  python -c 'import secrets; print(secrets.token_urlsafe(48))'"
            )
        secret_key = secrets.token_urlsafe(48)
        warnings.append(
            "SECRET_KEY not set; generated an ephemeral key. Sessions will not "
            "survive a restart and will not work across multiple workers."
        )
    elif len(secret_key) < 32 and env == "production":
        raise ConfigError("SECRET_KEY must be at least 32 characters in production.")

    # On Vercel the only writable path is /tmp, and it does not survive a cold
    # start. We default there so the demo works out of the box, and warn below
    # so nobody mistakes it for persistence. See docs/DEPLOYMENT.md.
    on_serverless = bool(
        os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
    )
    default_database = (
        f"{SERVERLESS_WRITABLE_DIR}/loanwise.db"
        if on_serverless
        else str(BASE_DIR / "loanwise.db")
    )
    database_path = os.environ.get("DATABASE_PATH", default_database)
    model_dir = os.environ.get("MODEL_DIR", str(BASE_DIR / "models"))

    samesite = os.environ.get("SESSION_COOKIE_SAMESITE", "Lax").strip().capitalize()
    # Validated because it is the one cookie setting passed straight through to
    # Werkzeug. An unrecognised value makes Werkzeug raise while *setting the
    # cookie*, i.e. on every response that touches the session — so a typo here
    # takes down every page rather than degrading one setting.
    if samesite not in ("Lax", "Strict", "None"):
        raise ConfigError(
            f"SESSION_COOKIE_SAMESITE must be Lax, Strict or None; got {samesite!r}"
        )
    if samesite == "None" and not _bool("SESSION_COOKIE_SECURE", env == "production"):
        # Browsers reject SameSite=None without Secure, which silently drops
        # the session cookie rather than erroring.
        raise ConfigError(
            "SESSION_COOKIE_SAMESITE=None requires SESSION_COOKIE_SECURE=true; "
            "browsers reject the cookie otherwise."
        )

    config = Config(
        env=env,
        secret_key=secret_key,
        database_path=database_path,
        model_dir=model_dir,
        session_lifetime_seconds=_int("SESSION_LIFETIME_SECONDS", 3600),
        cookie_secure=_bool("SESSION_COOKIE_SECURE", env == "production"),
        cookie_samesite=samesite,
        annual_interest_rate=_float("ANNUAL_INTEREST_RATE", 0.105),
        max_emi_to_income=_float("MAX_EMI_TO_INCOME", 0.50),
        approve_threshold=_float("APPROVE_THRESHOLD", 0.65),
        borderline_threshold=_float("BORDERLINE_THRESHOLD", 0.40),
        warnings=warnings,
    )

    if not 0 < config.borderline_threshold < config.approve_threshold < 1:
        raise ConfigError(
            "Require 0 < BORDERLINE_THRESHOLD < APPROVE_THRESHOLD < 1; got "
            f"{config.borderline_threshold} and {config.approve_threshold}"
        )
    if not 0 <= config.annual_interest_rate < 1:
        raise ConfigError("ANNUAL_INTEREST_RATE is a fraction, e.g. 0.105 for 10.5%")

    if on_serverless:
        if database_path.startswith(SERVERLESS_WRITABLE_DIR):
            warnings.append(
                "Serverless deployment detected: the database lives in /tmp and is "
                "destroyed on every cold start. Accounts and saved applications "
                "will not persist. Anonymous assessment works fine; for anything "
                "stateful see docs/DEPLOYMENT.md."
            )
        else:
            warnings.append(
                f"Serverless deployment with DATABASE_PATH={database_path}. The "
                "filesystem is read-only outside /tmp, so writes will fail."
            )

    return config
