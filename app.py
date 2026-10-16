"""WSGI entrypoint.

Kept at the repository root and named ``app`` so ``gunicorn app:app`` and every
platform that auto-detects a Flask app keep working. All real logic lives in
the :mod:`loanapp` package; this file only builds the application.

    Local:      python app.py            (reads .env next to this file)
    Production: gunicorn --workers 2 --bind 0.0.0.0:8000 app:app
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from loanapp import create_app
from loanapp.config import ConfigError, load_env_file

# Before create_app, which reads the environment. A real variable always wins
# over the file (override=False), so this is a local convenience that cannot
# shadow a production secret.
load_env_file(Path(__file__).with_name(".env"))

try:
    app = create_app()
except ConfigError as exc:
    # Refuse to start misconfigured rather than booting into a state where
    # sessions silently break. The message says exactly what to fix.
    print(f"Configuration error:\n\n  {exc}\n", file=sys.stderr)
    # Status 3, not 1. Gunicorn halts the server only when a worker exits
    # with 3 (its WORKER_BOOT_ERROR); any other status counts as a crash and
    # the worker is respawned every two seconds, forever, with the master
    # alive and the port open — a deploy that looks "running" and serves
    # nothing. With 3 gunicorn exits and the platform reports a failed
    # deploy. Outside gunicorn it is just a non-zero exit.
    raise SystemExit(3) from exc


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_ENV") == "development",
    )
