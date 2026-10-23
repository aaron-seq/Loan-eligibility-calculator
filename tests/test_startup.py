"""Startup under the production server.

A configuration error must stop gunicorn, not just the worker. ``app.py``
used to exit with status 1, which gunicorn treats as an ordinary worker
crash: it respawned the worker every two seconds, forever, with the master
alive and the port open. A misconfigured deploy looked "running" while
serving nothing, and the clear error message scrolled past in a loop.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

pytest.importorskip("gunicorn")


def test_entrypoint_reports_configuration_error_and_exits(tmp_path):
    """The config failure is also checked on platforms without Gunicorn."""
    entrypoint = tmp_path / "app.py"
    shutil.copy(REPO_ROOT / "app.py", entrypoint)
    env = {k: v for k, v in os.environ.items() if k != "SECRET_KEY"}
    env.update(FLASK_ENV="production", PYTHONPATH=str(REPO_ROOT))
    result = subprocess.run(
        [sys.executable, "app.py"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 3
    assert "SECRET_KEY is required in production" in result.stderr


@pytest.mark.slow
@pytest.mark.skipif(sys.platform == "win32", reason="gunicorn requires Unix (fcntl)")
def test_gunicorn_exits_on_a_configuration_error(tmp_path):
    # A copy of app.py, so a developer's .env next to the real one cannot
    # supply the SECRET_KEY this test relies on being absent.
    shutil.copy(REPO_ROOT / "app.py", tmp_path / "app.py")
    env = {k: v for k, v in os.environ.items() if k != "SECRET_KEY"}
    env.update(
        FLASK_ENV="production",
        PYTHONPATH=str(REPO_ROOT),
        DATABASE_PATH=str(tmp_path / "startup.db"),
        MODEL_DIR=str(REPO_ROOT / "models"),
    )
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "gunicorn",
                "--workers",
                "2",
                "--bind",
                "127.0.0.1:0",
                "app:app",
            ],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,  # asserted below, so the failure message can show stderr
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr
        pytest.fail(
            "gunicorn was still running 30 s after a configuration error; it is "
            f"respawning workers instead of halting.\n{(stderr or '')[-800:]}"
        )

    assert result.returncode != 0
    assert "SECRET_KEY is required in production" in result.stderr
    assert "Worker failed to boot" in result.stderr
