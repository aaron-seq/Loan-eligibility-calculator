""".env loading.

The README told contributors to ``cp .env.example .env``, but nothing ever
loaded it: ``python app.py`` and gunicorn ignored the file, so every setting
placed there was silently discarded. python-dotenv was pinned and never
imported.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from loanapp.config import load_env_file

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def clean_env(monkeypatch):
    for key in ("LOANWISE_TEST_A", "LOANWISE_TEST_B", "SECRET_KEY"):
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_values_in_the_file_are_loaded(tmp_path, clean_env):
    env = tmp_path / ".env"
    env.write_text("LOANWISE_TEST_A=from-file\n")
    assert load_env_file(env) is True
    assert os.environ["LOANWISE_TEST_A"] == "from-file"


def test_a_real_environment_variable_always_wins(tmp_path, clean_env):
    """Production sets real variables. A stray .env must never override them."""
    clean_env.setenv("LOANWISE_TEST_A", "from-environment")
    env = tmp_path / ".env"
    env.write_text("LOANWISE_TEST_A=from-file\n")
    load_env_file(env)
    assert os.environ["LOANWISE_TEST_A"] == "from-environment"


def test_a_missing_file_is_a_quiet_no_op(tmp_path, clean_env):
    assert load_env_file(tmp_path / "absent.env") is False


def test_a_byte_order_mark_does_not_drop_the_first_variable(tmp_path, clean_env):
    """Windows editors often save with a UTF-8 BOM. Before python-dotenv 1.2.3
    it silently discarded the *first* variable — SECRET_KEY in .env.example."""
    env = tmp_path / ".env"
    env.write_bytes("﻿LOANWISE_TEST_A=first\nLOANWISE_TEST_B=second\n".encode())
    load_env_file(env)
    assert os.environ.get("LOANWISE_TEST_A") == "first"
    assert os.environ.get("LOANWISE_TEST_B") == "second"


@pytest.mark.slow
def test_running_app_py_reads_the_env_file_next_to_it(tmp_path):
    """End to end: the entrypoint, as a contributor runs it.

    The .env supplies a production SECRET_KEY. Without it the app refuses to
    boot in production, so a clean start proves the file was read.
    """
    shutil.copy(REPO_ROOT / "app.py", tmp_path / "app.py")
    (tmp_path / ".env").write_text(
        "FLASK_ENV=production\n"
        f"SECRET_KEY={'k' * 48}\n"
        f"DATABASE_PATH={tmp_path / 'env.db'}\n"
        f"MODEL_DIR={REPO_ROOT / 'models'}\n"
        "SESSION_COOKIE_SECURE=true\n"
    )
    env = {k: v for k, v in os.environ.items() if k not in ("SECRET_KEY", "FLASK_ENV")}
    env["PYTHONPATH"] = str(REPO_ROOT)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import app; print('BOOTED', app.app.config['SECRET_KEY'][:4])",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,  # asserted below, so the failure message can show stderr
    )
    assert result.returncode == 0, result.stderr[-800:]
    assert "BOOTED kkkk" in result.stdout
