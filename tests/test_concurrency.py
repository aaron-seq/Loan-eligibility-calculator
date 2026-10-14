"""Cross-process startup invariants.

Background: while this branch was in flight, `main` fixed two SQLite races that
killed `gunicorn --workers 2` before it could serve a request — a per-connection
``PRAGMA journal_mode=WAL`` taking an exclusive lock, and a check-then-act
insert during startup. `loanapp/db.py` carried the same two shapes, so the fix
was ported here.

**Honest scope of these tests.** I tried to reproduce main's "database is
locked" failure against this architecture and could not, including with the
per-connection WAL switch *and* the connect timeout both removed, at four
concurrent workers. Two likely reasons: ``sqlite3.connect(timeout=...)`` sets
the busy handler before any PRAGMA runs, and this startup path writes far less
inside the lock window than main's did (there is no default-admin seeding, so
no password hash is computed while holding it).

So these do **not** reproduce that failure, and they are not a regression guard
for it. What they do verify, under real concurrent startup, is the end state:
one schema-version row, WAL actually persisted, the per-connection PRAGMAs
applied, and the foreign key genuinely enforced.

The fix itself *was* validated outside this file, by running the shipped
configuration for real: ``gunicorn --workers 6 --threads 4`` against one SQLite
file booted all six workers with zero failures, left ``journal_mode=wal`` and
exactly one ``schema_version`` row, and served 20 concurrent assessments with
no errors. That is the evidence for the change; the tests below are the cheap
guard that keeps the end state from drifting.

Subprocesses are real because the invariants are cross-process; a same-process
loop shares a connection pool and proves nothing about worker startup.
"""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Each worker does what a gunicorn worker does on boot: build the app, which
# initialises the database.
WORKER = (
    f"import sys; sys.path.insert(0, r'{REPO_ROOT}');"
    "from loanapp import create_app;"
    "from loanapp.config import load_config;"
    "create_app(load_config('development'))"
)


def _spawn_workers(database_path: Path, count: int) -> list[str]:
    """Start ``count`` workers against one database; return their failures."""
    env = {
        **os.environ,
        "FLASK_ENV": "development",
        "SECRET_KEY": "concurrent-startup-test-key-at-least-32-chars",
        "DATABASE_PATH": str(database_path),
        "SESSION_COOKIE_SECURE": "false",
    }
    processes = [
        subprocess.Popen(  # noqa: S603
            [sys.executable, "-c", WORKER],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        for _ in range(count)
    ]

    failures = []
    for process in processes:
        _, stderr = process.communicate(timeout=180)
        if process.returncode != 0:
            text = stderr.decode(errors="replace").strip()
            failures.append(text.splitlines()[-1] if text else "no stderr")
    return failures


@pytest.mark.slow
def test_concurrent_worker_startup_leaves_a_consistent_database(tmp_path):
    """Four workers initialise one database file without corrupting it.

    Asserts the end state, not the absence of a race — see the module
    docstring. A regression in the ``INSERT OR IGNORE`` would surface here as a
    duplicate schema-version row; one in WAL persistence as a non-WAL journal.
    """
    database = tmp_path / "concurrent.db"
    failures = _spawn_workers(database, count=4)
    assert not failures, "workers failed to start: " + "; ".join(failures)

    connection = sqlite3.connect(str(database))
    try:
        # Exactly one row, despite four concurrent initialisers.
        assert (
            connection.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 1
        )
        # WAL is persistent, so it must have survived whichever worker won.
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        assert str(mode).lower() == "wal", mode
    finally:
        connection.close()


@pytest.mark.slow
def test_schema_version_insert_is_genuinely_idempotent(tmp_path):
    """Regression: the ``INSERT OR IGNORE`` was a no-op.

    ``schema_version`` had no UNIQUE or PRIMARY KEY, so there was nothing for
    OR IGNORE to conflict on, and ``initialise`` still did check-then-act. A
    code review reproduced 6 rows from 6 concurrent initialisers; the earlier
    test asserting ``COUNT == 1`` passed only by timing luck.

    This drives ``initialise`` directly with 12 workers. Every one of them now
    inserts unconditionally — the pre-check is gone — so OR IGNORE against the
    PRIMARY KEY is the only thing keeping the count at one. If the constraint
    is dropped, this fails every run rather than occasionally.
    """
    database = tmp_path / "idempotent.db"
    code = (
        f"import sys; sys.path.insert(0, r'{REPO_ROOT}');"
        f"from loanapp.db import initialise; initialise(r'{database}')"
    )
    processes = [
        subprocess.Popen(  # noqa: S603
            [sys.executable, "-c", code], stderr=subprocess.PIPE
        )
        for _ in range(12)
    ]
    failures = []
    for process in processes:
        _, stderr = process.communicate(timeout=180)
        if process.returncode != 0:
            text = stderr.decode(errors="replace").strip()
            failures.append(text.splitlines()[-1] if text else "no stderr")

    assert not failures, "initialisers failed: " + "; ".join(failures)

    connection = sqlite3.connect(str(database))
    try:
        assert (
            connection.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 1
        )
    finally:
        connection.close()


def test_schema_version_has_a_uniqueness_constraint(tmp_path):
    """The cheap, deterministic half of the test above: without this
    constraint the OR IGNORE above is silently meaningless."""
    from loanapp.db import initialise

    database = str(tmp_path / "constraint.db")
    initialise(database)

    connection = sqlite3.connect(database)
    try:
        ddl = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = 'schema_version'"
        ).fetchone()[0]
        assert "PRIMARY KEY" in ddl.upper() or "UNIQUE" in ddl.upper(), ddl
    finally:
        connection.close()


def test_pragmas_are_applied_per_connection(tmp_path):
    """``foreign_keys`` is off by default in SQLite, so the FK that
    ``applications`` declares against ``users`` would never be enforced."""
    from loanapp.db import connect, initialise

    database = str(tmp_path / "pragmas.db")
    initialise(database)

    with connect(database) as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 30000


def test_wal_is_set_once_at_init_not_per_connection(tmp_path):
    """The journal mode is a property of the file, so a plain connection must
    inherit WAL without having to switch it."""
    from loanapp.db import connect, initialise

    database = str(tmp_path / "wal.db")
    initialise(database)

    with connect(database) as conn:
        assert str(conn.execute("PRAGMA journal_mode").fetchone()[0]).lower() == "wal"


def test_orphan_applications_are_rejected(tmp_path):
    """Proves the foreign key is actually enforced, not merely declared."""
    from loanapp.db import ApplicationRepository, initialise

    database = str(tmp_path / "fk.db")
    initialise(database)

    with pytest.raises(sqlite3.IntegrityError):
        ApplicationRepository(database).save(
            user_id="nobody-has-this-id",
            label="orphan",
            payload={"loan_amount": 1},
            assessment={
                "probability": 0.5,
                "band": "borderline",
                "model": {"version": "v"},
            },
        )
