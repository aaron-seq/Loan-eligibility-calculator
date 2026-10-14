"""SQLite storage.

Deliberately small: a connection helper, a schema with a version counter, and
two repositories. SQLite is the right choice for a single-container deployment
of this size; ``docs/DEPLOYMENT.md`` explains when it stops being the right
choice and what to move to.

Applications are stored as the canonical JSON payload plus the few columns
worth indexing. The previous schema mirrored 15 model inputs into 15 columns,
which meant every change to the feature set was a migration.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

SCHEMA = """
-- PRIMARY KEY, not a bare column: it is what gives INSERT OR IGNORE in
-- initialise() something to conflict on. Without a uniqueness constraint
-- OR IGNORE is a no-op and concurrent initialisers each append a row.
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER PRIMARY KEY);

CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT UNIQUE NOT NULL,
    email         TEXT UNIQUE NOT NULL,
    full_name     TEXT NOT NULL,
    password_hash TEXT NOT NULL,
    is_admin      INTEGER NOT NULL DEFAULT 0,
    is_active     INTEGER NOT NULL DEFAULT 1,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS applications (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id        TEXT NOT NULL,
    label          TEXT NOT NULL DEFAULT '',
    payload        TEXT NOT NULL,   -- canonical LoanApplication JSON
    assessment     TEXT NOT NULL,   -- full assessment JSON
    probability    REAL NOT NULL,
    band           TEXT NOT NULL,
    model_version  TEXT NOT NULL,
    created_at     TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (user_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_applications_user ON applications (user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_applications_created ON applications (created_at DESC);
"""


#: SQLite stores INTEGER as a signed 64-bit value and raises OverflowError on
#: anything wider. Row ids arrive from URLs, so they are user input.
MAX_ROW_ID = 2**63 - 1


def _is_valid_row_id(value: Any) -> bool:
    return isinstance(value, int) and 0 < value <= MAX_ROW_ID


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@contextmanager
def connect(database_path: str) -> Iterator[sqlite3.Connection]:
    """Open a connection with sane defaults, committing on clean exit.

    A fresh connection per request keeps things simple and thread-safe. The
    original code held one module-level cursor shared across every request,
    which is neither.

    Two PRAGMAs are set here, and one deliberately is not:

    - ``busy_timeout`` makes a blocked writer wait for the lock rather than
      failing instantly. It is set first, because it governs how long
      everything after it will wait.
    - ``foreign_keys`` is off by default in SQLite, so the foreign key
      ``applications`` declares against ``users`` would otherwise never be
      enforced and orphan rows would be accepted silently.
    - ``journal_mode=WAL`` is **not** set here. See :func:`initialise`.
    """
    directory = os.path.dirname(os.path.abspath(database_path))
    if directory:
        os.makedirs(directory, exist_ok=True)

    conn = sqlite3.connect(database_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _enable_wal(conn: sqlite3.Connection) -> None:
    """Switch the database to WAL journalling, once.

    WAL lets readers run concurrently with a writer; under the default rollback
    journal, one in-flight write blocks every read, which surfaces as
    "database is locked" as soon as two requests overlap.

    This is deliberately *not* in :func:`connect`. Switching the journal mode
    needs an exclusive lock and SQLite does not reliably invoke the busy
    handler for it, so running it per-connection makes concurrent gunicorn
    workers collide on startup and die before serving a request. WAL is a
    persistent property of the database file, so setting it once is enough and
    every later connection inherits it — including losing the race to another
    worker, which is why failure here is tolerated.
    """
    try:
        mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if str(mode).lower() != "wal":
            logger.warning("Could not enable WAL; journal_mode is %r", mode)
    except sqlite3.OperationalError as exc:
        # Another worker is mid-switch. Whoever wins sets it for everyone.
        logger.debug("WAL not set on this worker: %s", exc)


def initialise(database_path: str) -> None:
    with connect(database_path) as conn:
        _enable_wal(conn)
        conn.executescript(SCHEMA)
        # Unconditional INSERT OR IGNORE, with no SELECT first. The read was
        # the bug: checking for an empty table and then inserting is
        # check-then-act, and every gunicorn worker runs this path
        # concurrently against one file, so several can all observe it empty.
        # OR IGNORE against the PRIMARY KEY in SCHEMA makes the insert
        # idempotent, which removes the race rather than narrowing it.
        conn.execute(
            "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
            (SCHEMA_VERSION,),
        )


class UserRepository:
    def __init__(self, database_path: str):
        self.database_path = database_path

    def create(
        self,
        user_id: str,
        email: str,
        full_name: str,
        password_hash: str,
        is_admin: bool = False,
    ) -> dict[str, Any]:
        with connect(self.database_path) as conn:
            cursor = conn.execute(
                """INSERT INTO users (user_id, email, full_name, password_hash, is_admin, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_id, email, full_name, password_hash, int(is_admin), now()),
            )
            return {"id": cursor.lastrowid, "user_id": user_id, "email": email}

    def find(self, user_id: str) -> dict[str, Any] | None:
        with connect(self.database_path) as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE user_id = ? AND is_active = 1", (user_id,)
            ).fetchone()
            return dict(row) if row else None

    def exists(self, user_id: str, email: str) -> bool:
        with connect(self.database_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM users WHERE user_id = ? OR email = ?", (user_id, email)
            ).fetchone()
            return row is not None

    def set_password(self, user_id: str, password_hash: str) -> None:
        with connect(self.database_path) as conn:
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE user_id = ?",
                (password_hash, user_id),
            )

    def count(self) -> int:
        with connect(self.database_path) as conn:
            return int(conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"])


class ApplicationRepository:
    def __init__(self, database_path: str):
        self.database_path = database_path

    def save(
        self,
        user_id: str,
        label: str,
        payload: dict[str, Any],
        assessment: dict[str, Any],
    ) -> int:
        with connect(self.database_path) as conn:
            cursor = conn.execute(
                """INSERT INTO applications
                   (user_id, label, payload, assessment, probability, band, model_version, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id,
                    label,
                    json.dumps(payload),
                    json.dumps(assessment),
                    assessment["probability"],
                    assessment["band"],
                    assessment["model"]["version"],
                    now(),
                ),
            )
            # lastrowid is typed Optional because it is None before any
            # INSERT. After a successful INSERT it is always set, so a None
            # here means the write did not happen and must not be reported as
            # if it had.
            if cursor.lastrowid is None:
                raise RuntimeError("INSERT into applications returned no row id")
            return int(cursor.lastrowid)

    def list_for_user(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with connect(self.database_path) as conn:
            rows = conn.execute(
                """SELECT id, label, probability, band, model_version, created_at, payload
                   FROM applications WHERE user_id = ?
                   ORDER BY created_at DESC, id DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        return [{**dict(r), "payload": json.loads(r["payload"])} for r in rows]

    def get(
        self, application_id: int, user_id: str | None = None
    ) -> dict[str, Any] | None:
        """Fetch one application.

        ``user_id`` scopes the lookup to an owner. Callers serving a logged-in
        user must always pass it — filtering in SQL rather than after the fetch
        is what keeps this from becoming an IDOR.

        Returns ``None`` for an out-of-range id rather than letting it reach
        SQLite, which raises ``OverflowError`` on anything wider than 64 bits.
        A URL is user input, so that was a 500 on ``?from=999…9``.
        """
        if not _is_valid_row_id(application_id):
            return None

        query = "SELECT * FROM applications WHERE id = ?"
        params: list[Any] = [application_id]
        if user_id is not None:
            query += " AND user_id = ?"
            params.append(user_id)

        with connect(self.database_path) as conn:
            row = conn.execute(query, params).fetchone()
        if not row:
            return None
        record = dict(row)
        record["payload"] = json.loads(record["payload"])
        record["assessment"] = json.loads(record["assessment"])
        return record

    def delete(self, application_id: int, user_id: str) -> bool:
        if not _is_valid_row_id(application_id):
            return False
        with connect(self.database_path) as conn:
            cursor = conn.execute(
                "DELETE FROM applications WHERE id = ? AND user_id = ?",
                (application_id, user_id),
            )
            return cursor.rowcount > 0

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        with connect(self.database_path) as conn:
            rows = conn.execute(
                """SELECT a.id, a.user_id, a.label, a.probability, a.band,
                          a.model_version, a.created_at, u.full_name, u.email
                   FROM applications a
                   LEFT JOIN users u ON u.user_id = a.user_id
                   ORDER BY a.created_at DESC, a.id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def aggregate(self) -> dict[str, Any]:
        """Portfolio-level counters for the analyst view."""
        with connect(self.database_path) as conn:
            totals = conn.execute(
                """SELECT COUNT(*) AS total,
                          AVG(probability) AS mean_probability,
                          COUNT(DISTINCT user_id) AS applicants
                   FROM applications"""
            ).fetchone()
            bands = conn.execute(
                "SELECT band, COUNT(*) AS n FROM applications GROUP BY band"
            ).fetchall()
            by_day = conn.execute(
                """SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS n
                   FROM applications GROUP BY day ORDER BY day DESC LIMIT 14"""
            ).fetchall()
            versions = conn.execute(
                """SELECT model_version, COUNT(*) AS n FROM applications
                   GROUP BY model_version ORDER BY n DESC"""
            ).fetchall()

        return {
            "total": int(totals["total"] or 0),
            "applicants": int(totals["applicants"] or 0),
            "mean_probability": round(float(totals["mean_probability"] or 0.0), 4),
            "bands": {r["band"]: int(r["n"]) for r in bands},
            "by_day": [{"day": r["day"], "count": int(r["n"])} for r in by_day],
            "model_versions": [
                {"version": r["model_version"], "count": int(r["n"])} for r in versions
            ],
        }
