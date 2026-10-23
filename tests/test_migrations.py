"""Upgrade the actual shipped v1 schema using disposable SQLite files."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from test_concurrency import _spawn_workers

from loanapp.db import ApplicationRepository, initialise

LEGACY_SCHEMA = Path(__file__).parent / "fixtures" / "schema_v1.sql"
SUPPORTED = {
    "probability": 0.8,
    "band": "likely",
    "band_label": "Likely to be approved",
    "out_of_range": False,
    "model": {"version": "historical"},
    "factors": [],
}
UNSUPPORTED = {
    **SUPPORTED,
    "probability": 0.2,
    "band": "unlikely",
    "out_of_range": True,
    "out_of_range_reason": "no household income was declared",
    "factors": [{"impact": 0.2}],
    "suggestions": [{"delta": 0.1}],
}


@pytest.fixture
def legacy_db(tmp_path):
    path = tmp_path / "legacy.db"
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.executescript(LEGACY_SCHEMA.read_text(encoding="utf-8"))
        conn.execute("INSERT INTO schema_version VALUES (1)")
        conn.execute(
            "INSERT INTO users (user_id,email,full_name,password_hash,created_at) "
            "VALUES ('owner','owner@example.com','Fixture Owner','fixture','2020-01-01')"
        )
        for record_id, result in [(1, SUPPORTED), (2, UNSUPPORTED), (99, SUPPORTED)]:
            conn.execute(
                "INSERT INTO applications (id,user_id,label,payload,assessment,"
                "probability,band,model_version,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    record_id,
                    "owner",
                    "original",
                    '{"loan_amount":1000}',
                    json.dumps(result),
                    result["probability"],
                    result["band"],
                    "historical",
                    "2020-01-01T00:00:00+00:00",
                ),
            )
        conn.execute("DELETE FROM applications WHERE id=99")
    return path


def test_upgrade_preserves_history_and_removes_only_unsupported_claims(legacy_db):
    initialise(str(legacy_db))
    repository = ApplicationRepository(str(legacy_db))
    supported = repository.get(1, user_id="owner")
    unsupported = repository.get(2, user_id="owner")
    assert supported["assessment"] == SUPPORTED
    assert (
        supported["created_at"]
        == unsupported["created_at"]
        == "2020-01-01T00:00:00+00:00"
    )
    assert unsupported["probability"] is None
    assert unsupported["band"] == "unavailable"
    assert unsupported["assessment"]["probability"] is None
    assert (
        unsupported["assessment"]["factors"]
        == unsupported["assessment"]["suggestions"]
        == []
    )
    assert unsupported["model_version"] == "historical"
    assert repository.get(2, user_id="someone_else") is None
    new_id = repository.save("owner", "after upgrade", {}, SUPPORTED)
    assert new_id > 99  # deleted identifiers must never be reused by the upgrade
    with closing(sqlite3.connect(legacy_db)) as conn, conn:
        assert conn.execute("SELECT version FROM schema_version").fetchall() == [(2,)]
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        indexes = {r[1] for r in conn.execute("PRAGMA index_list(applications)")}
        assert {"idx_applications_user", "idx_applications_created"} <= indexes


def test_repeated_upgrade_is_idempotent(legacy_db):
    initialise(str(legacy_db))
    before = ApplicationRepository(str(legacy_db)).get(2, user_id="owner")
    initialise(str(legacy_db))
    after = ApplicationRepository(str(legacy_db)).get(2, user_id="owner")
    assert before == after
    assert after["probability"] is None


@pytest.mark.parametrize(
    "probability,band", [(None, "likely"), (0.5, "unavailable"), (1.1, "likely")]
)
def test_upgraded_database_rejects_inconsistent_score_and_band(
    legacy_db, probability, band
):
    initialise(str(legacy_db))
    with closing(sqlite3.connect(legacy_db)) as conn, conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "UPDATE applications SET probability=?,band=? WHERE id=1",
                (probability, band),
            )
        assert conn.execute(
            "SELECT probability,band FROM applications WHERE id=1"
        ).fetchone() == (0.8, "likely")


def test_failed_upgrade_rolls_back_schema_and_all_original_records(legacy_db):
    with closing(sqlite3.connect(legacy_db)) as conn, conn:
        conn.execute("UPDATE applications SET assessment='invalid-json' WHERE id=2")
    with pytest.raises(RuntimeError, match=r"assessment.*2"):
        initialise(str(legacy_db))
    with closing(sqlite3.connect(legacy_db)) as conn, conn:
        columns = {row[1]: row for row in conn.execute("PRAGMA table_info(applications)")}
        assert columns["probability"][3] == 1  # NOT NULL from v1 remains intact
        assert conn.execute(
            "SELECT id,probability FROM applications ORDER BY id"
        ).fetchall() == [(1, 0.8), (2, 0.2)]
        assert conn.execute("SELECT version FROM schema_version").fetchall() == [(1,)]
        assert (
            conn.execute(
                "SELECT name FROM sqlite_master WHERE name='applications_next'"
            ).fetchall()
            == []
        )


@pytest.mark.slow
def test_concurrent_startup_upgrades_legacy_database_once(legacy_db):
    assert not _spawn_workers(legacy_db, count=4)
    repository = ApplicationRepository(str(legacy_db))
    assert repository.get(2, user_id="owner")["probability"] is None
    with closing(sqlite3.connect(legacy_db)) as conn, conn:
        assert conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0] == 2
        assert conn.execute("SELECT version FROM schema_version").fetchall() == [(2,)]


def test_newer_schema_is_refused_without_downgrading(legacy_db):
    with closing(sqlite3.connect(legacy_db)) as conn, conn:
        conn.execute("UPDATE schema_version SET version=99")
    with pytest.raises(RuntimeError, match="newer"):
        initialise(str(legacy_db))
    with closing(sqlite3.connect(legacy_db)) as conn, conn:
        assert conn.execute("SELECT version FROM schema_version").fetchall() == [(99,)]


def test_sqlite_backup_restores_original_schema_and_history(legacy_db, tmp_path):
    backup = tmp_path / "before-upgrade.db"
    with (
        closing(sqlite3.connect(legacy_db)) as source,
        closing(sqlite3.connect(backup)) as destination,
    ):
        source.backup(destination)
    initialise(str(legacy_db))
    assert ApplicationRepository(str(legacy_db)).get(2)["probability"] is None
    restored = tmp_path / "restored.db"
    with (
        closing(sqlite3.connect(backup)) as source,
        closing(sqlite3.connect(restored)) as destination,
    ):
        source.backup(destination)
    with closing(sqlite3.connect(restored)) as conn:
        assert conn.execute("SELECT version FROM schema_version").fetchall() == [(1,)]
        assert conn.execute(
            "SELECT probability FROM applications WHERE id=2"
        ).fetchone() == (0.2,)
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
