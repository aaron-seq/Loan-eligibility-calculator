-- Shipped schema captured from fcdad2d (LoanWise 2.1.0).

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
