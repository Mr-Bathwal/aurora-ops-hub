"""SQLite schema and connection handling for the multi-tenant control plane.

Deliberately raw sqlite3 rather than an ORM. The whole data model is five tables with no
inheritance, no lazy loading and no relationship graph to speak of — an ORM would add a
dependency and a mapping layer to save perhaps forty lines of SQL, and it would hide exactly
the thing that matters most here: which column is indexed, and which query the auth check
runs on every single request.

WAL is on because the API and any background job runner read concurrently; foreign keys are
on because SQLite leaves them off by default and silently accepts orphaned rows otherwise.
Both are per-connection PRAGMAs, so they are set in _connect, not once at startup.
"""

import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "itops.db")

# sqlite3 connections are not safe to share across threads, and uvicorn runs sync endpoints in
# a thread pool. One connection per thread, created on first use.
_local = threading.local()


def utcnow() -> str:
    """ISO-8601 UTC. Stored as TEXT — SQLite has no native datetime, and an explicit,
    sortable, timezone-qualified string beats a float nobody can read in a shell."""
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        conn = _connect()
        _local.conn = conn
    return conn


@contextmanager
def tx():
    """Transaction scope. Commits on success, rolls back on any exception — without this,
    a failure partway through host creation leaves the host row but not its credential."""
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


SCHEMA = """
CREATE TABLE IF NOT EXISTS orgs (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id            TEXT PRIMARY KEY,
    org_id        TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    email         TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    role          TEXT NOT NULL DEFAULT 'admin',
    created_at    TEXT NOT NULL,
    last_login_at TEXT
);

-- Opaque session tokens, stored as a SHA-256 digest. A stolen database dump therefore does
-- not hand over live sessions, which is the same reason password_hash exists.
CREATE TABLE IF NOT EXISTS sessions (
    token_hash  TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL,
    user_agent  TEXT,
    revoked     INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);

-- A host is any machine the platform can inspect. connection_type decides which transport
-- runs its tools: 'local' (the box the API itself runs on), 'agent' (a daemon that enrols and
-- polls), or 'ssh' (we connect out with stored credentials).
CREATE TABLE IF NOT EXISTS hosts (
    id               TEXT PRIMARY KEY,
    org_id           TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    name             TEXT NOT NULL,
    connection_type  TEXT NOT NULL CHECK (connection_type IN ('local','agent','ssh')),
    status           TEXT NOT NULL DEFAULT 'pending'
                     CHECK (status IN ('pending','online','offline','error')),
    os_family        TEXT,
    os_version       TEXT,
    hostname         TEXT,
    agent_version    TEXT,
    last_seen_at     TEXT,
    last_error       TEXT,
    created_at       TEXT NOT NULL,
    created_by       TEXT REFERENCES users(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_hosts_org ON hosts(org_id);

-- One-time enrolment tokens for agent-based hosts, and the long-lived agent key that
-- replaces them. Both stored hashed: the plaintext is shown to the operator exactly once,
-- at the moment it is issued, and is not recoverable afterwards.
CREATE TABLE IF NOT EXISTS host_enrolment (
    host_id           TEXT PRIMARY KEY REFERENCES hosts(id) ON DELETE CASCADE,
    enrol_token_hash  TEXT,
    enrol_expires_at  TEXT,
    agent_key_hash    TEXT,
    enrolled_at       TEXT
);
CREATE INDEX IF NOT EXISTS idx_enrol_agentkey ON host_enrolment(agent_key_hash);

-- SSH/WinRM credentials, encrypted at rest with Fernet (see security.py). The secret column
-- never holds plaintext, and the encryption key lives in the environment rather than the DB —
-- otherwise a single stolen file would contain both the lock and the key.
CREATE TABLE IF NOT EXISTS host_credentials (
    host_id      TEXT PRIMARY KEY REFERENCES hosts(id) ON DELETE CASCADE,
    address      TEXT NOT NULL,
    port         INTEGER NOT NULL DEFAULT 22,
    username     TEXT NOT NULL,
    auth_method  TEXT NOT NULL CHECK (auth_method IN ('password','private_key')),
    secret_enc   BLOB NOT NULL,
    created_at   TEXT NOT NULL
);

-- Every agent run, scoped to a host and an org. This replaces the browser-local activity log:
-- the audit trail has to survive a cleared cache to be worth calling an audit trail.
CREATE TABLE IF NOT EXISTS runs (
    id          TEXT PRIMARY KEY,
    org_id      TEXT NOT NULL REFERENCES orgs(id) ON DELETE CASCADE,
    host_id     TEXT REFERENCES hosts(id) ON DELETE SET NULL,
    user_id     TEXT REFERENCES users(id) ON DELETE SET NULL,
    agent_key   TEXT NOT NULL,
    request     TEXT,
    report      TEXT,
    trace_json  TEXT,
    severity    TEXT,
    started_at  TEXT NOT NULL,
    finished_at TEXT,
    error       TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_org_started ON runs(org_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_runs_host ON runs(host_id);

-- Work queued for agent-based hosts. The agent polls, claims, executes, and posts back —
-- which is what lets it sit behind NAT with no inbound port open.
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    host_id      TEXT NOT NULL REFERENCES hosts(id) ON DELETE CASCADE,
    run_id       TEXT REFERENCES runs(id) ON DELETE CASCADE,
    kind         TEXT NOT NULL,
    payload_json TEXT,
    status       TEXT NOT NULL DEFAULT 'queued'
                 CHECK (status IN ('queued','claimed','done','failed','expired')),
    result_json  TEXT,
    created_at   TEXT NOT NULL,
    claimed_at   TEXT,
    finished_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_host_status ON jobs(host_id, status);

-- Numeric readings over time. Without this the platform can say "memory is at 74%" and has
-- no way to answer the only question that matters — is that normal for this host, or has it
-- been climbing for three days? A point-in-time check catches a server that is already on
-- fire; a trend catches the one that will be by Friday.
--
-- Columns rather than a generic (metric, value) key-value table: every reading writes all of
-- these at once from one snapshot, so a narrow table would mean six inserts and a pivot on
-- every read, to store the same numbers.
CREATE TABLE IF NOT EXISTS metrics (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    host_id        TEXT NOT NULL REFERENCES hosts(id) ON DELETE CASCADE,
    org_id         TEXT NOT NULL,
    recorded_at    TEXT NOT NULL,
    cpu            REAL,
    memory         REAL,
    disk           REAL,
    swap           REAL,
    process_count  INTEGER,
    thread_count   INTEGER,
    uptime_seconds INTEGER
);
CREATE INDEX IF NOT EXISTS idx_metrics_host_time ON metrics(host_id, recorded_at DESC);

-- Per-host thresholds. A database server idling at 90% memory is doing its job; a web server
-- at 90% is about to fall over. One global number cannot be right for both, so the default
-- lives in code and the exception lives here.
CREATE TABLE IF NOT EXISTS host_settings (
    host_id     TEXT PRIMARY KEY REFERENCES hosts(id) ON DELETE CASCADE,
    warn_pct    REAL,
    crit_pct    REAL,
    notes       TEXT,
    updated_at  TEXT NOT NULL
);
"""


def init_db() -> None:
    """Create every table if absent. Safe to call on each boot."""
    conn = get_conn()
    conn.executescript(SCHEMA)
    conn.commit()
