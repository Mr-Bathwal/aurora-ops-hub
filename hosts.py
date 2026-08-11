"""The host registry — every machine the platform can inspect, and how it reaches each one.

Three connection types, and the difference between them is entirely a question of who opens
the socket:

  local  the box the API itself runs on. What the product does today, kept as a first-class
         row so 'no hosts configured' is never a special case in the UI.

  agent  a daemon on the customer's server enrols once with a short-lived token, receives a
         long-lived key, then *polls outbound* for work. Nothing inbound is ever opened, so it
         crosses NAT and corporate firewalls without a conversation with their network team,
         and we never hold a credential that grants shell access.

  ssh    we connect out to them with stored credentials. No install on their side, which is
         genuinely easier to trial — but it means holding a secret that opens a shell, and it
         needs a route into their estate. Encrypted at rest (see security.py); the plaintext
         exists only in memory during a connection.

Both of the last two are supported because they suit different customers, not because either
is a fallback for the other: agent for anything long-lived and firewalled, SSH for a quick
proof-of-value or a box nobody will install software on.
"""

import uuid
from datetime import datetime, timedelta, timezone

from db import get_conn, tx, utcnow
from security import decrypt_secret, encrypt_secret, new_token, token_digest

ENROL_TOKEN_TTL_MINUTES = 60


def _row_to_host(row) -> dict:
    host = dict(row)
    host.pop("org_id", None)
    return host


def list_hosts(org_id: str) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT h.*, "
        "  (SELECT address FROM host_credentials WHERE host_id = h.id) AS address, "
        "  (SELECT enrolled_at FROM host_enrolment WHERE host_id = h.id) AS enrolled_at "
        "FROM hosts h WHERE h.org_id = ? ORDER BY h.created_at DESC",
        (org_id,),
    ).fetchall()
    return [_row_to_host(r) for r in rows]


def get_host(org_id: str, host_id: str) -> dict | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM hosts WHERE id = ? AND org_id = ?", (host_id, org_id)
    ).fetchone()
    return _row_to_host(row) if row else None


def ensure_local_host(org_id: str, user_id: str | None = None) -> dict:
    """Every org gets a 'local' host representing the machine the API runs on. Idempotent."""
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM hosts WHERE org_id = ? AND connection_type = 'local'", (org_id,)
    ).fetchone()
    if row:
        return _row_to_host(row)
    import platform
    host_id = f"hst_{uuid.uuid4().hex[:12]}"
    with tx() as c:
        c.execute(
            "INSERT INTO hosts (id, org_id, name, connection_type, status, os_family, "
            "os_version, hostname, last_seen_at, created_at, created_by) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (host_id, org_id, "This server (local)", "local", "online",
             platform.system(), platform.release(), platform.node(),
             utcnow(), utcnow(), user_id),
        )
    return get_host(org_id, host_id)


# --- agent-based hosts ---------------------------------------------------------------------

def create_agent_host(org_id: str, name: str, user_id: str) -> dict:
    """Creates a pending host and issues a one-time enrolment token.

    The token is returned in plaintext here and nowhere else — it is hashed before storage, so
    if the operator loses it the only remedy is to reissue. That is the intended trade: a
    retrievable enrolment token is a permanent credential sitting in a database."""
    host_id = f"hst_{uuid.uuid4().hex[:12]}"
    token = new_token("itops_enrol_")
    expires = (datetime.now(timezone.utc)
               + timedelta(minutes=ENROL_TOKEN_TTL_MINUTES)).isoformat()
    with tx() as c:
        c.execute(
            "INSERT INTO hosts (id, org_id, name, connection_type, status, created_at, created_by) "
            "VALUES (?,?,?,?,?,?,?)",
            (host_id, org_id, name, "agent", "pending", utcnow(), user_id),
        )
        c.execute(
            "INSERT INTO host_enrolment (host_id, enrol_token_hash, enrol_expires_at) "
            "VALUES (?,?,?)",
            (host_id, token_digest(token), expires),
        )
    host = get_host(org_id, host_id)
    host["enrol_token"] = token
    host["enrol_expires_at"] = expires
    return host


def reissue_enrolment(org_id: str, host_id: str) -> dict:
    host = get_host(org_id, host_id)
    if not host or host["connection_type"] != "agent":
        raise ValueError("Not an agent host.")
    token = new_token("itops_enrol_")
    expires = (datetime.now(timezone.utc)
               + timedelta(minutes=ENROL_TOKEN_TTL_MINUTES)).isoformat()
    with tx() as c:
        c.execute(
            "UPDATE host_enrolment SET enrol_token_hash = ?, enrol_expires_at = ? "
            "WHERE host_id = ?",
            (token_digest(token), expires, host_id),
        )
    return {"enrol_token": token, "enrol_expires_at": expires}


def redeem_enrolment(token: str, facts: dict) -> dict:
    """Called by the agent itself, unauthenticated, with the token from its install command.

    Exchanges it for a long-lived agent key and burns the enrolment token in the same
    transaction, so a token intercepted in transit is useless the moment it has been used
    once — and a replay is visibly a second machine trying to enrol."""
    conn = get_conn()
    row = conn.execute(
        "SELECT e.host_id, e.enrol_expires_at, e.agent_key_hash, h.org_id "
        "FROM host_enrolment e JOIN hosts h ON h.id = e.host_id "
        "WHERE e.enrol_token_hash = ?",
        (token_digest(token),),
    ).fetchone()
    if row is None:
        raise ValueError("Invalid or already-used enrolment token.")
    if row["enrol_expires_at"] and row["enrol_expires_at"] <= utcnow():
        raise ValueError("This enrolment token has expired. Reissue one from the dashboard.")

    agent_key = new_token("itops_agent_")
    with tx() as c:
        c.execute(
            "UPDATE host_enrolment SET agent_key_hash = ?, enrol_token_hash = NULL, "
            "enrol_expires_at = NULL, enrolled_at = ? WHERE host_id = ?",
            (token_digest(agent_key), utcnow(), row["host_id"]),
        )
        c.execute(
            "UPDATE hosts SET status = 'online', os_family = ?, os_version = ?, hostname = ?, "
            "agent_version = ?, last_seen_at = ? WHERE id = ?",
            (facts.get("os_family"), facts.get("os_version"), facts.get("hostname"),
             facts.get("agent_version"), utcnow(), row["host_id"]),
        )
    return {"host_id": row["host_id"], "agent_key": agent_key}


HEARTBEAT_WRITE_INTERVAL_SECONDS = 30


def touch_host(host_id: str) -> None:
    """Heartbeat, throttled.

    This is called on every agent poll, and agents poll every three seconds. Writing on each
    one is write amplification with a multiplier of the fleet size: a hundred hosts is thirty
    -odd `UPDATE`s per second, forever, purely to record that nothing has changed. SQLite
    serialises writes, so that queue is what every *other* write in the system — a login, a
    metric sample, a job result — ends up waiting behind.

    Measured here: eight stray agents polling was enough to push a browser login from about a
    second to sixty-eight. The fix is not a faster database, it is not doing the write.

    Skipping it costs nothing real. `last_seen_at` is used to answer "is this host alive",
    where thirty seconds of granularity is far finer than the question needs — a host is not
    declared dead until it has been silent for minutes."""
    conn = get_conn()
    row = conn.execute("SELECT last_seen_at, status FROM hosts WHERE id = ?",
                       (host_id,)).fetchone()
    if row is not None and row["status"] == "online" and row["last_seen_at"]:
        try:
            last = datetime.fromisoformat(row["last_seen_at"])
            if (datetime.now(timezone.utc) - last).total_seconds() < HEARTBEAT_WRITE_INTERVAL_SECONDS:
                return
        except ValueError:
            pass  # unparseable timestamp: fall through and rewrite it properly
    with tx() as c:
        c.execute("UPDATE hosts SET last_seen_at = ?, status = 'online' WHERE id = ?",
                  (utcnow(), host_id))


# --- SSH-based hosts -------------------------------------------------------------------------

def create_ssh_host(org_id: str, name: str, user_id: str, *, address: str, port: int,
                    username: str, auth_method: str, secret: str) -> dict:
    """Stores an SSH target. `secret` is a password or a private key body; either way it is
    encrypted before it touches the disk and is never returned by any read path."""
    if auth_method not in ("password", "private_key"):
        raise ValueError("auth_method must be 'password' or 'private_key'.")
    if not address or not username or not secret:
        raise ValueError("address, username and a credential are all required.")
    host_id = f"hst_{uuid.uuid4().hex[:12]}"
    with tx() as c:
        c.execute(
            "INSERT INTO hosts (id, org_id, name, connection_type, status, created_at, created_by) "
            "VALUES (?,?,?,?,?,?,?)",
            (host_id, org_id, name, "ssh", "pending", utcnow(), user_id),
        )
        c.execute(
            "INSERT INTO host_credentials (host_id, address, port, username, auth_method, "
            "secret_enc, created_at) VALUES (?,?,?,?,?,?,?)",
            (host_id, address, port or 22, username, auth_method,
             encrypt_secret(secret), utcnow()),
        )
    return get_host(org_id, host_id)


def get_ssh_credentials(host_id: str) -> dict:
    """Decrypts on the way out. Called only by the SSH transport, never by an API response."""
    conn = get_conn()
    row = conn.execute("SELECT * FROM host_credentials WHERE host_id = ?", (host_id,)).fetchone()
    if row is None:
        raise ValueError("No credentials stored for this host.")
    creds = dict(row)
    creds["secret"] = decrypt_secret(row["secret_enc"])
    creds.pop("secret_enc", None)
    return creds


def set_host_status(host_id: str, status: str, error: str | None = None) -> None:
    with tx() as c:
        c.execute("UPDATE hosts SET status = ?, last_error = ?, last_seen_at = ? WHERE id = ?",
                  (status, error, utcnow(), host_id))


def delete_host(org_id: str, host_id: str) -> bool:
    with tx() as c:
        cur = c.execute("DELETE FROM hosts WHERE id = ? AND org_id = ?", (host_id, org_id))
    return cur.rowcount > 0
