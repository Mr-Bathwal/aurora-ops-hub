"""Users, orgs and sessions — registration, login, and the dependency every protected route
hangs off.

Every row is scoped to an org rather than a user, because the unit a customer buys is a team,
not a person. Making that the shape from the start is far cheaper than retrofitting it: a
`user_id` column on hosts would have to become `org_id` later, and every query written against
it rewritten.
"""

import uuid
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, Request, status

from db import get_conn, tx, utcnow
from security import hash_password, new_token, token_digest, verify_password

SESSION_TTL_DAYS = 14
SESSION_COOKIE = "itops_session"


class Principal:
    """Who is making the request. Either a logged-in user, or an enrolled agent acting for its
    own host — both need to be authenticated, but they can do very different things."""

    def __init__(self, *, user_id=None, org_id=None, email=None, role=None, host_id=None):
        self.user_id = user_id
        self.org_id = org_id
        self.email = email
        self.role = role
        self.host_id = host_id

    @property
    def is_agent(self) -> bool:
        return self.host_id is not None


# --- registration / login ----------------------------------------------------------------

def create_account(email: str, password: str, org_name: str) -> dict:
    """First user of an org becomes its admin. Raises on duplicate email."""
    email = email.strip().lower()
    if "@" not in email:
        raise ValueError("A valid email address is required.")
    conn = get_conn()
    existing = conn.execute("SELECT 1 FROM users WHERE email = ?", (email,)).fetchone()
    if existing:
        raise ValueError("An account with that email already exists.")

    org_id = f"org_{uuid.uuid4().hex[:12]}"
    user_id = f"usr_{uuid.uuid4().hex[:12]}"
    now = utcnow()
    with tx() as c:
        c.execute("INSERT INTO orgs (id, name, created_at) VALUES (?,?,?)",
                  (org_id, org_name or f"{email.split('@')[0]}'s team", now))
        c.execute(
            "INSERT INTO users (id, org_id, email, password_hash, role, created_at) "
            "VALUES (?,?,?,?,?,?)",
            (user_id, org_id, email, hash_password(password), "admin", now),
        )
    return {"user_id": user_id, "org_id": org_id, "email": email}


def authenticate(email: str, password: str) -> dict | None:
    """Returns the user row on success, None on any failure.

    The dummy verify on the unknown-email path is not defensive clutter: without it, a missing
    account returns in microseconds while a wrong password takes ~100 ms, and that gap is a
    reliable oracle for enumerating which addresses are registered."""
    email = (email or "").strip().lower()
    conn = get_conn()
    row = conn.execute(
        "SELECT id, org_id, email, password_hash, role FROM users WHERE email = ?", (email,)
    ).fetchone()
    if row is None:
        verify_password(password or "", "scrypt$65536$8$1$AAAA$AAAA")
        return None
    if not verify_password(password or "", row["password_hash"]):
        return None
    with tx() as c:
        c.execute("UPDATE users SET last_login_at = ? WHERE id = ?", (utcnow(), row["id"]))
    return dict(row)


def start_session(user_id: str, user_agent: str | None = None) -> tuple[str, str]:
    """Returns (plaintext_token, expires_at). Only the digest is persisted."""
    token = new_token("itops_sess_")
    expires = (datetime.now(timezone.utc) + timedelta(days=SESSION_TTL_DAYS)).isoformat()
    with tx() as c:
        c.execute(
            "INSERT INTO sessions (token_hash, user_id, created_at, expires_at, user_agent) "
            "VALUES (?,?,?,?,?)",
            (token_digest(token), user_id, utcnow(), expires, (user_agent or "")[:200]),
        )
    return token, expires


def end_session(token: str) -> None:
    with tx() as c:
        c.execute("UPDATE sessions SET revoked = 1 WHERE token_hash = ?", (token_digest(token),))


def resolve_session(token: str) -> Principal | None:
    conn = get_conn()
    row = conn.execute(
        "SELECT s.expires_at, s.revoked, u.id AS user_id, u.org_id, u.email, u.role "
        "FROM sessions s JOIN users u ON u.id = s.user_id "
        "WHERE s.token_hash = ?",
        (token_digest(token),),
    ).fetchone()
    if row is None or row["revoked"]:
        return None
    if row["expires_at"] <= utcnow():
        return None
    return Principal(user_id=row["user_id"], org_id=row["org_id"],
                     email=row["email"], role=row["role"])


def resolve_agent_key(key: str) -> Principal | None:
    """An enrolled agent authenticating as its own host."""
    conn = get_conn()
    row = conn.execute(
        "SELECT h.id AS host_id, h.org_id FROM host_enrolment e "
        "JOIN hosts h ON h.id = e.host_id WHERE e.agent_key_hash = ?",
        (token_digest(key),),
    ).fetchone()
    if row is None:
        return None
    return Principal(org_id=row["org_id"], host_id=row["host_id"])


# --- FastAPI dependencies -----------------------------------------------------------------

def _bearer(request: Request) -> str | None:
    header = request.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return None


def current_principal(request: Request) -> Principal:
    """Accepts a session cookie, a session bearer token, or an agent key.

    Cookie *and* bearer because the two callers differ: the browser wants an HttpOnly cookie it
    cannot leak to XSS, while scripts and the agent want a header they can set explicitly."""
    token = request.cookies.get(SESSION_COOKIE) or _bearer(request)
    if token:
        if token.startswith("itops_agent_"):
            agent = resolve_agent_key(token)
            if agent:
                return agent
        principal = resolve_session(token)
        if principal:
            return principal
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def current_user(principal: Principal = Depends(current_principal)) -> Principal:
    """Routes a human must call. An agent key is authenticated but is not a user — it may post
    its own results and nothing else."""
    if principal.is_agent:
        raise HTTPException(status_code=403, detail="This endpoint requires a user session.")
    return principal


def current_agent(principal: Principal = Depends(current_principal)) -> Principal:
    if not principal.is_agent:
        raise HTTPException(status_code=403, detail="This endpoint requires an agent key.")
    return principal
