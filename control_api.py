"""Control-plane routes: accounts, hosts, agent enrolment, and the run history.

Mounted as a router on the existing app rather than replacing it. The legacy single-machine
endpoints keep working exactly as they did — the demo does not break the moment auth exists —
while everything new is authenticated from the first commit. `ITOPS_REQUIRE_AUTH=1` flips the
legacy routes closed too, which is the production posture; the default is open so that
switching it on is a deliberate act rather than something discovered in a meeting.
"""

import json
import os
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

import hosts as hosts_repo
from accounts import (
    SESSION_COOKIE, Principal, authenticate, create_account, current_agent,
    current_user, end_session, start_session,
)
from db import get_conn, tx, utcnow
from security import is_production_key
from transport import TransportError, transport_for

router = APIRouter(prefix="/api")


def require_auth_enabled() -> bool:
    return os.environ.get("ITOPS_REQUIRE_AUTH", "").lower() in ("1", "true", "yes")


def legacy_guard(request: Request) -> Principal | None:
    """Attached to the original single-machine endpoints.

    Open by default so the existing product keeps working the moment this ships, and closed
    the instant `ITOPS_REQUIRE_AUTH=1` is set. A flag that silently does nothing is worse than
    no flag, so this is a real dependency on every legacy route rather than a note in a
    README — flipping it is the one-line production cutover."""
    if not require_auth_enabled():
        return None
    from accounts import current_principal
    return current_principal(request)


# --- schemas -------------------------------------------------------------------------------

class SignupBody(BaseModel):
    email: str
    password: str = Field(min_length=8)
    org_name: str = ""


class LoginBody(BaseModel):
    email: str
    password: str


class AgentHostBody(BaseModel):
    name: str


class SSHHostBody(BaseModel):
    name: str
    address: str
    username: str
    port: int = 22
    auth_method: str = "password"
    secret: str


class EnrolBody(BaseModel):
    enrol_token: str
    hostname: str | None = None
    os_family: str | None = None
    os_version: str | None = None
    agent_version: str | None = None


class JobResultBody(BaseModel):
    job_id: str
    status: str
    result: dict = {}


# --- accounts -------------------------------------------------------------------------------

@router.post("/auth/signup")
def signup(body: SignupBody, request: Request, response: Response):
    try:
        account = create_account(body.email, body.password, body.org_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    hosts_repo.ensure_local_host(account["org_id"], account["user_id"])
    token, expires = start_session(account["user_id"], request.headers.get("user-agent"))
    _set_session_cookie(response, token)
    return {**account, "session_token": token, "expires_at": expires}


@router.post("/auth/login")
def login(body: LoginBody, request: Request, response: Response):
    user = authenticate(body.email, body.password)
    if user is None:
        # One message for both "no such account" and "wrong password" — anything more
        # specific tells an attacker which half they got right.
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    hosts_repo.ensure_local_host(user["org_id"], user["id"])
    token, expires = start_session(user["id"], request.headers.get("user-agent"))
    _set_session_cookie(response, token)
    return {
        "user_id": user["id"], "org_id": user["org_id"], "email": user["email"],
        "session_token": token, "expires_at": expires,
    }


@router.post("/auth/logout")
def logout(request: Request, response: Response, principal: Principal = Depends(current_user)):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        end_session(token)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"ok": True}


@router.get("/auth/me")
def me(principal: Principal = Depends(current_user)):
    return {"user_id": principal.user_id, "org_id": principal.org_id,
            "email": principal.email, "role": principal.role}


def _set_session_cookie(response: Response, token: str) -> None:
    """HttpOnly so XSS cannot read it; SameSite=Lax so a cross-site form post cannot ride it.
    `secure` is on only when a TLS origin is configured — setting it unconditionally would
    make the cookie silently vanish over plain http://localhost during development."""
    response.set_cookie(
        SESSION_COOKIE, token,
        httponly=True, samesite="lax",
        secure=os.environ.get("ITOPS_PUBLIC_URL", "").startswith("https"),
        max_age=60 * 60 * 24 * 14, path="/",
    )


# --- hosts -----------------------------------------------------------------------------------

@router.get("/hosts")
def get_hosts(principal: Principal = Depends(current_user)):
    hosts_repo.ensure_local_host(principal.org_id, principal.user_id)
    return {"hosts": hosts_repo.list_hosts(principal.org_id)}


@router.get("/hosts/{host_id}")
def get_one_host(host_id: str, principal: Principal = Depends(current_user)):
    """No conflict with the POST /hosts/agent and /hosts/ssh routes below — those are POST,
    this is GET, so FastAPI never has to choose between them."""
    host = hosts_repo.get_host(principal.org_id, host_id)
    if host is None:
        raise HTTPException(status_code=404, detail="Host not found.")
    return host


@router.post("/hosts/agent")
def add_agent_host(body: AgentHostBody, principal: Principal = Depends(current_user)):
    """Returns the enrolment token and a ready-to-paste install command. The token is visible
    exactly once — it is stored hashed, so this response is the only chance to copy it."""
    host = hosts_repo.create_agent_host(principal.org_id, body.name, principal.user_id)
    base = os.environ.get("ITOPS_PUBLIC_URL", "http://localhost:8000")
    host["install_command"] = (
        f"python itops_agent.py --server {base} --token {host['enrol_token']}"
    )
    return host


@router.post("/hosts/{host_id}/reissue")
def reissue(host_id: str, principal: Principal = Depends(current_user)):
    try:
        return hosts_repo.reissue_enrolment(principal.org_id, host_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/hosts/ssh")
def add_ssh_host(body: SSHHostBody, principal: Principal = Depends(current_user)):
    try:
        host = hosts_repo.create_ssh_host(
            principal.org_id, body.name, principal.user_id,
            address=body.address, port=body.port, username=body.username,
            auth_method=body.auth_method, secret=body.secret,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return host


@router.post("/hosts/{host_id}/test")
def test_host(host_id: str, principal: Principal = Depends(current_user)):
    """Reaches out and reports back. Also backfills the OS facts, so a successful test is
    what turns a 'pending' row into a real inventory entry."""
    host = hosts_repo.get_host(principal.org_id, host_id)
    if host is None:
        raise HTTPException(status_code=404, detail="Host not found.")
    try:
        facts = transport_for(host).check()
    except TransportError as exc:
        hosts_repo.set_host_status(host_id, "error", str(exc))
        return {"ok": False, "error": str(exc)}
    with tx() as c:
        c.execute(
            "UPDATE hosts SET status='online', last_error=NULL, last_seen_at=?, "
            "os_family=COALESCE(?,os_family), os_version=COALESCE(?,os_version), "
            "hostname=COALESCE(?,hostname) WHERE id=?",
            (utcnow(), facts.get("os_family"), facts.get("os_version"),
             facts.get("hostname"), host_id),
        )
    return {"ok": True, **facts}


@router.post("/hosts/{host_id}/health")
def host_health(host_id: str, query: str = "Check this host's health and give me a full report.",
                principal: Principal = Depends(current_user)):
    """Run the System Health agent against a specific host.

    This is the endpoint the legacy `/api/system-health` should have been: that one has no
    host parameter and can therefore only ever inspect the machine running the API, however
    many servers the customer has enrolled."""
    from health_agent import run_health_check

    host = hosts_repo.get_host(principal.org_id, host_id)
    if host is None:
        raise HTTPException(status_code=404, detail="Host not found.")
    try:
        transport = transport_for(host)
    except TransportError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    result = run_health_check(query, transport=transport, label=host["name"],
                              host_id=host_id, org_id=principal.org_id)
    # Severity comes from the arithmetic, not the prose. See run_health_check's docstring.
    severity = {"HEALTHY": "ok", "WARNING": "warn", "CRITICAL": "crit"}.get(
        result.get("status"), "unknown")
    run_id = record_run(
        org_id=principal.org_id, host_id=host_id, user_id=principal.user_id,
        agent_key="health", request_text=query, report=result["report"],
        trace=result["trace"], severity=severity,
    )
    return {**result, "run_id": run_id, "host_id": host_id,
            "host_name": host["name"], "severity": severity}



@router.get("/hosts/{host_id}/snapshot")
def host_snapshot(host_id: str, principal: Principal = Depends(current_user)):
    """Raw structured metrics for a host, with no LLM in the path. This is what a dashboard
    or a threshold alert should read — deterministic, fast, and free.

    Also records the reading, so polling this endpoint *is* the monitoring: history
    accumulates from ordinary dashboard use rather than needing a separate collector."""
    import metrics as metrics_repo
    import probes as probes_mod

    host = hosts_repo.get_host(principal.org_id, host_id)
    if host is None:
        raise HTTPException(status_code=404, detail="Host not found.")
    try:
        snap = transport_for(host).snapshot()
    except TransportError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    try:
        metrics_repo.record_snapshot(principal.org_id, host_id, snap)
    except Exception:
        pass
    thresholds = metrics_repo.get_thresholds(host_id)
    return {**snap, "evaluation": probes_mod.evaluate(
        snap, {"warn": thresholds["warn"], "crit": thresholds["crit"]})}


@router.get("/hosts/{host_id}/trends")
def host_trends(host_id: str, hours: int = 24, principal: Principal = Depends(current_user)):
    """Baselines, direction of travel, and anomalies for one host. No LLM."""
    import metrics as metrics_repo

    if hosts_repo.get_host(principal.org_id, host_id) is None:
        raise HTTPException(status_code=404, detail="Host not found.")
    hours = max(1, min(hours, 24 * 90))
    return {"host_id": host_id, "window_hours": hours,
            "trends": metrics_repo.summary(host_id, hours),
            "anomalies": metrics_repo.anomalies(host_id, hours),
            "thresholds": metrics_repo.get_thresholds(host_id)}


@router.get("/hosts/{host_id}/series/{metric}")
def host_series(host_id: str, metric: str, hours: int = 24,
                principal: Principal = Depends(current_user)):
    """Raw time series for charting."""
    import metrics as metrics_repo

    if hosts_repo.get_host(principal.org_id, host_id) is None:
        raise HTTPException(status_code=404, detail="Host not found.")
    try:
        return {"metric": metric, "hours": hours,
                "points": metrics_repo.series(host_id, metric, hours=max(1, min(hours, 2160)))}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class ThresholdBody(BaseModel):
    warn_pct: float | None = None
    crit_pct: float | None = None
    notes: str | None = None


@router.get("/hosts/{host_id}/thresholds")
def get_host_thresholds(host_id: str, principal: Principal = Depends(current_user)):
    import metrics as metrics_repo

    if hosts_repo.get_host(principal.org_id, host_id) is None:
        raise HTTPException(status_code=404, detail="Host not found.")
    return metrics_repo.get_thresholds(host_id)


@router.put("/hosts/{host_id}/thresholds")
def put_host_thresholds(host_id: str, body: ThresholdBody,
                        principal: Principal = Depends(current_user)):
    """Per-host warning and critical percentages. A database server idling at 90% memory is
    doing its job; a web server at 90% is about to fall over. One global number cannot be
    correct for both."""
    import metrics as metrics_repo

    if hosts_repo.get_host(principal.org_id, host_id) is None:
        raise HTTPException(status_code=404, detail="Host not found.")
    try:
        return metrics_repo.set_thresholds(host_id, body.warn_pct, body.crit_pct, body.notes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/hosts/{host_id}/probe/{probe_name}")
def host_probe(host_id: str, probe_name: str, principal: Principal = Depends(current_user)):
    host = hosts_repo.get_host(principal.org_id, host_id)
    if host is None:
        raise HTTPException(status_code=404, detail="Host not found.")
    try:
        return transport_for(host).probe(probe_name)
    except TransportError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.get("/hosts/{host_id}/vitals")
def host_vitals(host_id: str, principal: Principal = Depends(current_user)):
    host = hosts_repo.get_host(principal.org_id, host_id)
    if host is None:
        raise HTTPException(status_code=404, detail="Host not found.")
    try:
        return transport_for(host).vitals()
    except TransportError as exc:
        raise HTTPException(status_code=502, detail=str(exc))


@router.delete("/hosts/{host_id}")
def remove_host(host_id: str, principal: Principal = Depends(current_user)):
    if not hosts_repo.delete_host(principal.org_id, host_id):
        raise HTTPException(status_code=404, detail="Host not found.")
    return {"ok": True}


# --- the agent's own channel --------------------------------------------------------------------

@router.post("/agent/enrol")
def agent_enrol(body: EnrolBody):
    """Unauthenticated by necessity — the agent has no key yet; the enrolment token *is* the
    credential, which is why it is single-use and short-lived."""
    try:
        return hosts_repo.redeem_enrolment(body.enrol_token, body.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/agent/jobs")
def agent_poll(principal: Principal = Depends(current_agent)):
    """Long-poll-lite: claim the oldest queued job for this host, or return nothing.

    The claim is a conditional UPDATE rather than a SELECT followed by an UPDATE, so two
    agents racing on the same host cannot both win the same job."""
    hosts_repo.touch_host(principal.host_id)
    conn = get_conn()
    row = conn.execute(
        "SELECT id, kind, payload_json FROM jobs WHERE host_id = ? AND status = 'queued' "
        "ORDER BY created_at LIMIT 1",
        (principal.host_id,),
    ).fetchone()
    if row is None:
        return {"job": None}
    with tx() as c:
        cur = c.execute(
            "UPDATE jobs SET status='claimed', claimed_at=? WHERE id=? AND status='queued'",
            (utcnow(), row["id"]),
        )
    if cur.rowcount == 0:
        return {"job": None}
    return {"job": {"id": row["id"], "kind": row["kind"],
                    "payload": json.loads(row["payload_json"] or "{}")}}


@router.post("/agent/jobs/result")
def agent_result(body: JobResultBody, principal: Principal = Depends(current_agent)):
    status = "done" if body.status == "done" else "failed"
    with tx() as c:
        c.execute(
            "UPDATE jobs SET status=?, result_json=?, finished_at=? "
            "WHERE id=? AND host_id=?",
            (status, json.dumps(body.result), utcnow(), body.job_id, principal.host_id),
        )
    hosts_repo.touch_host(principal.host_id)
    return {"ok": True}


# --- run history ------------------------------------------------------------------------------

def record_run(*, org_id, host_id, user_id, agent_key, request_text,
               report, trace, severity, error=None) -> str:
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    with tx() as c:
        c.execute(
            "INSERT INTO runs (id, org_id, host_id, user_id, agent_key, request, report, "
            "trace_json, severity, started_at, finished_at, error) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, org_id, host_id, user_id, agent_key, request_text, report,
             json.dumps(trace or []), severity, utcnow(), utcnow(), error),
        )
    return run_id


@router.get("/runs")
def list_runs(limit: int = 50, principal: Principal = Depends(current_user)):
    conn = get_conn()
    rows = conn.execute(
        "SELECT r.id, r.host_id, r.agent_key, r.request, r.report, r.severity, "
        "r.started_at, r.error, h.name AS host_name "
        "FROM runs r LEFT JOIN hosts h ON h.id = r.host_id "
        "WHERE r.org_id = ? ORDER BY r.started_at DESC LIMIT ?",
        (principal.org_id, max(1, min(limit, 500))),
    ).fetchall()
    return {"runs": [dict(r) for r in rows]}


@router.get("/collector")
def collector_status(principal: Principal = Depends(current_user)):
    """Whether background collection is actually running, and how the last cycle went.

    Worth an endpoint of its own: a monitoring product whose own collector has quietly stopped
    still renders perfectly, still answers every question, and is silently blind. That failure
    has to be visible from the outside."""
    import collector

    return collector.status()


@router.post("/collector/run")
def collector_run_now(principal: Principal = Depends(current_user)):
    import collector

    return collector.collect_now()


@router.get("/health")
def health():
    """Deployment posture at a glance. Both flags being false is fine locally and is a
    release blocker in production — which is exactly why they are reported rather than
    assumed."""
    return {
        "ok": True,
        "auth_enforced": require_auth_enabled(),
        "production_encryption_key": is_production_key(),
    }
