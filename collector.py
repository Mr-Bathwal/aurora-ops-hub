"""Background metric collection.

Until now history only grew when somebody happened to look at a host, which makes a trend as
good as how often an operator clicked. That is backwards: the whole value of a baseline is that
it was being recorded *before* anyone suspected a problem.

This is a daemon thread inside the API rather than an external cron entry, for one reason that
matters more than elegance: an external scheduler is a second thing to install, configure and
forget to turn on, and a monitoring product whose monitoring depends on someone remembering to
set up a cron job will eventually be found not monitoring. The platform collects its own
telemetry or it does not deserve the name.

Three properties it has to have, and each is a real failure mode avoided:

  * One slow or dead host must not stall the others. Each host is collected independently and
    a failure is recorded against that host, not raised.
  * It must never take the API down. The loop catches everything; a collector that crashes the
    web server is worse than no collector.
  * It must be safe to run alongside operator-triggered checks. Both paths write to the same
    metrics table through the same function, and SQLite in WAL mode handles that.
"""

import os
import threading
import time
import traceback
from datetime import datetime, timezone

DEFAULT_INTERVAL_SECONDS = 300          # five minutes
MIN_INTERVAL_SECONDS = 30
PRUNE_EVERY_N_CYCLES = 288              # once a day at the default interval

_thread: threading.Thread | None = None
_stop = threading.Event()
_state: dict = {"running": False, "cycles": 0, "last_run": None,
                "last_error": None, "hosts_ok": 0, "hosts_failed": 0}


def interval_seconds() -> int:
    try:
        value = int(os.environ.get("ITOPS_COLLECT_INTERVAL", DEFAULT_INTERVAL_SECONDS))
    except ValueError:
        return DEFAULT_INTERVAL_SECONDS
    return max(MIN_INTERVAL_SECONDS, value)


def is_enabled() -> bool:
    """On by default. Opting out is for tests and for the second process in a multi-worker
    deployment — see the note in `start()`."""
    return os.environ.get("ITOPS_COLLECT", "1").lower() not in ("0", "false", "no")


def _collect_once() -> tuple[int, int]:
    """One pass over every host that could plausibly answer. Returns (ok, failed)."""
    import hosts as hosts_repo
    import metrics as metrics_repo
    from db import get_conn
    from transport import TransportError, transport_for

    conn = get_conn()
    rows = conn.execute(
        # 'pending' hosts are skipped deliberately: an agent host that has never enrolled has
        # nothing listening, and queuing a job for it every five minutes would build a backlog
        # that floods the agent the moment it finally connects.
        "SELECT id, org_id, name, connection_type FROM hosts "
        "WHERE status IN ('online', 'error')"
    ).fetchall()

    ok = failed = 0
    for row in rows:
        if _stop.is_set():
            break
        host = dict(row)
        try:
            snap = transport_for(host).snapshot()
            metrics_repo.record_snapshot(host["org_id"], host["id"], snap)
            hosts_repo.set_host_status(host["id"], "online", None)
            ok += 1
        except TransportError as exc:
            # Expected: the machine is off, or the agent is stopped. Record it against the
            # host so the inventory shows why, and carry on to the next one.
            hosts_repo.set_host_status(host["id"], "error", str(exc)[:400])
            failed += 1
        except Exception as exc:
            hosts_repo.set_host_status(host["id"], "error", f"collector: {exc}"[:400])
            failed += 1
    return ok, failed


def _loop() -> None:
    import metrics as metrics_repo

    _state["running"] = True
    while not _stop.is_set():
        started = time.monotonic()
        try:
            ok, failed = _collect_once()
            _state.update({"hosts_ok": ok, "hosts_failed": failed, "last_error": None,
                           "last_run": datetime.now(timezone.utc).isoformat()})
            _state["cycles"] += 1
            if _state["cycles"] % PRUNE_EVERY_N_CYCLES == 0:
                metrics_repo.prune()
        except Exception:
            # Nothing in a collection cycle is allowed to end the loop. Record and continue —
            # a collector that dies silently after one bad night is worse than one that logs.
            _state["last_error"] = traceback.format_exc(limit=3)

        # Wait out the remainder of the interval, but wake immediately on shutdown so the
        # process does not hang for five minutes on Ctrl+C.
        elapsed = time.monotonic() - started
        _stop.wait(max(1.0, interval_seconds() - elapsed))
    _state["running"] = False


def start() -> None:
    """Launch the collector.

    Daemon thread so it never blocks interpreter exit. Note for deployment: with multiple
    uvicorn workers each process would run its own collector and multiply the write rate by
    the worker count. Run a single worker, or set ITOPS_COLLECT=0 on all but one."""
    global _thread
    if not is_enabled():
        _state["last_error"] = "disabled via ITOPS_COLLECT"
        return
    if _thread and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_loop, name="itops-collector", daemon=True)
    _thread.start()


def stop() -> None:
    _stop.set()
    if _thread:
        _thread.join(timeout=5)


def status() -> dict:
    return {**_state, "enabled": is_enabled(), "interval_seconds": interval_seconds()}


def collect_now() -> dict:
    """Force a cycle immediately, for an operator who does not want to wait for the timer."""
    ok, failed = _collect_once()
    _state.update({"hosts_ok": ok, "hosts_failed": failed,
                   "last_run": datetime.now(timezone.utc).isoformat()})
    return {"hosts_ok": ok, "hosts_failed": failed}
