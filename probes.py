"""Host inspection primitives — the code that actually touches a machine.

Deliberately standalone: stdlib plus psutil, no imports from the rest of this project. That
constraint is the whole point. This same file runs in two very different places —

  * inside the API process, when the target is the local host, and
  * inside `itops_agent.py` on a customer's server, where nothing else from this codebase
    exists —

and the moment it grew a dependency on `db` or `agents` it would stop being shippable to a
customer. Two files to install instead of one is a cheap price for having exactly one
implementation of "what is this machine's memory usage", rather than two that drift.

Every probe returns a plain dict, never a formatted string. Formatting for the LLM happens
server-side in `agents.py`; keeping the wire format structured means the same reading can also
go into a metrics table, a threshold comparison, or a chart without being re-parsed out of
English.
"""

import platform
import re
import shutil
import subprocess
from datetime import datetime

import psutil

PROBE_VERSION = "1.1.0"

_IS_WINDOWS = platform.system() == "Windows"
_ROOT = "C:\\" if _IS_WINDOWS else "/"


# --- core resource probes -----------------------------------------------------------------

def cpu() -> dict:
    overall = psutil.cpu_percent(interval=0.5)
    return {
        "percent": overall,
        "per_core": psutil.cpu_percent(interval=None, percpu=True),
        "logical_cores": psutil.cpu_count(),
        "physical_cores": psutil.cpu_count(logical=False),
        "load_avg": list(psutil.getloadavg()) if hasattr(psutil, "getloadavg") else None,
    }


def memory() -> dict:
    m = psutil.virtual_memory()
    return {"percent": m.percent, "total_gb": round(m.total / 1024 ** 3, 2),
            "used_gb": round(m.used / 1024 ** 3, 2),
            "available_gb": round(m.available / 1024 ** 3, 2)}


def swap() -> dict:
    s = psutil.swap_memory()
    if s.total == 0:
        return {"configured": False, "percent": 0.0}
    return {"configured": True, "percent": s.percent,
            "total_mb": s.total // 1024 ** 2, "used_mb": s.used // 1024 ** 2}


def disk() -> dict:
    d = psutil.disk_usage(_ROOT)
    return {"mount": _ROOT, "percent": d.percent,
            "total_gb": round(d.total / 1024 ** 3, 2),
            "used_gb": round(d.used / 1024 ** 3, 2),
            "free_gb": round(d.free / 1024 ** 3, 2)}


def partitions() -> dict:
    out = []
    for p in psutil.disk_partitions(all=False):
        try:
            u = psutil.disk_usage(p.mountpoint)
            out.append({"device": p.device, "mount": p.mountpoint, "fstype": p.fstype,
                        "percent": u.percent, "total_gb": round(u.total / 1024 ** 3, 2),
                        "free_gb": round(u.free / 1024 ** 3, 2)})
        except (PermissionError, OSError):
            out.append({"device": p.device, "mount": p.mountpoint, "error": "unreadable"})
    return {"partitions": out}


def disk_io() -> dict:
    io = psutil.disk_io_counters()
    if io is None:
        return {"available": False}
    return {"available": True, "read_mb": round(io.read_bytes / 1024 ** 2, 1),
            "write_mb": round(io.write_bytes / 1024 ** 2, 1),
            "read_ops": io.read_count, "write_ops": io.write_count}


def network() -> dict:
    n = psutil.net_io_counters()
    return {"sent_mb": round(n.bytes_sent / 1024 ** 2, 1),
            "recv_mb": round(n.bytes_recv / 1024 ** 2, 1),
            "err_in": n.errin, "err_out": n.errout,
            "drop_in": n.dropin, "drop_out": n.dropout}


def connections() -> dict:
    try:
        conns = psutil.net_connections(kind="inet")
    except (psutil.AccessDenied, PermissionError):
        return {"available": False, "reason": "requires elevated privileges"}
    from collections import Counter
    counts = Counter(c.status for c in conns if c.status)
    return {"available": True, "total": len(conns), "by_status": dict(counts)}


def listening_ports() -> dict:
    try:
        conns = psutil.net_connections(kind="inet")
    except (psutil.AccessDenied, PermissionError):
        return {"available": False, "reason": "requires elevated privileges"}
    out = []
    for c in conns:
        if c.status != "LISTEN":
            continue
        name = None
        if c.pid:
            try:
                name = psutil.Process(c.pid).name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        out.append({"ip": c.laddr.ip, "port": c.laddr.port, "pid": c.pid, "process": name})
    return {"available": True, "count": len(out), "ports": out[:100]}


def interfaces() -> dict:
    addrs, stats = psutil.net_if_addrs(), psutil.net_if_stats()
    out = []
    for name, addr_list in addrs.items():
        up = stats[name].isup if name in stats else None
        ips = [a.address for a in addr_list if a.family.name in ("AF_INET", "AF_INET6")]
        out.append({"name": name, "up": up, "addresses": ips})
    return {"interfaces": out}


def top_processes(limit: int = 5) -> dict:
    procs = []
    for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
        try:
            procs.append(p.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    by_cpu = sorted(procs, key=lambda x: x.get("cpu_percent") or 0, reverse=True)[:limit]
    by_mem = sorted(procs, key=lambda x: x.get("memory_percent") or 0, reverse=True)[:limit]
    fmt = lambda rows, key: [
        {"name": r.get("name"), "pid": r.get("pid"), key: round(r.get(key) or 0, 1)}
        for r in rows
    ]
    return {"by_cpu": fmt(by_cpu, "cpu_percent"), "by_memory": fmt(by_mem, "memory_percent")}


def process_count(include_threads: bool = False) -> dict:
    """Process count is free; the thread total is not.

    Measured on Windows with ~380 processes: `len(psutil.pids())` is instant, while walking
    every process for `num_threads` took **4.77 seconds** — on its own, two thirds of the
    entire snapshot's 7.4s. Each process is a separate OS handle open, query and close, and
    the cost scales with how busy the machine is, which is exactly when you least want a
    monitoring agent taking five seconds of it.

    So the thread total is opt-in. Nothing in a health check needs it: process count answers
    "is something forking out of control", and a thread total that costs five seconds to
    obtain is a worse trade than not having it."""
    result = {"processes": len(psutil.pids())}
    if include_threads:
        threads = 0
        for p in psutil.process_iter(["num_threads"]):
            try:
                threads += p.info.get("num_threads") or 0
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        result["threads"] = threads
    return result


def uptime() -> dict:
    boot = datetime.fromtimestamp(psutil.boot_time())
    delta = datetime.now() - boot
    hours, rem = divmod(delta.seconds, 3600)
    return {"days": delta.days, "hours": hours, "minutes": rem // 60,
            "boot_time": boot.isoformat(), "uptime_seconds": int(delta.total_seconds())}


def battery() -> dict:
    try:
        b = psutil.sensors_battery()
    except (AttributeError, NotImplementedError):
        return {"present": False}
    if b is None:
        return {"present": False}
    return {"present": True, "percent": b.percent, "plugged_in": b.power_plugged}


def temperature() -> dict:
    try:
        temps = psutil.sensors_temperatures()
    except (AttributeError, NotImplementedError):
        return {"available": False, "reason": "not supported on this platform"}
    if not temps:
        return {"available": False, "reason": "no sensor exposed to the OS"}
    readings = [{"chip": chip, "label": e.label or "core", "celsius": e.current}
                for chip, entries in temps.items() for e in entries if e.current]
    return {"available": bool(readings), "readings": readings}


def logged_in_users() -> dict:
    return {"users": [
        {"name": u.name, "terminal": u.terminal or "console",
         "since": datetime.fromtimestamp(u.started).isoformat()}
        for u in psutil.users()
    ]}


# --- Windows-specific posture probes ---------------------------------------------------------
#
# These are the checks an auditor asks for and live resource graphs never answer. Each degrades
# to {"available": False} off Windows rather than raising, so the same probe set is safe to run
# against a Linux host — the agent reports what it can and says so about the rest.

def _powershell(script: str, timeout: int = 25) -> tuple[bool, str]:
    if not _IS_WINDOWS or not shutil.which("powershell"):
        return False, "PowerShell is not available on this host."
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=timeout, shell=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {timeout}s."
    except FileNotFoundError:
        return False, "PowerShell not found."
    return True, ((proc.stdout or "") + (proc.stderr or "")).strip()


def pending_reboot() -> dict:
    """A server that has been waiting to reboot for three weeks is a live incident nobody has
    noticed. Windows records this in three separate places and agrees with itself in none of
    them, so all three are checked."""
    if not _IS_WINDOWS:
        return {"available": False, "reason": "Windows-only check"}
    ok, out = _powershell(
        "$k=@("
        "'HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Component Based Servicing\\RebootPending',"
        "'HKLM:\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\WindowsUpdate\\Auto Update\\RebootRequired',"
        "'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\PendingFileRenameOperations');"
        "$r=@(); foreach($p in $k){ if(Test-Path $p){$r+=$p} }; "
        "if($r.Count -gt 0){'PENDING: '+($r -join '; ')}else{'NONE'}"
    )
    if not ok:
        return {"available": False, "reason": out}
    return {"available": True, "reboot_pending": out.startswith("PENDING"), "detail": out}


def windows_updates() -> dict:
    """Patch level. The first question in any compliance review, and nothing else here
    answers it."""
    if not _IS_WINDOWS:
        return {"available": False, "reason": "Windows-only check"}
    ok, out = _powershell(
        "$s=(Get-HotFix | Sort-Object InstalledOn -Descending | Select-Object -First 3 "
        "| ForEach-Object { \"$($_.HotFixID) $($_.InstalledOn)\" }) -join ' | '; "
        "$c=(Get-HotFix | Measure-Object).Count; \"COUNT=$c RECENT=$s\"", timeout=40,
    )
    if not ok:
        return {"available": False, "reason": out}
    m = re.search(r"COUNT=(\d+)", out)
    return {"available": True, "hotfix_count": int(m.group(1)) if m else None, "detail": out}


def failed_logins(hours: int = 24) -> dict:
    """Security event 4625. A spike here is a brute-force attempt in progress, and no
    resource metric will ever show it."""
    if not _IS_WINDOWS:
        return {"available": False, "reason": "Windows-only check"}
    ok, out = _powershell(
        f"$t=(Get-Date).AddHours(-{int(hours)}); "
        "$e=Get-WinEvent -FilterHashtable @{LogName='Security';Id=4625;StartTime=$t} "
        "-ErrorAction SilentlyContinue; "
        "if($e){\"COUNT=$($e.Count)\"}else{'COUNT=0'}", timeout=40,
    )
    if not ok:
        return {"available": False, "reason": out}
    m = re.search(r"COUNT=(\d+)", out)
    count = int(m.group(1)) if m else 0
    return {"available": True, "window_hours": hours, "failed_logins": count,
            "note": "Reading the Security log usually requires an elevated process."}


def scheduled_task_failures() -> dict:
    """The backup job that silently stopped running three weeks ago lives here."""
    if not _IS_WINDOWS:
        return {"available": False, "reason": "Windows-only check"}
    ok, out = _powershell(
        "$f=Get-ScheduledTask | Get-ScheduledTaskInfo -ErrorAction SilentlyContinue "
        "| Where-Object { $_.LastTaskResult -ne 0 -and $_.LastRunTime -gt (Get-Date).AddDays(-7) }; "
        "if($f){ 'FAILED=' + $f.Count + ' :: ' + (($f | Select-Object -First 5 "
        "| ForEach-Object { $_.TaskName }) -join ', ') } else { 'FAILED=0' }", timeout=45,
    )
    if not ok:
        return {"available": False, "reason": out}
    m = re.search(r"FAILED=(\d+)", out)
    return {"available": True, "failed_count": int(m.group(1)) if m else 0, "detail": out}


def time_sync() -> dict:
    """Clock skew quietly breaks Kerberos authentication and makes log correlation across
    hosts meaningless. Cheap to check, invisible until it bites."""
    if not _IS_WINDOWS:
        return {"available": False, "reason": "Windows-only check"}
    ok, out = _powershell("w32tm /query /status 2>&1 | Select-String 'Source|Last Successful'")
    if not ok:
        return {"available": False, "reason": out}
    return {"available": True, "detail": out[:600]}


def certificate_expiry(days: int = 60) -> dict:
    """Certificates expire on a schedule everyone knows and nobody watches."""
    if not _IS_WINDOWS:
        return {"available": False, "reason": "Windows-only check"}
    ok, out = _powershell(
        f"$d=(Get-Date).AddDays({int(days)}); "
        "$c=Get-ChildItem Cert:\\LocalMachine\\My -ErrorAction SilentlyContinue "
        "| Where-Object { $_.NotAfter -lt $d }; "
        "if($c){ 'EXPIRING=' + $c.Count + ' :: ' + (($c | Select-Object -First 5 "
        "| ForEach-Object { \"$($_.Subject) expires $($_.NotAfter.ToString('yyyy-MM-dd'))\" }) "
        "-join '; ') } else { 'EXPIRING=0' }", timeout=35,
    )
    if not ok:
        return {"available": False, "reason": out}
    m = re.search(r"EXPIRING=(\d+)", out)
    return {"available": True, "window_days": days,
            "expiring_count": int(m.group(1)) if m else 0, "detail": out}


def facts() -> dict:
    return {"hostname": platform.node(), "os_family": platform.system(),
            "os_version": platform.release(), "machine": platform.machine(),
            "python": platform.python_version(), "probe_version": PROBE_VERSION}


# --- registry ---------------------------------------------------------------------------------

def posture() -> dict:
    """The six audit-facing checks, in one round trip.

    Kept out of `snapshot()` deliberately: each shells out to PowerShell and costs seconds,
    where every probe in the snapshot is an in-process psutil read. Bundling them here means a
    full health check is two dispatches rather than seven, without making the cheap path pay
    for the expensive one."""
    return {name: run_probe(name) for name in
            ("pending_reboot", "windows_updates", "failed_logins",
             "scheduled_task_failures", "time_sync", "certificate_expiry")}


PROBES = {
    "posture": posture,
    "cpu": cpu, "memory": memory, "swap": swap, "disk": disk, "partitions": partitions,
    "disk_io": disk_io, "network": network, "connections": connections,
    "listening_ports": listening_ports, "interfaces": interfaces,
    "top_processes": top_processes, "process_count": process_count, "uptime": uptime,
    "battery": battery, "temperature": temperature, "logged_in_users": logged_in_users,
    "pending_reboot": pending_reboot, "windows_updates": windows_updates,
    "failed_logins": failed_logins, "scheduled_task_failures": scheduled_task_failures,
    "time_sync": time_sync, "certificate_expiry": certificate_expiry,
    "facts": facts,
}


def run_probe(name: str, **kwargs) -> dict:
    fn = PROBES.get(name)
    if fn is None:
        return {"error": f"Unknown probe '{name}'. Known: {', '.join(sorted(PROBES))}"}
    try:
        return fn(**kwargs)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def vitals() -> dict:
    """The three-number summary the dashboard polls."""
    return {"cpu": psutil.cpu_percent(interval=0.3),
            "memory": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage(_ROOT).percent}


def snapshot() -> dict:
    """Everything cheap, in one round trip.

    This is why remote health checks are viable at all: dispatching 21 individual jobs to an
    agent would mean 21 poll cycles and the better part of a minute. One snapshot is one
    round trip, and the expensive PowerShell probes stay opt-in."""
    return {name: run_probe(name) for name in
            ("facts", "cpu", "memory", "swap", "disk", "partitions", "disk_io", "network",
             "connections", "top_processes", "process_count", "uptime", "battery",
             "temperature", "logged_in_users", "interfaces")}


# --- threshold evaluation -----------------------------------------------------------------
#
# Comparing a number to a threshold is arithmetic, and an LLM asked to do arithmetic will
# eventually get it wrong in a way that reads perfectly fluently. Observed in testing: a model
# handed 80.7% and a stated 85% threshold reported "80.7%, which is above the warning threshold
# of 85%" — confidently, in an otherwise correct report.
#
# So the model is not asked. Findings are computed here, in Python, and handed to it
# pre-classified; its job is to explain them, which is the part it is actually good at.

DEFAULT_THRESHOLDS = {"warn": 85.0, "crit": 95.0}


def _level(value: float, warn: float, crit: float) -> str:
    if value >= crit:
        return "CRITICAL"
    if value >= warn:
        return "WARNING"
    return "OK"


def evaluate(snap: dict, thresholds: dict | None = None) -> dict:
    """Turn a snapshot into explicit findings. Pure, deterministic, no model involved."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    warn, crit = float(t["warn"]), float(t["crit"])
    findings = []

    for key, label in (("cpu", "CPU"), ("memory", "Memory"), ("disk", "Disk")):
        block = snap.get(key) or {}
        pct = block.get("percent")
        if isinstance(pct, (int, float)):
            findings.append({"metric": key, "label": label, "value": pct,
                             "level": _level(pct, warn, crit),
                             "threshold_warn": warn, "threshold_crit": crit})

    swap = snap.get("swap") or {}
    if swap.get("configured") and isinstance(swap.get("percent"), (int, float)):
        findings.append({"metric": "swap", "label": "Swap", "value": swap["percent"],
                         "level": _level(swap["percent"], warn, crit),
                         "threshold_warn": warn, "threshold_crit": crit})

    for part in (snap.get("partitions") or {}).get("partitions", []):
        pct = part.get("percent")
        if isinstance(pct, (int, float)) and pct >= warn:
            findings.append({"metric": "partition", "label": f"Partition {part.get('mount')}",
                             "value": pct, "level": _level(pct, warn, crit),
                             "threshold_warn": warn, "threshold_crit": crit})

    # Posture findings are boolean, not thresholded — present means not OK.
    reboot = snap.get("pending_reboot") or {}
    if reboot.get("available") and reboot.get("reboot_pending"):
        findings.append({"metric": "pending_reboot", "label": "Pending reboot",
                         "value": True, "level": "WARNING"})

    tasks = snap.get("scheduled_task_failures") or {}
    if tasks.get("available") and (tasks.get("failed_count") or 0) > 0:
        findings.append({"metric": "scheduled_task_failures", "label": "Failed scheduled tasks",
                         "value": tasks["failed_count"], "level": "WARNING"})

    certs = snap.get("certificate_expiry") or {}
    if certs.get("available") and (certs.get("expiring_count") or 0) > 0:
        findings.append({"metric": "certificate_expiry", "label": "Certificates expiring",
                         "value": certs["expiring_count"], "level": "WARNING"})

    logins = snap.get("failed_logins") or {}
    if logins.get("available") and (logins.get("failed_logins") or 0) >= 10:
        findings.append({"metric": "failed_logins", "label": "Failed logon attempts",
                         "value": logins["failed_logins"], "level": "WARNING"})

    unavailable = [k for k, v in snap.items()
                   if isinstance(v, dict) and v.get("available") is False]

    levels = [f["level"] for f in findings]
    overall = "CRITICAL" if "CRITICAL" in levels else "WARNING" if "WARNING" in levels else "HEALTHY"
    return {"overall_status": overall, "findings": findings,
            "checks_unavailable": unavailable, "thresholds": t}


if __name__ == "__main__":
    # Runnable standalone so the SSH transport can copy this one file to a target and execute
    # it there, instead of maintaining a second implementation in shell that parses `top` and
    # `free` and disagrees with this one about what "memory used" means.
    #
    #     python3 probes.py snapshot
    #     python3 probes.py cpu
    import json as _json
    import sys as _sys

    _what = _sys.argv[1] if len(_sys.argv) > 1 else "snapshot"
    print(_json.dumps(snapshot() if _what == "snapshot" else run_probe(_what)))
