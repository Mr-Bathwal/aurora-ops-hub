"""The Aurora Ops host agent — the program a customer installs on a server they want managed.

    Enrol once, then run:
        python itops_agent.py --server https://ops.example.com --token itops_enrol_xxx

    On subsequent starts the saved key is reused:
        python itops_agent.py --server https://ops.example.com

Design in one line: **it only ever makes outbound requests.** It opens no port, needs no
inbound firewall rule, and works from behind NAT — which is what makes it installable on a
customer's network without a conversation with their security team. The control plane never
connects to this machine; this machine asks the control plane whether there is anything to do.

Three properties worth stating because each is a deliberate limit:

  * The enrolment token is spent immediately and replaced by a long-lived agent key stored at
    0600 in the user's home directory. The token has a one-hour life and one use, so a copy of
    an install command found later in a shell history is worthless.
  * Command execution is allowlisted (ALLOWED_COMMANDS). The control plane can ask for
    `ipconfig`; it cannot ask for `rm -rf /`. A compromised control plane therefore cannot use
    its fleet as a botnet, which is the failure mode that matters most for software like this.
  * File reads are capped and refuse anything that looks like a credential, mirroring the
    server-side rule so the boundary holds even if the caller is trusted.

Dependencies: `psutil` and `httpx`. Both are already required by the server, so a customer
running Python needs nothing exotic.
"""

import argparse
import json
import os
import platform
import stat
import subprocess
import sys
import time

AGENT_VERSION = "1.0.0"
POLL_INTERVAL_SECONDS = 3
POLL_BACKOFF_MAX_SECONDS = 60
KEY_FILE = os.path.join(os.path.expanduser("~"), ".itops_agent_key")

# The complete set of things a control plane may ask this machine to run. Anything else is
# refused locally, whatever the server says.
ALLOWED_COMMANDS = {
    "ipconfig": ["ipconfig", "/all"],
    "netstat": ["netstat", "-ano"],
    "tasklist": ["tasklist"],
    "systeminfo": ["systeminfo"],
    "arp": ["arp", "-a"],
    "route": ["route", "print"],
    "uname": ["uname", "-a"],
    "df": ["df", "-h"],
    "free": ["free", "-m"],
    "ps": ["ps", "aux"],
    "who": ["who"],
    "uptime": ["uptime"],
}

SECRET_MARKERS = ("env", "secret", "credential", "password", "pwd", "token", "key")


def _log(msg: str) -> None:
    print(f"[itops-agent] {msg}", flush=True)


# --- key storage ------------------------------------------------------------------------------

def save_key(key: str) -> None:
    with open(KEY_FILE, "w", encoding="utf-8") as fh:
        fh.write(key)
    try:
        os.chmod(KEY_FILE, stat.S_IRUSR | stat.S_IWUSR)  # 0600; no-op on some Windows filesystems
    except OSError:
        pass


def load_key() -> str | None:
    if not os.path.exists(KEY_FILE):
        return None
    with open(KEY_FILE, "r", encoding="utf-8") as fh:
        return fh.read().strip() or None


# --- facts about this machine -------------------------------------------------------------------

def local_facts() -> dict:
    return {
        "hostname": platform.node(),
        "os_family": platform.system(),
        "os_version": platform.release(),
        "agent_version": AGENT_VERSION,
    }


def collect_vitals() -> dict:
    import psutil
    root = "C:\\" if platform.system() == "Windows" else "/"
    return {
        "cpu": psutil.cpu_percent(interval=0.3),
        "memory": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage(root).percent,
    }


# --- job handlers -------------------------------------------------------------------------------

def do_run_command(payload: dict) -> dict:
    argv = payload.get("argv") or []
    if not argv:
        return {"error": "No command supplied."}
    name = argv[0]
    allowed = ALLOWED_COMMANDS.get(name)
    if allowed is None:
        return {"error": f"Command '{name}' is not on this agent's allowlist."}
    try:
        proc = subprocess.run(allowed, capture_output=True, text=True,
                              timeout=min(int(payload.get("timeout", 20)), 60), shell=False)
    except FileNotFoundError:
        return {"error": f"'{name}' is not available on this system."}
    except subprocess.TimeoutExpired:
        return {"error": f"'{name}' timed out."}
    return {"output": ((proc.stdout or "") + (proc.stderr or "")).strip()[:20000]}


def do_read_file(payload: dict) -> dict:
    path = payload.get("path") or ""
    base = os.path.basename(path).lower()
    stem = base.rsplit(".", 1)[0]
    if base.startswith(".") or any(m in stem for m in SECRET_MARKERS):
        return {"error": f"Refusing to read '{base}' — it looks like a credential file."}
    if not os.path.isfile(path):
        return {"error": f"File not found: {path}"}
    limit = min(int(payload.get("max_bytes", 200_000)), 1_000_000)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return {"content": fh.read(limit)}


def do_probe(payload: dict) -> dict:
    """Run one named probe from probes.py. The allowlist is the probe registry itself — the
    control plane can ask for a reading, never for arbitrary code."""
    import probes
    name = payload.get("name") or ""
    if name not in probes.PROBES:
        return {"error": f"Unknown probe '{name}'."}
    return probes.run_probe(name, **(payload.get("kwargs") or {}))


def do_snapshot(_payload: dict) -> dict:
    import probes
    return probes.snapshot()


HANDLERS = {
    "run_command": do_run_command,
    "read_file": do_read_file,
    "probe": do_probe,
    "snapshot": do_snapshot,
    "vitals": lambda _p: collect_vitals(),
    "check": lambda _p: {"ok": True, **local_facts()},
}


# --- main loop ------------------------------------------------------------------------------------

def enrol(client, server: str, token: str) -> str:
    resp = client.post(f"{server}/api/agent/enrol",
                       json={"enrol_token": token, **local_facts()}, timeout=20)
    if resp.status_code != 200:
        detail = resp.json().get("detail", resp.text) if resp.text else resp.status_code
        raise SystemExit(f"Enrolment failed: {detail}")
    key = resp.json()["agent_key"]
    save_key(key)
    _log(f"Enrolled successfully. Key saved to {KEY_FILE}")
    return key


def run(server: str, token: str | None) -> None:
    import httpx

    server = server.rstrip("/")
    with httpx.Client() as client:
        key = load_key()
        if token:
            key = enrol(client, server, token)
        if not key:
            raise SystemExit(
                "No agent key found and no --token supplied. Create a host in the dashboard "
                "and run this with the enrolment token it gives you."
            )

        headers = {"Authorization": f"Bearer {key}"}
        _log(f"Polling {server} every {POLL_INTERVAL_SECONDS}s. Ctrl+C to stop.")
        backoff = POLL_INTERVAL_SECONDS
        while True:
            try:
                resp = client.get(f"{server}/api/agent/jobs", headers=headers, timeout=20)
                if resp.status_code == 401:
                    raise SystemExit("Agent key rejected. Re-enrol with a fresh --token.")
                resp.raise_for_status()
                job = resp.json().get("job")
                backoff = POLL_INTERVAL_SECONDS
            except SystemExit:
                raise
            except Exception as exc:
                # The control plane being unreachable is normal (restarts, flaky links) and
                # must never kill the agent — back off instead, so a fleet reconnecting after
                # an outage does not arrive as a thundering herd.
                _log(f"Poll failed ({exc}); retrying in {backoff}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, POLL_BACKOFF_MAX_SECONDS)
                continue

            if not job:
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            _log(f"Job {job['id']}: {job['kind']}")
            handler = HANDLERS.get(job["kind"])
            if handler is None:
                result, status = {"error": f"Unknown job kind '{job['kind']}'."}, "failed"
            else:
                try:
                    result = handler(job.get("payload") or {})
                    status = "failed" if "error" in result else "done"
                except Exception as exc:
                    result, status = {"error": str(exc)}, "failed"
            try:
                client.post(f"{server}/api/agent/jobs/result", headers=headers,
                            json={"job_id": job["id"], "status": status, "result": result},
                            timeout=20)
            except Exception as exc:
                _log(f"Could not post result for {job['id']}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aurora Ops host agent")
    parser.add_argument("--server", required=True, help="Control plane base URL")
    parser.add_argument("--token", help="One-time enrolment token (first run only)")
    args = parser.parse_args()
    try:
        run(args.server, args.token)
    except KeyboardInterrupt:
        _log("Stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
