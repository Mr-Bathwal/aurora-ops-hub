"""How a tool actually reaches the machine it is inspecting.

This is the seam the whole multi-host story hangs on. Today `agents.py` calls `psutil` in
process, which silently hard-codes an assumption: the box being inspected is the box running
the code. That is fine for one machine and impossible for a customer's fleet.

A Transport answers three questions for *some* host — run this command, read this file, give
me vitals — and there are three implementations:

  LocalTransport   subprocess + psutil, i.e. exactly what the code does now.
  SSHTransport     the same commands over paramiko, with credentials decrypted per call.
  AgentTransport   queue a job and wait; the agent polls, executes locally, and posts back.

The asymmetry is deliberate. Local and SSH are *pull* — we act, and block until the machine
answers. Agent is *push-pull* — we leave work on a queue and the daemon collects it, which is
the only one of the three that survives NAT and an outbound-only firewall.

Vitals are returned in one shape regardless of transport, so everything upstream stays
transport-blind. That is the point of the abstraction: `agents.py` should never learn which
kind of host it is talking to.
"""

import json
import shlex
import time
import uuid

from db import get_conn, tx, utcnow

JOB_WAIT_TIMEOUT_SECONDS = 45
JOB_POLL_INTERVAL_SECONDS = 0.4


class TransportError(RuntimeError):
    """Reaching the host failed — distinct from the host answering with bad news."""


class Transport:
    kind = "unknown"

    def run_command(self, argv: list[str], timeout: int = 20) -> str:
        raise NotImplementedError

    def read_text_file(self, path: str, max_bytes: int = 200_000) -> str:
        raise NotImplementedError

    def vitals(self) -> dict:
        raise NotImplementedError

    def probe(self, name: str, **kwargs) -> dict:
        """Run one named probe from `probes.PROBES` on this host and return its dict."""
        raise NotImplementedError

    def snapshot(self) -> dict:
        """Every cheap probe in a single round trip.

        The reason this exists rather than looping over `probe()`: against an agent, each call
        is a queue write plus a poll cycle. Twenty-one of those is most of a minute and twenty
        one chances to time out. One snapshot is one round trip."""
        raise NotImplementedError

    def check(self) -> dict:
        """Prove the host is reachable and report what it is. Used by the 'Test connection'
        button, and by enrolment to fill in OS facts."""
        raise NotImplementedError


# --- local -----------------------------------------------------------------------------------

class LocalTransport(Transport):
    kind = "local"

    def run_command(self, argv: list[str], timeout: int = 20) -> str:
        import subprocess
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=timeout, shell=False)
        except FileNotFoundError:
            raise TransportError(f"Command '{argv[0]}' is not available on this system.")
        except subprocess.TimeoutExpired:
            raise TransportError(f"'{argv[0]}' timed out after {timeout}s.")
        return ((proc.stdout or "") + (proc.stderr or "")).strip()

    def read_text_file(self, path: str, max_bytes: int = 200_000) -> str:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read(max_bytes)

    def vitals(self) -> dict:
        import probes
        return probes.vitals()

    def probe(self, name: str, **kwargs) -> dict:
        import probes
        return probes.run_probe(name, **kwargs)

    def snapshot(self) -> dict:
        import probes
        return probes.snapshot()

    def check(self) -> dict:
        import probes
        return {"ok": True, **probes.facts()}


# --- ssh -------------------------------------------------------------------------------------

class SSHTransport(Transport):
    """Connects out per call rather than holding a pool.

    A long-lived SSH connection per host would be faster, but it also means an open shell
    session sitting in memory for every customer machine indefinitely. Reconnecting costs a
    few hundred milliseconds against agent runs measured in seconds, and it keeps the window
    in which a decrypted credential exists as short as possible."""

    kind = "ssh"

    def __init__(self, host_id: str):
        self.host_id = host_id

    def _client(self):
        import paramiko
        from hosts import get_ssh_credentials
        creds = get_ssh_credentials(self.host_id)
        client = paramiko.SSHClient()
        # AutoAddPolicy trusts an unknown host key on first contact. Acceptable for a
        # control plane that owns its inventory; a hardened build should pin the key at
        # enrolment and refuse a change, since silently accepting a new key is exactly
        # what a man-in-the-middle needs.
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        kwargs = {
            "hostname": creds["address"],
            "port": creds["port"],
            "username": creds["username"],
            "timeout": 12,
            "allow_agent": False,
            "look_for_keys": False,
        }
        if creds["auth_method"] == "password":
            kwargs["password"] = creds["secret"]
        else:
            import io
            key_body = io.StringIO(creds["secret"])
            pkey = None
            for loader in (paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey):
                try:
                    key_body.seek(0)
                    pkey = loader.from_private_key(key_body)
                    break
                except Exception:
                    continue
            if pkey is None:
                raise TransportError("Stored private key could not be parsed "
                                     "(expected OpenSSH Ed25519, RSA or ECDSA).")
            kwargs["pkey"] = pkey
        try:
            client.connect(**kwargs)
        except Exception as exc:
            raise TransportError(f"SSH connection to {creds['address']} failed: {exc}")
        return client

    def run_command(self, argv: list[str], timeout: int = 20) -> str:
        client = self._client()
        try:
            # shlex.join, never a bare join — an argument containing a space or a semicolon
            # would otherwise become extra commands on the remote shell.
            cmd = shlex.join(argv)
            _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
            out = stdout.read().decode("utf-8", "replace")
            err = stderr.read().decode("utf-8", "replace")
            return (out + err).strip()
        finally:
            client.close()

    def read_text_file(self, path: str, max_bytes: int = 200_000) -> str:
        client = self._client()
        try:
            sftp = client.open_sftp()
            try:
                with sftp.open(path, "r") as fh:
                    return fh.read(max_bytes).decode("utf-8", "replace")
            finally:
                sftp.close()
        finally:
            client.close()

    def vitals(self) -> dict:
        """Parsed from standard POSIX output rather than psutil, which is not installed on
        the target and should not have to be — an agentless transport that requires software
        on the target is not agentless."""
        raw = self.run_command(["sh", "-c",
                                "top -bn1 | head -3; free -m | head -2; df -P / | tail -1"])
        cpu = mem = disk = 0.0
        for line in raw.splitlines():
            low = line.lower()
            if "%cpu" in low or low.startswith("%cpu"):
                for part in line.replace(",", " ").split():
                    if part.replace(".", "").isdigit():
                        idle_next = "id" in line.split(part)[-1][:4]
                        if idle_next:
                            cpu = round(100.0 - float(part), 1)
                            break
            elif low.startswith("mem:"):
                nums = [p for p in line.split() if p.replace(".", "").isdigit()]
                if len(nums) >= 2 and float(nums[0]):
                    mem = round(float(nums[1]) / float(nums[0]) * 100, 1)
            elif line.strip().endswith("/"):
                for part in line.split():
                    if part.endswith("%"):
                        disk = float(part.rstrip("%"))
        return {"cpu": cpu, "memory": mem, "disk": disk}

    _REMOTE_PROBE_PATH = "/tmp/.itops_probes.py"

    def _run_remote_probes(self, arg: str) -> dict:
        """Copy probes.py to the target and execute it there.

        The alternative — reimplementing every probe as shell parsing — means two versions of
        "what is memory usage" that will eventually disagree, and the SSH one would be the
        worse of the two. Shipping the same file costs a few KB per call and keeps exactly one
        definition of every metric across local, agent and SSH.

        Requires python3 and psutil on the target. That is a real prerequisite and is reported
        plainly rather than silently degrading to worse numbers."""
        import json
        import os

        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "probes.py")
        client = self._client()
        try:
            sftp = client.open_sftp()
            try:
                sftp.put(local_path, self._REMOTE_PROBE_PATH)
                sftp.chmod(self._REMOTE_PROBE_PATH, 0o600)
            finally:
                sftp.close()
            cmd = f"python3 {shlex.quote(self._REMOTE_PROBE_PATH)} {shlex.quote(arg)}"
            _, stdout, stderr = client.exec_command(cmd, timeout=60)
            out = stdout.read().decode("utf-8", "replace").strip()
            err = stderr.read().decode("utf-8", "replace").strip()
        finally:
            client.close()

        if not out:
            if "ModuleNotFoundError" in err and "psutil" in err:
                raise TransportError(
                    "The target has python3 but not psutil. Install it there "
                    "(`pip3 install psutil`), or enrol this host with the agent instead."
                )
            if "not found" in err.lower():
                raise TransportError(
                    "python3 was not found on the target. Agentless health checks need "
                    "python3 and psutil; the installed agent has no such requirement."
                )
            raise TransportError(f"Remote probe produced no output. stderr: {err[:300]}")
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            raise TransportError(f"Remote probe returned unparseable output: {out[:300]}")

    def probe(self, name: str, **kwargs) -> dict:
        if kwargs:
            # The standalone runner takes a probe name only. Anything parameterised has to go
            # through the agent, and saying so beats returning a default the caller did not ask
            # for and cannot tell apart from the real answer.
            return {"error": "Parameterised probes are not supported over SSH; "
                             "enrol this host with the agent."}
        return self._run_remote_probes(name)

    def snapshot(self) -> dict:
        return self._run_remote_probes("snapshot")

    def check(self) -> dict:
        out = self.run_command(["sh", "-c", "uname -s; uname -r; hostname"])
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        return {
            "ok": bool(lines),
            "os_family": lines[0] if len(lines) > 0 else None,
            "os_version": lines[1] if len(lines) > 1 else None,
            "hostname": lines[2] if len(lines) > 2 else None,
        }


# --- agent ------------------------------------------------------------------------------------

class AgentTransport(Transport):
    """Queues work for a daemon that polls us.

    Every method here is the same shape: write a job, block until the agent writes a result or
    the clock runs out. The blocking wait is what lets a request-response API sit on top of a
    fundamentally asynchronous channel — and the timeout is what stops a request hanging
    forever when the machine is simply switched off."""

    kind = "agent"

    def __init__(self, host_id: str):
        self.host_id = host_id

    def _dispatch(self, kind: str, payload: dict, timeout: int) -> dict:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        with tx() as c:
            c.execute(
                "INSERT INTO jobs (id, host_id, kind, payload_json, status, created_at) "
                "VALUES (?,?,?,?,'queued',?)",
                (job_id, self.host_id, kind, json.dumps(payload), utcnow()),
            )
        deadline = time.time() + timeout
        conn = get_conn()
        while time.time() < deadline:
            row = conn.execute(
                "SELECT status, result_json FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row and row["status"] in ("done", "failed"):
                result = json.loads(row["result_json"] or "{}")
                if row["status"] == "failed":
                    raise TransportError(result.get("error", "The agent reported a failure."))
                return result
            time.sleep(JOB_POLL_INTERVAL_SECONDS)
        with tx() as c:
            c.execute("UPDATE jobs SET status = 'expired' WHERE id = ? AND status = 'queued'",
                      (job_id,))
        raise TransportError(
            f"The agent on this host did not respond within {timeout}s. "
            "It may be stopped, or the machine may be offline."
        )

    def run_command(self, argv: list[str], timeout: int = 20) -> str:
        return self._dispatch("run_command", {"argv": argv, "timeout": timeout},
                              JOB_WAIT_TIMEOUT_SECONDS).get("output", "")

    def read_text_file(self, path: str, max_bytes: int = 200_000) -> str:
        return self._dispatch("read_file", {"path": path, "max_bytes": max_bytes},
                              JOB_WAIT_TIMEOUT_SECONDS).get("content", "")

    def vitals(self) -> dict:
        return self._dispatch("vitals", {}, JOB_WAIT_TIMEOUT_SECONDS)

    def probe(self, name: str, **kwargs) -> dict:
        return self._dispatch("probe", {"name": name, "kwargs": kwargs},
                              JOB_WAIT_TIMEOUT_SECONDS)

    def snapshot(self) -> dict:
        # A longer budget than a single probe: the agent runs sixteen collectors, and
        # `cpu` alone blocks half a second by design to produce a meaningful average.
        return self._dispatch("snapshot", {}, JOB_WAIT_TIMEOUT_SECONDS + 30)

    def check(self) -> dict:
        return self._dispatch("check", {}, 15)


# --- factory -------------------------------------------------------------------------------------

def transport_for(host: dict) -> Transport:
    kind = host["connection_type"]
    if kind == "local":
        return LocalTransport()
    if kind == "ssh":
        return SSHTransport(host["id"])
    if kind == "agent":
        return AgentTransport(host["id"])
    raise TransportError(f"Unknown connection type '{kind}'.")
