# Aurora Ops — Hub

The backend for Aurora Ops: three specialist LLM agents that inspect a machine, an
orchestrator that routes a plain-English request to the right one, and a chain that
diagnoses, decides, acts, and then verifies its own work.

FastAPI + LangGraph + LangChain, on Groq. The console that drives it lives in
[`it-ops-frontend`](../it-ops-frontend).

## What it does

| Agent | Tools | Can it change anything? |
|---|---|---|
| **System Health** | 18 — CPU, memory, disk, swap, network, processes, services, ports, and a fixed 13-command Windows diagnostic menu | No. Read-only |
| **Log Analyzer** | 7 — read, tail, count levels, search patterns, time ranges | No. Read-only |
| **Backup & DR** | 8 — create, verify integrity, restore, clean up, check DR status | **Yes** — the only agent that acts |

Two LangGraph state machines sit on top:

- **`orchestrator.py`** — an LLM router reads your request and picks one of the three.
- **`auto_ops.py`** — diagnose → decide whether remediation is needed → act → **verify**.
  The verify node runs no tools; it reviews what happened and states whether the issue
  is actually resolved.

## Running it

```bash
python -m venv venv
venv\Scripts\activate          # Windows;  source venv/bin/activate on macOS/Linux
pip install -r requirements.txt

cp .env.example .env           # then put your Groq key in it
uvicorn api:app --reload --port 8000
```

The database is created on first run. There is nothing to migrate.

## Reaching machines

`transport.py` is the seam that makes this about a fleet rather than one laptop. Three
implementations, and the only difference is who opens the connection:

| Type | Who connects | Install needed | Notes |
|---|---|---|---|
| `local` | Nobody — same machine | No | psutil in-process |
| `ssh` | **We** connect out to them | No | Needs a route in; we hold a credential that opens a shell |
| `agent` | **They** connect out to us | Yes, one small daemon | Crosses NAT and outbound-only firewalls |

The agent enrols once with a short-lived, single-use token, receives a long-lived key,
then polls for work every 3 seconds. Nothing inbound is ever opened.

## Security posture

What is done properly:

- Passwords hashed with scrypt at double the OWASP floor. Only the hash is stored.
- Session tokens, enrolment tokens and agent keys are all stored as SHA-256 digests —
  the server never holds the value it checks against.
- Enrolment tokens expire in 60 minutes and are destroyed on first use.
- SSH credentials encrypted at rest with Fernet; plaintext exists only in memory for the
  length of a connection.
- LLM-generated code runs through an AST validator that denies by default, with a
  restricted global namespace and a 10-second timeout.
- Log reads are path-guarded — no absolute paths, no `..` escapes, and filenames
  containing `env`, `secret`, `password`, `token` or `key` are refused. Without this a
  plain `read_log_file('.env')` would hand the API key to the model.

**Known gaps — do not deploy as-is:**

- If `ITOPS_SECRET_KEY` is unset, credential encryption falls back to a fixed development
  key. The health endpoint reports which is in use, but nothing refuses to start.
- The SSH transport uses `AutoAddPolicy`, trusting an unknown host key on first contact.
  A hardened build should pin the key at enrolment and refuse a change.
- SQLite, single process. Fine for one operator; not a multi-tenant control plane.
- The diagnostic command menu is Windows-only.
