import ast
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import threading
import types as _types
import psutil
from datetime import datetime, timedelta
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.tools import tool, StructuredTool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

load_dotenv()
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ---------- TOOLS the agent is allowed to use ----------

@tool
def get_cpu_usage() -> str:
    """Returns the current CPU usage as a percentage, plus per-core breakdown."""
    overall = psutil.cpu_percent(interval=1)
    per_core = psutil.cpu_percent(interval=None, percpu=True)
    core_str = ", ".join(f"Core{i}: {v}%" for i, v in enumerate(per_core))
    return f"CPU usage: {overall}% overall | Per-core: [{core_str}]"

@tool
def get_memory_usage() -> str:
    """Returns the current memory (RAM) usage."""
    mem = psutil.virtual_memory()
    return f"Memory usage: {mem.percent}% ({mem.used // (1024**3)} GB of {mem.total // (1024**3)} GB used)"

@tool
def get_disk_usage() -> str:
    """Returns the current disk usage of the main drive."""
    disk = psutil.disk_usage("C:\\")   # use "/" instead of "C:\\" on Mac/Linux
    return f"Disk usage: {disk.percent}% ({disk.used // (1024**3)} GB of {disk.total // (1024**3)} GB used)"

@tool
def get_swap_usage() -> str:
    """Returns swap / virtual memory usage. Swap is overflow space used when RAM is full."""
    swap = psutil.swap_memory()
    if swap.total == 0:
        return "Swap: not configured on this system."
    return (f"Swap usage: {swap.percent}% "
            f"({swap.used // (1024**2)} MB of {swap.total // (1024**2)} MB used)")

@tool
def get_network_stats() -> str:
    """Returns network traffic stats: bytes sent/received and any errors or dropped packets."""
    net = psutil.net_io_counters()
    mb_sent = net.bytes_sent / (1024**2)
    mb_recv = net.bytes_recv / (1024**2)
    return (f"Network — Sent: {mb_sent:.1f} MB | Received: {mb_recv:.1f} MB | "
            f"Errors (in/out): {net.errin}/{net.errout} | "
            f"Dropped packets (in/out): {net.dropin}/{net.dropout}")

@tool
def get_active_connections() -> str:
    """Returns the count of active network connections grouped by status (ESTABLISHED, LISTEN, etc.)."""
    try:
        conns = psutil.net_connections(kind="inet")
    except psutil.AccessDenied:
        return "Access denied — run as administrator to read connection list."
    from collections import Counter
    counts = Counter(c.status for c in conns if c.status)
    summary = ", ".join(f"{status}: {n}" for status, n in sorted(counts.items()))
    return f"Active network connections: {len(conns)} total | {summary}"

@tool
def get_top_processes() -> str:
    """Returns the top 5 processes consuming the most CPU and the top 5 consuming the most RAM."""
    procs = []
    for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
        try:
            procs.append(p.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    top_cpu = sorted(procs, key=lambda x: x["cpu_percent"] or 0, reverse=True)[:5]
    top_mem = sorted(procs, key=lambda x: x["memory_percent"] or 0, reverse=True)[:5]
    cpu_str = " | ".join(f"{p['name']} ({p['cpu_percent']}%)" for p in top_cpu)
    mem_str = " | ".join(f"{p['name']} ({p['memory_percent']:.1f}%)" for p in top_mem)
    return f"Top CPU: [{cpu_str}]\nTop RAM: [{mem_str}]"

@tool
def get_system_uptime() -> str:
    """Returns how long the computer has been running since its last restart."""
    boot_time = datetime.fromtimestamp(psutil.boot_time())
    uptime = datetime.now() - boot_time
    days = uptime.days
    hours, remainder = divmod(uptime.seconds, 3600)
    minutes = remainder // 60
    return (f"System uptime: {days}d {hours}h {minutes}m "
            f"(last boot: {boot_time.strftime('%Y-%m-%d %H:%M')})")

@tool
def get_disk_io() -> str:
    """Returns disk read/write activity — how much data is being read from and written to the disk."""
    io = psutil.disk_io_counters()
    if io is None:
        return "Disk I/O counters not available on this system."
    read_mb  = io.read_bytes  / (1024**2)
    write_mb = io.write_bytes / (1024**2)
    return (f"Disk I/O — Total read: {read_mb:.1f} MB | Total written: {write_mb:.1f} MB | "
            f"Read ops: {io.read_count} | Write ops: {io.write_count}")

@tool
def get_cpu_temperature() -> str:
    """Returns CPU temperature if the hardware sensor is accessible."""
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return "CPU temperature: sensor not available on this system (common on Windows without third-party drivers)."
        results = []
        for chip, entries in temps.items():
            for entry in entries:
                if entry.current:
                    results.append(f"{chip}/{entry.label or 'core'}: {entry.current}°C")
        return "CPU temperatures: " + (", ".join(results) if results else "no readings available")
    except AttributeError:
        return "CPU temperature: psutil.sensors_temperatures() not supported on this OS."

@tool
def get_running_processes_count() -> str:
    """Returns total number of running processes and threads on the system."""
    proc_count = len(psutil.pids())
    thread_count = sum(p.num_threads() for p in psutil.process_iter(["num_threads"])
                       if p.info.get("num_threads"))
    return f"Running processes: {proc_count} | Total threads: {thread_count}"

@tool
def check_service_status(service_name: str) -> str:
    """Checks whether a named Windows service is running. Input: the exact service name (e.g. 'wuauserv', 'Spooler')."""
    try:
        svc = psutil.win_service_get(service_name)
        info = svc.as_dict()
        return (f"Service '{service_name}': status={info['status']}, "
                f"start_type={info['start_type']}, "
                f"pid={info.get('pid', 'N/A')}")
    except AttributeError:
        return "Service checks are only supported on Windows."
    except psutil.NoSuchProcess:
        return f"Service '{service_name}' not found."
    except Exception as e:
        return f"Could not check service '{service_name}': {e}"

@tool
def get_battery_status() -> str:
    """Returns battery percentage, plugged-in status, and time remaining (laptops only)."""
    try:
        batt = psutil.sensors_battery()
    except AttributeError:
        return "Battery status: not supported on this system."
    if batt is None:
        return "Battery status: no battery detected (desktop system)."
    plugged = "plugged in" if batt.power_plugged else "on battery"
    if batt.secsleft in (psutil.POWER_TIME_UNLIMITED, psutil.POWER_TIME_UNKNOWN, -1, -2):
        remaining = "unknown"
    else:
        remaining = f"{batt.secsleft // 3600}h {(batt.secsleft % 3600) // 60}m remaining"
    return f"Battery: {batt.percent}% ({plugged}, {remaining})"

@tool
def get_logged_in_users() -> str:
    """Returns the list of users currently logged into the system and their session start times."""
    users = psutil.users()
    if not users:
        return "No logged-in users found."
    lines = [
        f"{u.name} on {u.terminal or 'console'} since "
        f"{datetime.fromtimestamp(u.started).strftime('%Y-%m-%d %H:%M')}"
        for u in users
    ]
    return "Logged-in users:\n" + "\n".join(lines)

@tool
def get_network_interfaces() -> str:
    """Returns every network interface (adapter) and its IP addresses, plus which ones are up."""
    addrs = psutil.net_if_addrs()
    stats = psutil.net_if_stats()
    lines = []
    for name, addr_list in addrs.items():
        state = stats[name].isup if name in stats else None
        state_str = "UP" if state else "DOWN" if state is False else "unknown"
        ips = [a.address for a in addr_list if a.family.name in ("AF_INET", "AF_INET6")]
        lines.append(f"{name} [{state_str}]: {', '.join(ips) if ips else 'no IP'}")
    return "Network interfaces:\n" + "\n".join(lines)

@tool
def get_listening_ports() -> str:
    """Returns all ports currently in LISTEN state and the process name/PID bound to each."""
    try:
        conns = psutil.net_connections(kind="inet")
    except psutil.AccessDenied:
        return "Access denied — run as administrator to read listening ports."
    listening = [c for c in conns if c.status == "LISTEN"]
    if not listening:
        return "No listening ports found."
    lines = []
    for c in listening:
        proc_name = "unknown"
        if c.pid:
            try:
                proc_name = psutil.Process(c.pid).name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        lines.append(f"{c.laddr.ip}:{c.laddr.port} — {proc_name} (pid {c.pid})")
    return f"Listening ports ({len(lines)}):\n" + "\n".join(lines)

# ---------- Fixed-allowlist diagnostic commands ----------
# Every entry is a real argv list (never a shell string), so there is no shell-injection
# surface even for the entries that splice in a caller-supplied, regex-validated argument.
_DIAG_ARG_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,64}$")

_DIAG_COMMANDS = {
    "ipconfig":        (["ipconfig", "/all"], False),
    "netstat":         (["netstat", "-ano"], False),
    "tasklist":        (["tasklist"], False),
    "systeminfo":      (["systeminfo"], False),
    "arp":             (["arp", "-a"], False),
    "route":           (["route", "print"], False),
    "driverquery":     (["driverquery"], False),
    "wifi_status":     (["netsh", "wlan", "show", "interfaces"], False),
    "defender_status": (["powershell", "-NoProfile", "-Command",
                          "Get-MpComputerStatus | Select-Object AntivirusEnabled,"
                          "RealTimeProtectionEnabled,AntivirusSignatureLastUpdated | Format-List"], False),
    "firewall_status": (["netsh", "advfirewall", "show", "allprofiles", "state"], False),
    "recent_errors":   (["powershell", "-NoProfile", "-Command",
                          "Get-WinEvent -LogName System -MaxEvents 5 -ErrorAction SilentlyContinue "
                          "| Select-Object TimeCreated,LevelDisplayName,Message | Format-List"], False),
    "ping":            (["ping", "-n", "2"], True),
    "nslookup":        (["nslookup"], True),
}

@tool
def run_diagnostic_command(command: str, argument: str = "") -> str:
    """Runs a fixed Windows diagnostic command. command: ipconfig, netstat, tasklist,
    systeminfo, arp, route, driverquery, wifi_status, defender_status, firewall_status,
    recent_errors, ping, nslookup. argument: hostname/IP, only for ping/nslookup."""
    entry = _DIAG_COMMANDS.get(command)
    if entry is None:
        return f"Unknown command '{command}'. Valid options: {', '.join(sorted(_DIAG_COMMANDS))}."
    argv, needs_arg = entry
    if needs_arg:
        if not argument or not _DIAG_ARG_RE.match(argument):
            return "This command requires a valid hostname/IP argument (letters, digits, '.', '-', '_' only)."
        argv = argv + [argument]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=15, shell=False)
    except FileNotFoundError:
        return f"Command '{argv[0]}' is not available on this system."
    except subprocess.TimeoutExpired:
        return f"'{command}' timed out after 15 seconds."
    output = (proc.stdout or "") + (proc.stderr or "")
    return output.strip()[:4000] or f"'{command}' produced no output (exit code {proc.returncode})."

@tool
def get_disk_partitions() -> str:
    """Lists all disk partitions/drives and their usage (useful when there are multiple drives)."""
    parts = psutil.disk_partitions(all=False)
    lines = []
    for p in parts:
        try:
            usage = psutil.disk_usage(p.mountpoint)
            lines.append(
                f"{p.device} ({p.fstype}) at {p.mountpoint}: "
                f"{usage.percent}% used — {usage.used // (1024**3)} GB / {usage.total // (1024**3)} GB"
            )
        except PermissionError:
            lines.append(f"{p.device}: permission denied (likely CD/removable drive)")
    return "Disk partitions:\n" + "\n".join(lines)

# ---------- Self-expanding custom check tools ----------
#
# Defense in depth, in order:
#  1. AST validation (_validate_ast) — the real boundary. Rejects imports, defs, dunder
#     attribute access (blocks the classic `().__class__.__base__.__subclasses__()` sandbox
#     escape) and any name not on an explicit allow-list, *before* the code ever runs.
#  2. A restricted __builtins__ dict with no exec/eval/compile/open/getattr/__import__ — even
#     if something slipped past step 1, the names simply aren't there to call.
#  3. `os` is replaced by a read-only proxy exposing only inspection helpers (no os.system,
#     os.remove, etc. — those attributes don't exist on the proxy at all, so there is nothing
#     for a bypass to call). os.environ is exposed as a *copy* with anything that looks like a
#     secret (KEY/SECRET/TOKEN/PASSWORD/CREDENTIAL) redacted, since the real process env holds
#     GROQ_API_KEY.
#  4. A short substring pre-filter catches obvious junk cheaply before we even parse.
#  5. A daemon-thread timeout bounds runaway loops.
# None of these layers is sufficient alone (see the research this was based on — substring
# blocklists and RestrictedPython-style checks are routinely bypassed on their own); together
# they're the right amount of isolation for a single-user local tool. True untrusted/multi-tenant
# execution would need a real container/VM sandbox (Docker, E2B, gVisor) instead.

import builtins as _builtins_mod
import collections as _collections

_BLOCKED = [
    "open(", "__import__(", "subprocess", "os.system", "os.popen",
    "shutil", "requests", "urllib", "http.client", "socket.connect",
    "socket.bind", "eval(", "compile(", "exec(",
]

_ALLOWED_BUILTINS = {
    "len", "str", "int", "float", "bool", "list", "dict", "tuple", "set",
    "sorted", "reversed", "round", "min", "max", "sum", "abs",
    "range", "enumerate", "zip", "isinstance", "type", "print",
    "format", "repr", "any", "all",
}
_SAFE_BUILTINS = {k: getattr(_builtins_mod, k) for k in _ALLOWED_BUILTINS}

_SECRET_ENV_MARKERS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "PWD", "CREDENTIAL")


class _SafeOS:
    """Read-only subset of the os module — inspection helpers only, nothing that writes,
    deletes, spawns processes, or touches the raw (secret-bearing) environment."""
    path = os.path
    sep = os.sep
    linesep = os.linesep
    name = os.name
    cpu_count = staticmethod(os.cpu_count)
    getcwd = staticmethod(os.getcwd)
    listdir = staticmethod(os.listdir)
    getlogin = staticmethod(os.getlogin)
    walk = staticmethod(os.walk)

    @property
    def environ(self) -> dict:
        return {
            k: v for k, v in os.environ.items()
            if not any(marker in k.upper() for marker in _SECRET_ENV_MARKERS)
        }


_safe_os = _SafeOS()

_SAFE_GLOBALS: dict = {
    "__builtins__": _SAFE_BUILTINS,
    "psutil":   psutil,
    "os":       _safe_os,
    "platform": platform,
    "datetime": datetime,
    "re":       re,
    "Counter":  _collections.Counter,
}

# Names that must never resolve inside a snippet even though they aren't builtins we handed
# out — catches attempts to reach them via comprehension tricks or shadowed imports.
_BLOCKED_NAMES = {
    "exec", "eval", "compile", "open", "__import__", "input", "getattr", "setattr",
    "delattr", "vars", "globals", "locals", "dir", "help", "breakpoint", "exit", "quit",
    "memoryview", "__builtins__",
}

_ALLOWED_AST_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.AugAssign, ast.AnnAssign,
    ast.Load, ast.Store, ast.Name, ast.Constant, ast.Call, ast.Attribute,
    ast.Subscript, ast.Slice,
    ast.List, ast.Tuple, ast.Dict, ast.Set,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.And, ast.Or, ast.Not, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn, ast.Is, ast.IsNot, ast.UAdd, ast.USub,
    ast.If, ast.IfExp, ast.For, ast.While, ast.Break, ast.Continue, ast.Pass,
    ast.comprehension, ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp,
    ast.JoinedStr, ast.FormattedValue, ast.Starred, ast.keyword, ast.Try, ast.ExceptHandler,
)


def _validate_ast(code: str) -> str | None:
    """Returns an error string if the snippet does anything outside the allowed subset,
    or None if it's clean. This is the actual security boundary, not the substring list."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return f"syntax error: {exc}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "imports are not allowed"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return "defining functions/classes/lambdas is not allowed"
        if isinstance(node, (ast.With, ast.AsyncWith)):
            return "'with' statements are not allowed"
        if isinstance(node, (ast.Global, ast.Nonlocal, ast.Delete)):
            return "global/nonlocal/del are not allowed"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"access to dunder attribute '{node.attr}' is not allowed"
        if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
            return f"use of '{node.id}' is not allowed"
        if not isinstance(node, _ALLOWED_AST_NODES):
            return f"disallowed syntax: {type(node).__name__}"
    return None


def _exec_in(code: str, sandbox_globals: dict, blocked: list, timeout: int = 10) -> str:
    """Run code string safely against a *specific* domain's globals/blocklist: substring
    pre-filter, AST validation, restricted globals, timeout. Shared by every domain's sandbox
    so the security boundary only has one implementation to keep correct."""
    for pat in blocked:
        if pat in code:
            return f"Safety check failed: '{pat}' is not allowed."
    ast_error = _validate_ast(code)
    if ast_error:
        return f"Safety check failed: {ast_error}."
    box: dict = {"result": None, "error": None}

    def _run() -> None:
        try:
            local: dict = {}
            exec(code, dict(sandbox_globals), local)  # copy so globals stay clean
            box["result"] = local.get(
                "result",
                "Code ran but no 'result' variable was set. Add: result = '…' at the end.",
            )
        except Exception as exc:
            box["error"] = str(exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        return "Custom check timed out after 10 seconds."
    if box["error"]:
        return f"Custom check error: {box['error']}"
    return str(box["result"])


_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_TOOL_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _make_custom_tool_kit(
    file_name: str,
    sandbox_globals: dict,
    reserved_names: set,
    available_desc: str,
    extra_blocked: list = (),
    run_name: str = "run_custom_check",
    save_name: str = "save_custom_tool",
    list_name: str = "list_custom_tools",
):
    """Builds one domain's self-expanding trio (run/save/list) plus a startup loader, all backed
    by their own JSON file and their own sandbox globals — but sharing the same _exec_in security
    boundary as every other domain. This is what lets each agent (System Health, Log Analyzer,
    Backup & DR) grow its own tool library independently instead of one shared, ever-growing bag."""
    custom_file = os.path.join(_PROJECT_ROOT, file_name)
    blocked = _BLOCKED + list(extra_blocked)

    def run_custom_check(python_code: str) -> str:
        return _exec_in(python_code, sandbox_globals, blocked)
    run_custom_check.__doc__ = (
        f"Execute a custom Python check when no built-in tool covers the request. "
        f"{available_desc} are already available as globals — do NOT write import statements, "
        f"they will be rejected. No dunder attributes (__class__ etc). The snippet MUST store "
        f"its answer in a variable named 'result'."
    )

    def save_custom_tool(name: str, description: str, code: str) -> str:
        if not _TOOL_NAME_RE.match(name):
            return "Cannot save: name must be short_snake_case (letters, digits, underscore, starting with a letter)."
        if name in reserved_names:
            return f"Cannot save: '{name}' is already a built-in tool name — pick a different name."
        for pat in blocked:
            if pat in code:
                return f"Cannot save: '{pat}' is not allowed."
        ast_error = _validate_ast(code)
        if ast_error:
            return f"Cannot save: {ast_error}."
        existing: dict = {}
        if os.path.exists(custom_file):
            try:
                with open(custom_file, "r", encoding="utf-8") as fh:
                    existing = json.load(fh)
            except Exception:
                pass
        existing[name] = {"description": description, "code": code}
        try:
            with open(custom_file, "w", encoding="utf-8") as fh:
                json.dump(existing, fh, indent=2)
        except Exception as exc:
            return f"Failed to save: {exc}"
        return f"Custom tool '{name}' saved. It will auto-load on the next server restart."
    save_custom_tool.__doc__ = (
        "Permanently save a new check so it auto-loads on every future server start. "
        "name: short_snake_case (e.g. 'check_port_8080'), must not clash with a built-in tool "
        "name. description: one sentence. code: same rules as the run tool, must set 'result'."
    )

    def list_custom_tools() -> str:
        if not os.path.exists(custom_file):
            return "No custom tools saved yet."
        try:
            with open(custom_file, "r", encoding="utf-8") as fh:
                saved: dict = json.load(fh)
        except Exception as exc:
            return f"Could not read custom tools file: {exc}"
        if not saved:
            return "No custom tools saved yet."
        lines = [f"Saved custom tools ({len(saved)} total):"]
        for n, info in saved.items():
            lines.append(f"  • {n}: {info['description']}")
        return "\n".join(lines)
    list_custom_tools.__doc__ = "List all permanently saved custom tools in this domain."

    def load_startup_tools() -> list:
        """At startup: read this domain's JSON file and return LangChain Tool objects."""
        if not os.path.exists(custom_file):
            return []
        try:
            with open(custom_file, "r", encoding="utf-8") as fh:
                saved: dict = json.load(fh)
        except Exception:
            return []
        loaded = []
        for tool_name, info in saved.items():
            code = info["code"]
            desc = info["description"]

            def _make(c: str):
                def _fn() -> str:
                    return _exec_in(c, sandbox_globals, blocked)
                return _fn

            loaded.append(StructuredTool.from_function(func=_make(code), name=tool_name, description=desc))
        return loaded

    run_tool = StructuredTool.from_function(func=run_custom_check, name=run_name, description=run_custom_check.__doc__)
    save_tool = StructuredTool.from_function(func=save_custom_tool, name=save_name, description=save_custom_tool.__doc__)
    list_tool = StructuredTool.from_function(func=list_custom_tools, name=list_name, description=list_custom_tools.__doc__)
    return run_tool, save_tool, list_tool, load_startup_tools, custom_file


run_custom_check, save_custom_tool, list_custom_tools, _load_custom_tools, _CUSTOM_TOOLS_FILE = _make_custom_tool_kit(
    file_name="custom_tools.json",
    sandbox_globals=_SAFE_GLOBALS,
    reserved_names={
        "get_cpu_usage", "get_memory_usage", "get_disk_usage", "get_swap_usage",
        "get_network_stats", "get_active_connections", "get_top_processes",
        "get_system_uptime", "get_disk_io", "get_cpu_temperature",
        "get_running_processes_count", "check_service_status", "get_battery_status",
        "get_logged_in_users", "get_network_interfaces", "get_listening_ports",
        "run_diagnostic_command", "get_disk_partitions", "run_custom_check",
        "save_custom_tool", "list_custom_tools",
    },
    available_desc="psutil, os, platform, datetime, re, Counter",
)

# ---------- The System Health agent ----------
_startup_custom_tools = _load_custom_tools()

health_tools = [
    get_cpu_usage,
    get_memory_usage,
    get_disk_usage,
    get_swap_usage,
    get_network_stats,
    get_active_connections,
    get_top_processes,
    get_system_uptime,
    get_disk_io,
    get_cpu_temperature,
    get_running_processes_count,
    check_service_status,
    get_disk_partitions,
    get_battery_status,
    get_logged_in_users,
    get_network_interfaces,
    get_listening_ports,
    run_diagnostic_command,
    run_custom_check,
    save_custom_tool,
    list_custom_tools,
    *_startup_custom_tools,
]
system_health_agent = create_react_agent(llm, health_tools)

HEALTH_SYSTEM_PROMPT = (
    "You are a System Health monitoring agent. Never refuse a check for lack of a tool — "
    "pick in this order: "
    "1. A built-in tool, if one covers it (fastest, most reliable). "
    "2. run_diagnostic_command, for common Windows admin/network checks it lists. "
    "3. run_custom_check — write a psutil/os/platform/datetime/re/Counter snippet that sets "
    "'result'. This is the catch-all for anything not covered above. "
    "After a custom check proves useful and reusable (not a one-off with hardcoded specifics "
    "like a PID), call save_custom_tool right away — don't wait to be asked. Use "
    "list_custom_tools first if unsure whether an equivalent already exists. "
    "For a full health check run: cpu, memory, disk, swap, network_stats, uptime, top_processes. "
    "For a specific question, run only the relevant tool(s). "
    "Flag metrics above 85% as WARNING, above 95% as CRITICAL. "
    "End every report with overall status: HEALTHY / WARNING / CRITICAL."
)

def run_system_health(query: str = "Check the system health and give me a full report."):
    try:
        result = system_health_agent.invoke({
            "messages": [
                SystemMessage(content=HEALTH_SYSTEM_PROMPT),
                HumanMessage(content=query)
            ]
        })
    except Exception as exc:
        return f"The agent hit an error mid-run and couldn't finish this request: {exc}"
    return result["messages"][-1].content

# ---------- TOOLS for the Log Analyzer agent ----------
#
# Every log tool reads file *content* straight into the LLM's context — unlike the System
# Health sandbox (which never touches the filesystem), this one has to. So every read here goes
# through _safe_log_path: no absolute paths, no '..' escapes out of the project directory, and no
# filenames that look like secrets/config (env, key, secret, credential, password, token). Without
# this, a plain `read_log_file('.env')` would have handed GROQ_API_KEY straight to the model.
_LOG_ROOT = _PROJECT_ROOT
_SECRET_FILENAME_MARKERS = ("env", "secret", "credential", "password", "pwd", "token", "key")


def _safe_log_path(filename: str):
    """Resolves filename against the project root. Returns (path, None) or (None, error_message)."""
    if not filename or os.path.isabs(filename) or "\x00" in filename:
        return None, f"Access denied: '{filename}' must be a relative path within the project."
    candidate = os.path.normpath(os.path.join(_LOG_ROOT, filename))
    if candidate != _LOG_ROOT and not candidate.startswith(_LOG_ROOT + os.sep):
        return None, "Access denied: path escapes the project directory."
    base = os.path.basename(candidate).lower()
    stem = base.rsplit(".", 1)[0]
    if base.startswith(".") or any(m in stem for m in _SECRET_FILENAME_MARKERS):
        return None, f"Access denied: '{filename}' looks like a secret/config file, not a log."
    if not os.path.isfile(candidate):
        return None, f"File '{filename}' not found."
    return candidate, None


@tool
def read_log_file(filename: str) -> str:
    """Reads a log file and returns its most recent lines. Input is the log file name
    (relative to the project directory — no absolute paths, no '..')."""
    path, err = _safe_log_path(filename)
    if err:
        return err
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    return "".join(lines[-100:])   # keep only the last 100 lines so we don't overload the model

@tool
def list_log_files(directory: str = ".") -> str:
    """Lists log-like files (.log, .txt) in a directory under the project root, with size and
    last-modified time. Use this to discover what log files are actually available."""
    if os.path.isabs(directory) or ".." in directory.replace("\\", "/").split("/"):
        return "Access denied: directory must be relative, within the project."
    dir_path = os.path.normpath(os.path.join(_LOG_ROOT, directory))
    if dir_path != _LOG_ROOT and not dir_path.startswith(_LOG_ROOT + os.sep):
        return "Access denied: path escapes the project directory."
    if not os.path.isdir(dir_path):
        return f"Directory '{directory}' not found."
    entries = []
    for name in sorted(os.listdir(dir_path)):
        if not name.lower().endswith((".log", ".txt")):
            continue
        full = os.path.join(dir_path, name)
        if not os.path.isfile(full):
            continue
        size_kb = os.path.getsize(full) / 1024
        mtime = datetime.fromtimestamp(os.path.getmtime(full)).strftime("%Y-%m-%d %H:%M")
        entries.append(f"{name} ({size_kb:.1f} KB, modified {mtime})")
    if not entries:
        return f"No .log/.txt files found in '{directory}'."
    return "Log files:\n" + "\n".join(entries)

@tool
def tail_log_file(filename: str, lines: int = 100) -> str:
    """Reads the last N lines of a log file (default 100, max 500). Use this instead of
    read_log_file when you need more or fewer lines than the fixed default."""
    path, err = _safe_log_path(filename)
    if err:
        return err
    n = max(1, min(lines, 500))
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.readlines()
    return "".join(content[-n:])

@tool
def get_log_file_info(filename: str) -> str:
    """Returns size, last-modified time, and total line count for a log file."""
    path, err = _safe_log_path(filename)
    if err:
        return err
    size_kb = os.path.getsize(path) / 1024
    mtime = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        line_count = sum(1 for _ in f)
    return f"'{filename}': {size_kb:.1f} KB, {line_count} lines, last modified {mtime}."

@tool
def count_log_levels(filename: str) -> str:
    """Counts occurrences of common log severity levels (CRITICAL, ERROR, WARNING/WARN, INFO,
    DEBUG) in a log file. Good for a quick 'how bad is this log' summary."""
    path, err = _safe_log_path(filename)
    if err:
        return err
    counts = {"CRITICAL": 0, "ERROR": 0, "WARNING": 0, "INFO": 0, "DEBUG": 0}
    pattern = re.compile(r"\b(CRITICAL|ERROR|WARN(?:ING)?|INFO|DEBUG)\b", re.IGNORECASE)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                level = m.group(1).upper()
                level = "WARNING" if level.startswith("WARN") else level
                counts[level] += 1
    summary = ", ".join(f"{k}: {v}" for k, v in counts.items())
    return f"Log level counts for '{filename}': {summary}"

@tool
def search_log_pattern(filename: str, pattern: str, max_matches: int = 20) -> str:
    """Searches a log file for a regex pattern and returns matching lines with line numbers.
    Use this to find specific errors, IDs, IPs, or any text the built-in tools don't cover."""
    path, err = _safe_log_path(filename)
    if err:
        return err
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        return f"Invalid regex pattern: {exc}"
    matches = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f, start=1):
            if regex.search(line):
                matches.append(f"L{i}: {line.rstrip()}")
                if len(matches) >= max(1, min(max_matches, 100)):
                    break
    if not matches:
        return f"No matches for '{pattern}' in '{filename}'."
    return f"{len(matches)} match(es):\n" + "\n".join(matches)

@tool
def get_log_time_range(filename: str) -> str:
    """Finds the first and last timestamp present in a log file (recognizes common formats:
    ISO 8601, 'YYYY-MM-DD HH:MM:SS', syslog-style). Useful to know what time window a log covers."""
    path, err = _safe_log_path(filename)
    if err:
        return err
    ts_pattern = re.compile(
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}|"
        r"\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}"
    )
    first, last = None, None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = ts_pattern.search(line)
            if m:
                if first is None:
                    first = m.group(0)
                last = m.group(0)
    if first is None:
        return f"No recognizable timestamps found in '{filename}'."
    return f"'{filename}' spans from '{first}' to '{last}' (first/last timestamp seen)."

# ---------- Self-expanding custom check tools for the Log Analyzer ----------
# Unlike the System Health sandbox, this one needs file content — so instead of exposing raw
# open()/os file-read primitives (which would need their own path-safety checks re-implemented
# inside the AST-validated snippet, easy to get wrong), it exposes a single trusted helper,
# read_lines(), that already goes through _safe_log_path before touching disk. The snippet
# itself never gets an unrestricted file handle.
def _log_read_lines(filename: str, max_lines: int = 200):
    """Sandbox-only helper bound into custom log checks: reads up to max_lines from a log file,
    enforcing the same root/secret-file restrictions as the read_log_file tool."""
    path, err = _safe_log_path(filename)
    if err:
        return [err]
    n = max(1, min(max_lines, 1000))
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.readlines()[-n:]


_LOG_SAFE_GLOBALS: dict = {
    "__builtins__": _SAFE_BUILTINS,
    "os": _safe_os,
    "re": re,
    "datetime": datetime,
    "Counter": _collections.Counter,
    "read_lines": _log_read_lines,
}

(
    run_custom_log_check, save_custom_log_tool, list_custom_log_tools,
    _load_log_custom_tools, _LOG_CUSTOM_TOOLS_FILE,
) = _make_custom_tool_kit(
    file_name="custom_tools_log.json",
    sandbox_globals=_LOG_SAFE_GLOBALS,
    reserved_names={
        "read_log_file", "list_log_files", "tail_log_file", "get_log_file_info",
        "count_log_levels", "search_log_pattern", "get_log_time_range",
        "run_custom_log_check", "save_custom_log_tool", "list_custom_log_tools",
    },
    available_desc=(
        "os (read-only), re, datetime, Counter, and read_lines(filename, max_lines=200) "
        "for log file content"
    ),
    run_name="run_custom_log_check",
    save_name="save_custom_log_tool",
    list_name="list_custom_log_tools",
)
_startup_log_custom_tools = _load_log_custom_tools()

# ---------- The Log Analyzer agent ----------
log_tools = [
    read_log_file,
    list_log_files,
    tail_log_file,
    get_log_file_info,
    count_log_levels,
    search_log_pattern,
    get_log_time_range,
    run_custom_log_check,
    save_custom_log_tool,
    list_custom_log_tools,
    *_startup_log_custom_tools,
]
log_analyzer_agent = create_react_agent(llm, log_tools)

LOG_SYSTEM_PROMPT = (
    "You are a Log Analyzer agent. Never refuse a request for lack of a tool — pick in order: "
    "1. A built-in tool if one covers it (read_log_file, list_log_files, tail_log_file, "
    "get_log_file_info, count_log_levels, search_log_pattern, get_log_time_range). "
    "2. run_custom_log_check — the catch-all for anything not covered above (custom pattern "
    "extraction, cross-file correlation, time-window filtering, etc): write a snippet using "
    "os/re/datetime/Counter and the read_lines(filename) helper — never raw open/import — that "
    "sets 'result'. "
    "After a custom check proves useful and reusable, call save_custom_log_tool right away — "
    "don't wait to be asked. Use list_custom_log_tools first if unsure whether one already exists. "
    "Always count errors/warnings, list the most important issues in plain English, and suggest "
    "what to look into. Keep it concise."
)

def run_log_analyzer(filename="sample.log", request: str | None = None):
    """Analyse a log file. `request` is what the operator actually asked, when there is one.

    It is appended rather than replacing the instruction because the agent still has to do the
    same job — read the file, count the levels — and a free-form request substituted for the
    task would let 'is the disk ok?' produce a run that never opens the log at all. Given as
    context, it steers which findings get emphasised without changing what gets inspected.
    """
    task = f"Analyze the log file named '{filename}' and give me a summary."
    if request and request.strip():
        task += (
            f"\n\nThe operator described the problem this way: \"{request.strip()}\"\n"
            "Lead with whatever in the log speaks to that, then cover the rest. "
            "If nothing in the log relates to it, say so plainly rather than inventing a link."
        )
    try:
        result = log_analyzer_agent.invoke({
            "messages": [
                SystemMessage(content=LOG_SYSTEM_PROMPT),
                HumanMessage(content=task)
            ]
        })
    except Exception as exc:
        return f"The agent hit an error mid-run and couldn't finish this request: {exc}"
    return result["messages"][-1].content

# ---------- TOOLS for the Backup & DR agent ----------
#
# DR best practice (3-2-1-1-0: multiple copies, verified integrity, zero errors on restore test)
# is why create_backup now writes a checksummed manifest instead of just copying files — a
# backup nobody has verified is unrestorable is not a backup. It's also why the split below
# matters: creating/restoring/deleting backups are real, reviewed, fixed tools; nothing that
# writes, restores, or deletes ever goes through the self-expanding sandbox below — only
# read-only reporting does. Least-privilege agent design: separate the tools that can act from
# the tools that can only observe, and never let free-form generated code be the one taking the
# destructive action.
_BACKUPS_DIR = "backups"
_MANIFEST_NAME = "_manifest.json"


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _backup_dirs() -> list:
    if not os.path.isdir(_BACKUPS_DIR):
        return []
    return sorted(
        (d for d in os.listdir(_BACKUPS_DIR) if d.startswith("backup_")
         and os.path.isdir(os.path.join(_BACKUPS_DIR, d))),
        key=lambda d: os.path.getmtime(os.path.join(_BACKUPS_DIR, d)),
        reverse=True,
    )


@tool
def create_backup(source_folder: str) -> str:
    """Creates a timestamped backup copy of a folder into a 'backups' directory, plus a
    checksum manifest used later by verify_backup_integrity. Input is the source folder name."""
    if not os.path.isdir(source_folder):
        return f"Error: source folder '{source_folder}' not found."
    os.makedirs(_BACKUPS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")   # microseconds = always unique
    dest = os.path.join(_BACKUPS_DIR, f"backup_{timestamp}")
    try:
        shutil.copytree(source_folder, dest)
    except FileExistsError:
        return "A backup with this name already exists. Skipped to avoid overwriting."
    files = {}
    for root, _, filenames in os.walk(dest):
        for fn in filenames:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, dest)
            files[rel] = {"sha256": _hash_file(full), "size": os.path.getsize(full)}
    manifest = {
        "source": os.path.abspath(source_folder),
        "created": datetime.now().isoformat(),
        "files": files,
    }
    with open(os.path.join(dest, _MANIFEST_NAME), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return f"Backup created at '{dest}' with {len(files)} file(s), checksummed for later verification."

@tool
def check_dr_status() -> str:
    """Checks disaster-recovery status by finding the most recent backup and how old it is."""
    backups = _backup_dirs()
    if not backups:
        return "DR status: AT RISK — no backups exist yet."
    latest = backups[0]
    age_hours = (datetime.now().timestamp() - os.path.getmtime(os.path.join(_BACKUPS_DIR, latest))) / 3600
    status = "HEALTHY" if age_hours < 24 else "AT RISK (last backup over 24h old)"
    return f"Most recent backup: {latest} ({age_hours:.1f} hours ago). Total backups: {len(backups)}. DR status: {status}."

@tool
def list_backups() -> str:
    """Lists every backup with its age, size, and file count — newest first."""
    backups = _backup_dirs()
    if not backups:
        return "No backups exist yet."
    lines = []
    for name in backups:
        path = os.path.join(_BACKUPS_DIR, name)
        size_mb = sum(os.path.getsize(os.path.join(r, f))
                      for r, _, fs in os.walk(path) for f in fs) / (1024 ** 2)
        file_count = sum(len(fs) for _, _, fs in os.walk(path))
        age_hours = (datetime.now().timestamp() - os.path.getmtime(path)) / 3600
        lines.append(f"{name}: {size_mb:.1f} MB, {file_count} files, {age_hours:.1f}h old")
    return f"{len(backups)} backup(s):\n" + "\n".join(lines)

@tool
def get_backup_details(backup_name: str) -> str:
    """Returns the source folder, creation time, size, and file count for one specific backup."""
    path = os.path.join(_BACKUPS_DIR, backup_name)
    if not os.path.isdir(path):
        return f"Backup '{backup_name}' not found."
    manifest_path = os.path.join(path, _MANIFEST_NAME)
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        size_mb = sum(f["size"] for f in manifest["files"].values()) / (1024 ** 2)
        return (f"'{backup_name}': source='{manifest['source']}', created={manifest['created']}, "
                f"{len(manifest['files'])} files, {size_mb:.1f} MB, manifest present (verifiable).")
    file_count = sum(len(fs) for _, _, fs in os.walk(path))
    return f"'{backup_name}': {file_count} files, no manifest (created before integrity tracking — cannot verify)."

@tool
def verify_backup_integrity(backup_name: str) -> str:
    """Recomputes checksums for a backup and compares them against its manifest, catching
    corruption, missing files, or tampering. This is what makes 'we have a backup' meaningful."""
    path = os.path.join(_BACKUPS_DIR, backup_name)
    if not os.path.isdir(path):
        return f"Backup '{backup_name}' not found."
    manifest_path = os.path.join(path, _MANIFEST_NAME)
    if not os.path.isfile(manifest_path):
        return f"Cannot verify '{backup_name}': no manifest (created before integrity tracking)."
    with open(manifest_path, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    missing, corrupted = [], []
    on_disk = set()
    for root, _, filenames in os.walk(path):
        for fn in filenames:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, path)
            if rel == _MANIFEST_NAME:
                continue
            on_disk.add(rel)
    for rel, info in manifest["files"].items():
        if rel not in on_disk:
            missing.append(rel)
        elif _hash_file(os.path.join(path, rel)) != info["sha256"]:
            corrupted.append(rel)
    extra = on_disk - set(manifest["files"])
    if not missing and not corrupted:
        return f"'{backup_name}': VERIFIED — all {len(manifest['files'])} files intact." + (
            f" ({len(extra)} extra file(s) present, not part of the original backup.)" if extra else "")
    parts = [f"'{backup_name}': CORRUPTED."]
    if missing:
        parts.append(f"Missing ({len(missing)}): {', '.join(missing[:10])}")
    if corrupted:
        parts.append(f"Checksum mismatch ({len(corrupted)}): {', '.join(corrupted[:10])}")
    return " ".join(parts)

@tool
def restore_backup(backup_name: str, destination_folder: str) -> str:
    """Restores a backup's files into destination_folder. Refuses to overwrite an existing
    non-empty destination — restore to a fresh or empty folder to avoid clobbering current data."""
    src = os.path.join(_BACKUPS_DIR, backup_name)
    if not os.path.isdir(src):
        return f"Backup '{backup_name}' not found."
    if os.path.exists(destination_folder) and os.listdir(destination_folder):
        return f"Refusing to restore: '{destination_folder}' already exists and is not empty."
    shutil.copytree(
        src, destination_folder, dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(_MANIFEST_NAME),
    )
    file_count = sum(len(fs) for _, _, fs in os.walk(destination_folder))
    return f"Restored {file_count} file(s) from '{backup_name}' into '{destination_folder}'."

@tool
def cleanup_old_backups(keep_latest: int = 5) -> str:
    """Deletes the oldest backups beyond keep_latest, freeing disk space. Always keeps at least
    1 backup no matter what's requested — this can never wipe out every backup."""
    keep = max(1, keep_latest)
    backups = _backup_dirs()   # newest first
    to_delete = backups[keep:]
    if not to_delete:
        return f"Nothing to clean up — {len(backups)} backup(s), keeping {keep}."
    for name in to_delete:
        shutil.rmtree(os.path.join(_BACKUPS_DIR, name))
    return f"Deleted {len(to_delete)} old backup(s): {', '.join(to_delete)}. Kept the {keep} most recent."

@tool
def get_available_disk_space() -> str:
    """Returns free disk space, useful for checking there's room before creating a new backup."""
    usage = psutil.disk_usage(os.path.abspath("."))
    free_gb = usage.free / (1024 ** 3)
    return f"Free disk space: {free_gb:.1f} GB ({100 - usage.percent:.1f}% free, drive {usage.percent}% used)."

# ---------- Self-expanding custom check tools for Backup & DR ----------
# Strictly read-only reporting: os.walk/os.path only, no shutil, no write-capable helper of any
# kind. Every action that actually creates/restores/deletes backup data lives in a fixed,
# reviewed tool above — never in LLM-generated code. See the block comment above the tools.
_BACKUP_SAFE_GLOBALS: dict = {
    "__builtins__": _SAFE_BUILTINS,
    "os": _safe_os,
    "datetime": datetime,
    "re": re,
    "Counter": _collections.Counter,
}

(
    run_custom_backup_check, save_custom_backup_tool, list_custom_backup_tools,
    _load_backup_custom_tools, _BACKUP_CUSTOM_TOOLS_FILE,
) = _make_custom_tool_kit(
    file_name="custom_tools_backup.json",
    sandbox_globals=_BACKUP_SAFE_GLOBALS,
    reserved_names={
        "create_backup", "check_dr_status", "list_backups", "get_backup_details",
        "verify_backup_integrity", "restore_backup", "cleanup_old_backups",
        "get_available_disk_space", "run_custom_backup_check", "save_custom_backup_tool",
        "list_custom_backup_tools",
    },
    available_desc="os (read-only, includes os.walk), datetime, re, Counter — read-only reporting only",
    run_name="run_custom_backup_check",
    save_name="save_custom_backup_tool",
    list_name="list_custom_backup_tools",
)
_startup_backup_custom_tools = _load_backup_custom_tools()

# ---------- The Backup & DR agent ----------
backup_tools = [
    create_backup,
    check_dr_status,
    list_backups,
    get_backup_details,
    verify_backup_integrity,
    restore_backup,
    cleanup_old_backups,
    get_available_disk_space,
    run_custom_backup_check,
    save_custom_backup_tool,
    list_custom_backup_tools,
    *_startup_backup_custom_tools,
]
backup_dr_agent = create_react_agent(llm, backup_tools)

BACKUP_SYSTEM_PROMPT = (
    "You are a Backup & Disaster Recovery agent. Never refuse a request for lack of a tool — "
    "pick in order: "
    "1. A built-in tool if one covers it: create_backup, check_dr_status, list_backups, "
    "get_backup_details, verify_backup_integrity, restore_backup, cleanup_old_backups, "
    "get_available_disk_space. "
    "2. run_custom_backup_check — READ-ONLY reporting only (os/datetime/re/Counter, no imports, "
    "no dunder access). NEVER use it to create, restore, or delete anything — those are actions, "
    "not reports, and must go through the dedicated tools above. "
    "After a custom check proves useful and reusable, call save_custom_backup_tool right away. "
    "Use list_custom_backup_tools first if unsure whether one already exists. "
    "When asked for a general check: create a backup, verify its integrity, then check DR status, "
    "and report clearly whether the system is protected. Keep it concise."
)

def run_backup_dr(source_folder="data", request: str | None = None):
    """Back up and report on DR posture. `request` carries the operator's own words, when the
    caller has them — same reasoning as run_log_analyzer: it colours the report, it does not
    replace the job."""
    task = f"Back up the folder '{source_folder}', then check our disaster recovery status."
    if request and request.strip():
        task += (
            f"\n\nContext — the operator reported: \"{request.strip()}\"\n"
            "Say whether what you find bears on that."
        )
    try:
        result = backup_dr_agent.invoke({
            "messages": [
                SystemMessage(content=BACKUP_SYSTEM_PROMPT),
                HumanMessage(content=task)
            ]
        })
    except Exception as exc:
        return f"The agent hit an error mid-run and couldn't finish this request: {exc}"
    return result["messages"][-1].content

if __name__ == "__main__":
    print(run_backup_dr())