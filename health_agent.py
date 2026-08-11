"""The System Health agent, made host-aware.

`agents.py` builds its tools around direct `psutil` calls, which quietly hard-codes the
assumption that the machine being inspected is the machine running the code. That is correct
for a single-box product and wrong for a fleet — an enrolled server could be enrolled, online,
and answering job requests, and the health agent would still report on the API host.

The fix is not to thread a `host_id` parameter through twenty-one LangChain tools. Tools are
module-level singletons registered once at import; giving each an extra argument means the LLM
has to supply it correctly on every call, and a model that forgets it silently reports on the
wrong machine. A wrong answer that looks right is the worst failure mode available here.

Instead the target is ambient: `use_host()` binds a transport for the duration of a request via
a ContextVar, and the tools read it. ContextVar rather than a module global because uvicorn
runs sync endpoints in a thread pool — a global would let two concurrent requests for different
customers overwrite each other's target, which is a cross-tenant data leak, not a race.

One snapshot per run, cached for that run. The agent typically calls four or five tools per
report; without the cache that is five round trips to a remote host instead of one.
"""

import contextvars
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langgraph.prebuilt import create_react_agent

from agents import llm
from transport import LocalTransport, Transport, TransportError

_target: contextvars.ContextVar = contextvars.ContextVar("itops_target", default=None)


class _Target:
    """The transport for the current request, plus a per-request snapshot cache."""

    def __init__(self, transport: Transport, label: str,
                 host_id: str | None = None, org_id: str | None = None):
        self.transport = transport
        self.label = label
        self.host_id = host_id
        self.org_id = org_id
        self._snapshot = None
        self._thresholds = None
        # Every evaluation computed during this run, so the endpoint can take severity from
        # the arithmetic instead of re-reading it out of the model's prose. Scanning the
        # report for "CRITICAL" classified a healthy host as critical the moment the model
        # wrote the sentence "findings with a level of WARNING or CRITICAL include:".
        self.evaluations: list[dict] = []

    _RANK = {"HEALTHY": 0, "OK": 0, "WARNING": 1, "CRITICAL": 2}

    def thresholds(self) -> dict:
        """Per-host warn/crit percentages, falling back to the defaults. Cached for the run so
        two tool calls do not hit the database twice for a value that cannot change mid-run."""
        if self._thresholds is None:
            self._thresholds = {"warn": 85.0, "crit": 95.0}
            if self.host_id:
                try:
                    import metrics
                    t = metrics.get_thresholds(self.host_id)
                    self._thresholds = {"warn": t["warn"], "crit": t["crit"]}
                except Exception:
                    pass
        return self._thresholds

    def record_evaluation(self, evaluation: dict) -> None:
        self.evaluations.append(evaluation)

    def overall_status(self) -> str:
        """Most severe verdict across every evaluation run. Two tool calls each return their
        own status; the host is as unhealthy as its worst finding."""
        if not self.evaluations:
            return "UNKNOWN"
        return max((e.get("overall_status", "HEALTHY") for e in self.evaluations),
                   key=lambda s: self._RANK.get(s, 0))

    def snapshot(self) -> dict:
        if self._snapshot is None:
            self._snapshot = self.transport.snapshot()
            # Every collection is also a data point. Recording here rather than at the
            # endpoint means the history fills up from ordinary use — nobody has to remember
            # to turn monitoring on, and a host has a baseline by the second time anyone
            # looks at it.
            if self.host_id and self.org_id:
                try:
                    import metrics
                    metrics.record_snapshot(self.org_id, self.host_id, self._snapshot)
                except Exception:
                    # Never let bookkeeping break the health check the operator asked for.
                    pass
        return self._snapshot

    def probe(self, name: str, **kwargs) -> dict:
        snap = self._snapshot
        if snap and name in snap and not kwargs:
            return snap[name]
        return self.transport.probe(name, **kwargs)


class use_host:
    """Context manager binding the machine every tool in this module will inspect."""

    def __init__(self, transport: Transport | None = None, label: str = "this server",
                 host_id: str | None = None, org_id: str | None = None):
        self._target = _Target(transport or LocalTransport(), label, host_id, org_id)
        self._token = None

    def __enter__(self) -> _Target:
        self._token = _target.set(self._target)
        return self._target

    def __exit__(self, *exc) -> None:
        _target.reset(self._token)


def _current() -> _Target:
    target = _target.get()
    if target is None:
        # Falling back to local keeps the module usable from a script or a test without
        # ceremony. Every API path binds a target explicitly.
        return _Target(LocalTransport(), "this server")
    return target


def _fmt(value) -> str:
    """Probes return dicts; the model reads text. Compact JSON rather than prose because the
    numbers must survive the trip unaltered — a tool that renders '89.2%' as 'high memory
    usage' has already done the judging the report is supposed to show its working for."""
    return json.dumps(value, default=str)


# --- tools -------------------------------------------------------------------------------

def _probe_tool(probe_name: str, description: str, **fixed):
    def _fn() -> str:
        try:
            return _fmt(_current().probe(probe_name, **fixed))
        except TransportError as exc:
            return f"Could not reach the host: {exc}"
    _fn.__name__ = f"get_{probe_name}"
    return StructuredTool.from_function(func=_fn, name=f"get_{probe_name}",
                                        description=description)


def get_full_snapshot() -> str:
    """Every cheap health metric for the target host in one call — CPU, memory, swap, disk,
    partitions, disk I/O, network, connections, top processes, process count, uptime, battery,
    temperature, logged-in users and interfaces. Use this first for any general health check;
    it is a single round trip where the individual tools are one each.

    Also returns an `evaluation` block containing the overall status and every threshold
    breach, already computed. Use those verdicts verbatim — do not re-derive them."""
    try:
        import probes
        target = _current()
        snap = target.snapshot()
        # The verdicts travel with the data, so the model never has to compare a number to a
        # threshold. It got that backwards in testing — "80.7%, which is above the warning
        # threshold of 85%" — and a fluent wrong answer is worse than no answer.
        evaluation = probes.evaluate(snap, target.thresholds())
        target.record_evaluation(evaluation)
        return _fmt({**snap, "evaluation": evaluation})
    except TransportError as exc:
        return f"Could not reach the host: {exc}"


def get_metric_trends(window_hours: int = 24) -> str:
    """How CPU, memory, disk and swap have moved on this host over a recent window, with
    min/max/average/p95, the change across the window, and a direction of rising, falling or
    stable — plus any anomalies.

    Call this whenever the question is whether a reading is *normal*, when investigating a
    slow degradation, or when a metric looks high but is below its threshold. A single
    reading cannot distinguish a host that has always sat at 74% from one that was at 40%
    on Monday; this can. Returns sufficient_data=false where too little history exists yet."""
    target = _current()
    if not target.host_id:
        return _fmt({"error": "No host bound; trends need a registered host."})
    try:
        import metrics
        return _fmt({
            "window_hours": window_hours,
            "trends": metrics.summary(target.host_id, window_hours),
            "anomalies": metrics.anomalies(target.host_id, window_hours),
            "thresholds": target.thresholds(),
        })
    except Exception as exc:
        return _fmt({"error": f"Could not read history: {exc}"})


def get_posture_checks() -> str:
    """The six audit-facing checks in one call: pending reboot, Windows Update level, failed
    logon attempts, failed scheduled tasks, time synchronisation and certificate expiry.
    Windows only. Call this alongside get_full_snapshot for any full or general health check —
    resource metrics alone will not reveal a server that has been waiting to reboot for a
    fortnight or a backup task that stopped running. Returns findings already classified."""
    try:
        import probes
        target = _current()
        result = target.probe("posture")
        evaluation = probes.evaluate(result, target.thresholds())
        target.record_evaluation(evaluation)
        return _fmt({**result, "evaluation": evaluation})
    except TransportError as exc:
        return f"Could not reach the host: {exc}"


HEALTH_TOOLS = [
    StructuredTool.from_function(func=get_full_snapshot, name="get_full_snapshot",
                                 description=get_full_snapshot.__doc__),
    StructuredTool.from_function(func=get_posture_checks, name="get_posture_checks",
                                 description=get_posture_checks.__doc__),
    StructuredTool.from_function(func=get_metric_trends, name="get_metric_trends",
                                 description=get_metric_trends.__doc__),
    _probe_tool("cpu", "CPU usage overall and per core, with core counts and load average."),
    _probe_tool("memory", "RAM usage: percent, total, used and available in GB."),
    _probe_tool("swap", "Swap/virtual memory usage, or whether swap is configured at all."),
    _probe_tool("disk", "Disk usage of the root/system volume."),
    _probe_tool("partitions", "Every mounted partition with its usage and free space."),
    _probe_tool("disk_io", "Cumulative disk read/write volume and operation counts."),
    _probe_tool("network", "Network bytes sent/received plus error and dropped-packet counters."),
    _probe_tool("connections", "Count of active network connections grouped by TCP state."),
    _probe_tool("listening_ports", "Every listening port with the process and PID bound to it."),
    _probe_tool("interfaces", "Network adapters, whether each is up, and their IP addresses."),
    _probe_tool("top_processes", "Top processes by CPU and by memory."),
    _probe_tool("process_count", "Total running processes and threads."),
    _probe_tool("uptime", "How long the host has been up and when it last booted."),
    _probe_tool("battery", "Battery percentage and charging state (laptops only)."),
    _probe_tool("temperature", "CPU temperature, where the platform exposes a sensor."),
    _probe_tool("logged_in_users", "Users currently logged in and when their session started."),
    # --- posture checks: slower, Windows-only, and the ones an auditor actually asks about ---
    _probe_tool("pending_reboot",
                "Whether the host is waiting on a reboot to finish applying updates. Windows only."),
    _probe_tool("windows_updates",
                "Patch level: how many hotfixes are installed and the most recent ones. Windows only."),
    _probe_tool("failed_logins",
                "Failed logon attempts (Security event 4625) in the last 24 hours — a spike "
                "indicates a brute-force attempt. Windows only, needs an elevated agent."),
    _probe_tool("scheduled_task_failures",
                "Scheduled tasks that failed in the last 7 days — catches backup jobs that "
                "silently stopped running. Windows only."),
    _probe_tool("time_sync",
                "Clock synchronisation source and last successful sync. Skew breaks Kerberos "
                "and log correlation. Windows only."),
    _probe_tool("certificate_expiry",
                "Machine certificates expiring within 60 days. Windows only."),
]

HEALTH_SYSTEM_PROMPT = (
    "You are a System Health agent inspecting ONE specific host.\n"
    "\n"
    "For any general or full health check, call BOTH get_full_snapshot and "
    "get_posture_checks. Resource metrics alone are not a health check — they will not show a "
    "server pending reboot for a fortnight or a backup task that stopped running. For a "
    "narrow question ('how much disk is free?'), call only the tool that answers it.\n"
    "\n"
    "Both tools return an `evaluation` block with `overall_status` and a `findings` list, "
    "each finding already carrying a `level` of OK, WARNING or CRITICAL. Those verdicts are "
    "computed in code and are authoritative. Report them as given. Never compare a number to "
    "a threshold yourself and never restate a level differently from the one supplied.\n"
    "\n"
    "ANSWER THE QUESTION THAT WAS ASKED FIRST, in one or two sentences, before anything "
    "else. If asked whether memory is trending upward, open by saying whether it is and by "
    "how much. A generic report that never addresses the specific question is a failure "
    "however accurate its numbers.\n"
    "When you called get_metric_trends, state what it found even if the answer is 'stable' — "
    "'memory has held between 73.4% and 73.9% across 7 readings' is the answer, and omitting "
    "it leaves the reader unable to tell whether you checked.\n"
    "\n"
    "Then WRITE A COMPLETE REPORT, not a one-line verdict. Always include:\n"
    "  1. A short paragraph on the host's overall condition.\n"
    "  2. A bulleted list of the key metrics WITH their actual values — CPU, memory (used of "
    "total), disk (used of total, free), swap, uptime, process count. Quote the real numbers.\n"
    "  3. Every finding whose level is WARNING or CRITICAL, spelled out with what it means "
    "and what to do about it.\n"
    "  4. Any check reporting available=false, named, with the reason. An unavailable check "
    "is not a pass.\n"
    "  5. A final line: 'Overall status: X' using evaluation.overall_status verbatim. When "
    "the two tools disagree, take the more severe of the two.\n"
    "\n"
    "Never estimate, never invent a metric you did not read, and never omit a value you were "
    "given."
)

health_agent = create_react_agent(llm, HEALTH_TOOLS)


def run_health_check(query: str = "Check this host's health and give me a full report.",
                     transport: Transport | None = None,
                     label: str = "this server",
                     host_id: str | None = None,
                     org_id: str | None = None) -> dict:
    """Run the health agent against one host.

    Returns the report, the tool trace, and — separately from both — the status computed in
    Python. Callers should take severity from `status`, never by reading the prose: the report
    legitimately contains the words WARNING and CRITICAL while explaining what they mean, so
    any substring scan over it misclassifies a healthy host the moment the model writes a
    sentence like "findings with a level of WARNING or CRITICAL include:"."""
    from api import run_with_trace  # local import: api imports this module at startup

    with use_host(transport, label, host_id, org_id) as target:
        trace, report = run_with_trace(health_agent, [
            SystemMessage(content=HEALTH_SYSTEM_PROMPT + f"\nThe host is: {label}."),
            HumanMessage(content=query),
        ])
        status = target.overall_status()
        evaluations = list(target.evaluations)
    return {"report": report, "trace": trace, "status": status, "evaluations": evaluations}
