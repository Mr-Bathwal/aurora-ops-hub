import warnings
warnings.filterwarnings("ignore")

import os
import psutil
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from groq import APIStatusError, RateLimitError
from pydantic import BaseModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agents import (
    system_health_agent, log_analyzer_agent, backup_dr_agent,
    HEALTH_SYSTEM_PROMPT, LOG_SYSTEM_PROMPT, BACKUP_SYSTEM_PROMPT,
    _CUSTOM_TOOLS_FILE, _LOG_CUSTOM_TOOLS_FILE, _BACKUP_CUSTOM_TOOLS_FILE,
)
from orchestrator import orchestrator as orchestrator_graph
from auto_ops import auto_ops as auto_ops_graph

from control_api import router as control_router, legacy_guard
from db import init_db

app = FastAPI(title="IT Ops Assistant Hub API")

# allow_credentials is required for the session cookie to survive a cross-origin request from
# the Next.js dev server; with it set, the origin list may not be a wildcard, which is why
# these are named explicitly rather than "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()
app.include_router(control_router)


# An unhandled exception in a route is caught by Starlette's ServerErrorMiddleware, which sits
# *outside* CORSMiddleware — so its 500 goes back without the CORS headers, the browser blocks
# the response, and `fetch` rejects with a network error. The console then reported the one
# thing that was not true: "Can't reach the backend. Is it running?" — sending you to check a
# server that answered fine.
#
# Handlers registered for a specific exception class run in ExceptionMiddleware instead, which
# is *inside* CORS, so these responses reach the browser intact and the real reason is shown.
@app.exception_handler(APIStatusError)
async def _groq_error(request: Request, exc: APIStatusError) -> JSONResponse:
    if isinstance(exc, RateLimitError):
        return JSONResponse(
            status_code=503,
            content={"detail": "The model provider's rate limit is exhausted. It refills on a "
                               "rolling window — wait a few minutes and run it again."},
        )
    return JSONResponse(
        status_code=502,
        content={"detail": f"The model provider rejected the request ({exc.status_code}). "
                           "This is usually a malformed tool call; running it again normally clears it."},
    )


@app.on_event("startup")
def _start_collector() -> None:
    """Begin background metric collection once the app is up.

    Started here rather than at import so that importing this module for a test or a script
    does not silently spawn a thread that writes to the database."""
    import collector
    collector.start()


@app.on_event("shutdown")
def _stop_collector() -> None:
    import collector
    collector.stop()


class OrchestrateRequest(BaseModel):
    request: str


class AutoRemediateRequest(BaseModel):
    """Optional, so the existing callers that POST an empty body keep working."""
    request: str | None = None


def run_with_trace(agent, messages):
    """Invoke a react agent and turn its message history into a step-by-step trace
    of real tool calls and tool outputs, instead of just the final answer."""
    try:
        result = agent.invoke({"messages": messages})
    except Exception as exc:
        # The model occasionally emits a malformed tool call (e.g. a syntax error in
        # generated code) which the Groq API rejects outright, or the request hits a
        # transient network/rate-limit error. Either way, surface it instead of a 500.
        return (
            [{"type": "tool_error", "name": "agent", "content": str(exc)}],
            "The agent hit an error mid-run and couldn't finish this request "
            "(often a rate limit or a malformed generated snippet) — please try again.",
        )
    trace = []
    for m in result["messages"]:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                trace.append({"type": "tool_call", "name": tc["name"], "args": tc.get("args", {})})
        elif isinstance(m, ToolMessage):
            trace.append({"type": "tool_result", "name": m.name, "content": m.content})
    return trace, result["messages"][-1].content


@app.get("/api/vitals", dependencies=[Depends(legacy_guard)])
def get_vitals():
    return {
        "cpu": psutil.cpu_percent(interval=0.3),
        "memory": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage("C:\\").percent,
    }


@app.get("/api/system-health", dependencies=[Depends(legacy_guard)])
def system_health(query: str = "Check the system health and give me a full report."):
    trace, report = run_with_trace(system_health_agent, [
        SystemMessage(content=HEALTH_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ])
    return {"report": report, "trace": trace}


@app.get("/api/log-analysis", dependencies=[Depends(legacy_guard)])
def log_analysis(filename: str = "sample.log"):
    trace, report = run_with_trace(log_analyzer_agent, [
        SystemMessage(content=LOG_SYSTEM_PROMPT),
        HumanMessage(content=f"Analyze the log file named '{filename}' and give me a summary.")
    ])
    return {"report": report, "trace": trace}


@app.post("/api/backup", dependencies=[Depends(legacy_guard)])
def backup(source_folder: str = "data"):
    trace, report = run_with_trace(backup_dr_agent, [
        SystemMessage(content=BACKUP_SYSTEM_PROMPT),
        HumanMessage(content=f"Back up the folder '{source_folder}', then check our disaster recovery status.")
    ])
    return {"report": report, "trace": trace}


@app.post("/api/orchestrate", dependencies=[Depends(legacy_guard)])
def orchestrate(body: OrchestrateRequest):
    final_state = orchestrator_graph.invoke({"request": body.request})
    return {"report": final_state["result"], "agent": final_state["agent"]}


@app.get("/api/custom-tools", dependencies=[Depends(legacy_guard)])
def list_custom_tools_endpoint():
    """Returns every self-expanded custom tool saved by any of the three agents so far."""
    import json

    def _load(domain: str, path: str) -> list:
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r", encoding="utf-8") as fh:
                saved = json.load(fh)
        except Exception:
            return []
        return [
            {"domain": domain, "name": name, "description": info["description"], "code": info["code"]}
            for name, info in saved.items()
        ]

    tools = (
        _load("system-health", _CUSTOM_TOOLS_FILE)
        + _load("log-analyzer", _LOG_CUSTOM_TOOLS_FILE)
        + _load("backup-dr", _BACKUP_CUSTOM_TOOLS_FILE)
    )
    return {"tools": tools}


@app.post("/api/auto-remediate", dependencies=[Depends(legacy_guard)])
def auto_remediate(body: AutoRemediateRequest | None = None):
    # The body is optional and so is the field inside it, so a bare POST still runs the
    # default sweep — the console had been calling this with no body at all.
    request = (body.request if body and body.request else "").strip()
    final = auto_ops_graph.invoke({
        "request": request or "Check the system for problems and fix anything you can."
    })
    return {
        "report": final["final_report"],
        "diagnosis": final["diagnosis"],
        "needs_backup": final["needs_backup"],
        "remediation": final.get("remediation"),
    }
