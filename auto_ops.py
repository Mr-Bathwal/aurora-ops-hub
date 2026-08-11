import warnings
warnings.filterwarnings("ignore")

from typing import TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from agents import run_log_analyzer, run_backup_dr, llm

# The shared clipboard for this collaborative workflow
class OpsState(TypedDict):
    request: str
    diagnosis: str
    needs_backup: bool
    remediation: str
    final_report: str

# STEP 1 — DIAGNOSE: the Log Analyzer inspects, then we decide if a backup is needed
#
# The operator's own words reach both halves of this node, and that is the point of the fix.
# Before, `request` was carried the length of the graph and then read only by the supervisor
# writing the summary — so the chain walked an identical route no matter what anyone typed,
# and the request changed nothing but the wording of the closing paragraph.
#
# It now steers two things: what the Log Analyzer leads with, and whether the router opens the
# remediation lane. The second matters more. Asked "are my backups running?", the old router
# saw a log diagnosis with no backup errors in it, answered 'no', and skipped the one agent
# that could have answered the question.
def diagnose_node(state: OpsState) -> dict:
    request = state.get("request", "")
    diagnosis = run_log_analyzer(request=request)
    check = llm.invoke([HumanMessage(content=(
        "You route work in an IT operations platform. Decide whether the Backup & "
        "disaster-recovery agent should run.\n\n"
        "Answer 'yes' if EITHER the diagnosis points at a backup or DR problem, OR the "
        "operator's request asks about backups, snapshots, restores or disaster recovery.\n"
        "Answer 'no' otherwise.\n\n"
        "Answer ONLY 'yes' or 'no'.\n\n"
        f"Operator's request: {request or '(none given)'}\n\n"
        f"Diagnosis: {diagnosis}"
    ))]).content.strip().lower()
    return {"diagnosis": diagnosis, "needs_backup": "yes" in check}

# DECISION — hand off to the Backup agent, or skip to verification?
def route_after_diagnosis(state: OpsState) -> str:
    return "remediate" if state["needs_backup"] else "verify"

# STEP 2 — REMEDIATE: the Backup agent fixes the problem
def remediate_node(state: OpsState) -> dict:
    return {"remediation": run_backup_dr(request=state.get("request", ""))}

# STEP 3 — VERIFY: the supervisor reviews everything and writes the final report
def verify_node(state: OpsState) -> dict:
    prompt = f"""You are the supervisor of an IT operations platform.
Review the work your agents did and write a short, clear final report for the user.
State clearly whether the issue appears resolved.

Original request: {state['request']}
Diagnosis (from Log Analyzer): {state['diagnosis']}
Remediation taken (from Backup agent): {state.get('remediation', 'None needed - no backup issue found.')}
"""
    return {"final_report": llm.invoke([HumanMessage(content=prompt)]).content}

# Build the collaborative graph
builder = StateGraph(OpsState)
builder.add_node("diagnose", diagnose_node)
builder.add_node("remediate", remediate_node)
builder.add_node("verify", verify_node)

builder.add_edge(START, "diagnose")
builder.add_conditional_edges("diagnose", route_after_diagnosis, {
    "remediate": "remediate",
    "verify": "verify",
})
builder.add_edge("remediate", "verify")
builder.add_edge("verify", END)

auto_ops = builder.compile()

def run_auto_ops(request="Check the system for problems and fix anything you can."):
    final = auto_ops.invoke({"request": request})
    return final["final_report"]

if __name__ == "__main__":
    print(run_auto_ops())