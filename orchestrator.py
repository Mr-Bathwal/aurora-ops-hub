import warnings
warnings.filterwarnings("ignore")

from typing import TypedDict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from agents import run_system_health, run_log_analyzer, run_backup_dr, llm

# 1. The shared "clipboard" that flows through the graph
class HubState(TypedDict):
    request: str
    agent: str
    result: str

# 2. The ROUTER node — the LLM reads the request and picks one agent
def router_node(state: HubState) -> dict:
    prompt = f"""You are a router for an IT operations platform.
Based on the user's request, choose which ONE agent should handle it.
Reply with ONLY one word:
- health  : checking CPU, memory, disk, or overall system health
- log     : analyzing or summarizing log files
- backup  : backups or disaster recovery status
- unknown : if none of these fit

User request: "{state['request']}"
"""
    decision = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
    for option in ("health", "log", "backup"):
        if option in decision:
            return {"agent": option}
    return {"agent": "unknown"}

# 3. One node per agent — each runs its specialist and saves the result
def health_node(state: HubState) -> dict:
    return {"result": run_system_health()}

def log_node(state: HubState) -> dict:
    return {"result": run_log_analyzer()}

def backup_node(state: HubState) -> dict:
    return {"result": run_backup_dr()}

def unknown_node(state: HubState) -> dict:
    return {"result": "I couldn't match that to an agent. Try asking about system health, logs, or backups."}

# 4. Reads the router's choice to decide which path to take
def choose_agent(state: HubState) -> str:
    return state["agent"]

# 5. Build the flowchart
builder = StateGraph(HubState)
builder.add_node("router", router_node)
builder.add_node("health", health_node)
builder.add_node("log", log_node)
builder.add_node("backup", backup_node)
builder.add_node("unknown", unknown_node)

builder.add_edge(START, "router")
builder.add_conditional_edges("router", choose_agent, {
    "health": "health",
    "log": "log",
    "backup": "backup",
    "unknown": "unknown",
})
builder.add_edge("health", END)
builder.add_edge("log", END)
builder.add_edge("backup", END)
builder.add_edge("unknown", END)

orchestrator = builder.compile()

# 6. A simple function the platform can call
def run_orchestrator(request: str) -> str:
    final_state = orchestrator.invoke({"request": request})
    return final_state["result"]

if __name__ == "__main__":
    print(run_orchestrator("how is my computer doing right now?"))