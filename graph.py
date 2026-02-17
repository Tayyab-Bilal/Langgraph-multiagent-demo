from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage
import operator
from agents import (
    run_greeter,
    run_retention_agent,
    run_processor,
    run_tech_support,
    run_billing,
)

# ── Shared State ─────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    intent: str
    cancellation_reason: str
    customer_email: str
    customer_data: dict
    outcome: str
    retention_action: str
    processed: bool

# ── Routing Logic ─────────────────────────────────────────
def route_after_greeter(state: AgentState) -> str:
    """Decide which agent gets the customer after greeting."""
    intent = state.get("intent", "OTHER")
    email = state.get("customer_email", "")

    # If we don't have email yet, loop back to greeter
    if (not email or "@" not in email) and intent not in ("TECH_SUPPORT", "BILLING"):
            return END
    
    if intent == "RETENTION":
        return "retention_agent"
    elif intent == "TECH_SUPPORT":
        return "tech_support"
    elif intent == "BILLING":
        return "billing"
    else:
        return END


def route_after_retention(state: AgentState) -> str:
    """After retention agent responds, decide what happens next."""
    outcome = state.get("outcome", "IN_PROGRESS")

    if outcome == "CANCEL" or outcome == "RETAINED":
        return "processor"
    else:
        # Still in conversation — go back to retention agent
        return END


# ── Build the Graph ───────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("greeter", run_greeter)
    graph.add_node("retention_agent", run_retention_agent)
    graph.add_node("processor", run_processor)
    graph.add_node("tech_support", run_tech_support)
    graph.add_node("billing", run_billing)

    # Entry point
    graph.add_edge(START, "greeter")

    # After greeter → route based on intent
    graph.add_conditional_edges(
        "greeter",
        route_after_greeter,
        {
            END: END,
            "retention_agent": "retention_agent",
            "tech_support": "tech_support",
            "billing": "billing",
        }
    )

    # After retention agent → route based on outcome
    graph.add_conditional_edges(
        "retention_agent",
        route_after_retention,
        {
            END: END,
            "processor": "processor",
        }
    )

    # Terminal nodes → END
    graph.add_edge("processor", END)
    graph.add_edge("tech_support", END)
    graph.add_edge("billing", END)

    return graph.compile()
