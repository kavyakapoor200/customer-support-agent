"""LangGraph StateGraph definition and workflow compiler."""
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.workflow.nodes import (
    decide_node,
    draft_node,
    execute_node,
    gate_node,
    human_review_node,
    intake_node,
    retrieve_policy_node,
    verify_node,
)
from src.workflow.state import AgentState


def route_after_gate(state: AgentState) -> Literal["draft_auto", "draft_review", "draft_direct"]:
    """Conditional router based on deterministic gating outcome."""
    outcome = state.get("gating_outcome", "human_review")
    if outcome == "auto_execute":
        return "draft_auto"
    elif outcome == "human_review":
        return "draft_review"
    else:
        return "draft_direct"


def create_customer_support_graph(checkpointer: BaseCheckpointSaver | None = None):
    """Assembles and compiles the customer support LangGraph workflow.

    Args:
        checkpointer: State persistence checkpointer. Defaults to MemorySaver for tests.

    Returns:
        CompiledStateGraph instance with interrupt support.
    """
    workflow = StateGraph(AgentState)

    # 1. Register Nodes
    workflow.add_node("intake", intake_node)
    workflow.add_node("retrieve_policy", retrieve_policy_node)
    workflow.add_node("decide", decide_node)
    workflow.add_node("gate", gate_node)
    workflow.add_node("draft", draft_node)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("verify", verify_node)
    workflow.add_node("execute", execute_node)

    # 2. Add Fixed Edges
    workflow.add_edge(START, "intake")
    workflow.add_edge("intake", "retrieve_policy")
    workflow.add_edge("retrieve_policy", "decide")
    workflow.add_edge("decide", "gate")

    # 3. Add Conditional Routing from Gate
    workflow.add_conditional_edges(
        "gate",
        route_after_gate,
        {
            "draft_auto": "draft",
            "draft_review": "draft",
            "draft_direct": "draft",
        }
    )

    # 4. Route from Draft to either Human Review or Direct Verification
    def route_after_draft(state: AgentState) -> Literal["human_review", "verify"]:
        if state.get("gating_outcome") == "human_review":
            return "human_review"
        return "verify"

    workflow.add_conditional_edges(
        "draft",
        route_after_draft,
        {
            "human_review": "human_review",
            "verify": "verify",
        }
    )

    workflow.add_edge("human_review", "verify")
    workflow.add_edge("verify", "execute")
    workflow.add_edge("execute", END)

    # Compile with checkpointer (required for LangGraph interrupts)
    saver = checkpointer if checkpointer is not None else MemorySaver()
    return workflow.compile(checkpointer=saver)
