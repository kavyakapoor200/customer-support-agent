"""Unit tests for LangGraph state machine, deterministic gating, and HITL interrupts."""
import json
import uuid

import pytest
from langgraph.types import Command

from src.workflow.graph import create_customer_support_graph


@pytest.mark.asyncio
async def test_workflow_auto_execute_path():
    """Validates complete auto-execution for an eligible low-risk refund under $50."""
    graph = create_customer_support_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "ticket_id": "TK-AUTO-01",
        "customer_id": "CUST-10",
        "raw_text": "I was charged twice, duplicate accidental renewal, please refund my money back for $45",
        "trajectory": [],
    }

    # Execute graph to completion
    async for _ in graph.astream(initial_state, config=config):
        pass

    state = await graph.aget_state(config)
    # Execution should be fully complete with no pending next nodes
    assert len(state.next) == 0
    assert state.values["decision_action"] == "refund"
    assert state.values["gating_outcome"] == "auto_execute"
    assert state.values["final_action_taken"] == "refund"
    assert state.values["tool_result"]["success"] is True
    assert "refund" in state.values["draft_reply"].lower()

    # Invariant: State must be 100% JSON-serializable
    serialized = json.dumps(state.values)
    assert len(serialized) > 0


@pytest.mark.asyncio
async def test_workflow_hitl_interrupt_and_approval():
    """Validates that tickets over $50 trigger a human_review interrupt, and can be approved."""
    graph = create_customer_support_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "ticket_id": "TK-HITL-02",
        "customer_id": "CUST-20",
        "raw_text": "Need a refund for $180 corporate plan charge immediately",
        "trajectory": [],
    }

    # Stream graph until interrupt
    async for _ in graph.astream(initial_state, config=config):
        pass

    state_before_resume = await graph.aget_state(config)
    # Graph should be paused waiting on human_review
    assert "human_review" in state_before_resume.next
    assert state_before_resume.values["gating_outcome"] == "human_review"
    assert state_before_resume.tasks[0].interrupts[0].value["ticket_id"] == "TK-HITL-02"

    # Resume with approval
    resume_payload = Command(
        resume={
            "approved": True,
            "edited_reply": "Supervisor approved: We have processed your $180 refund.",
            "notes": "Manager authorized refund override",
        }
    )
    async for _ in graph.astream(resume_payload, config=config):
        pass

    state_after_resume = await graph.aget_state(config)
    assert len(state_after_resume.next) == 0
    assert state_after_resume.values["review_status"] == "approved"
    assert state_after_resume.values["final_action_taken"] == "refund"
    assert state_after_resume.values["draft_reply"] == "Supervisor approved: We have processed your $180 refund."
    assert state_after_resume.values["tool_result"]["success"] is True


@pytest.mark.asyncio
async def test_workflow_hitl_interrupt_and_rejection():
    """Validates that a human reviewer can reject a high-risk request."""
    graph = create_customer_support_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "ticket_id": "TK-HITL-03",
        "customer_id": "CUST-30",
        "raw_text": "Refund me $300 right now",
        "trajectory": [],
    }

    async for _ in graph.astream(initial_state, config=config):
        pass

    # Resume with rejection
    resume_payload = Command(
        resume={"approved": False, "notes": "Violates 14-day SLA window"}
    )
    async for _ in graph.astream(resume_payload, config=config):
        pass

    final_state = await graph.aget_state(config)
    assert len(final_state.next) == 0
    assert final_state.values["review_status"] == "rejected"
    assert final_state.values["final_action_taken"] == "denied"
    assert final_state.values["tool_result"]["status"] == "DENIED"


@pytest.mark.asyncio
async def test_workflow_hinglish_language_mirroring():
    """Validates that Hinglish tickets receive Hinglish tone mirroring in draft response."""
    graph = create_customer_support_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "ticket_id": "TK-HINGLISH-04",
        "customer_id": "CUST-40",
        "raw_text": "Mera $35 ka refund kardo bhai, auto renew galti se ho gaya",
        "trajectory": [],
    }

    async for _ in graph.astream(initial_state, config=config):
        pass

    state = await graph.aget_state(config)
    assert state.values["detected_language"] == "hinglish"
    assert state.values["detected_script"] == "latin"
    assert "rahul" in state.values["draft_reply"].lower() or "humne" in state.values["draft_reply"].lower()
