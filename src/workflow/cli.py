"""CLI probe for testing the LangGraph customer support workflow."""
import argparse
import asyncio
import json
import uuid

from langgraph.types import Command

from src.workflow.graph import create_customer_support_graph


async def main() -> None:
    parser = argparse.ArgumentParser(description="Test LangGraph Customer Support workflow.")
    parser.add_argument("--ticket", type=str, required=True, help="Customer message.")
    parser.add_argument("--customer-id", type=str, default="CUST-4091")
    parser.add_argument("--interrupt-test", action="store_true", help="Demonstrate interrupt & resume.")

    args = parser.parse_args()

    graph = create_customer_support_graph()
    ticket_id = f"TK-{uuid.uuid4().hex[:6].upper()}"
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "ticket_id": ticket_id,
        "customer_id": args.customer_id,
        "raw_text": args.ticket,
        "trajectory": [],
    }

    print(f"\n--- [1] Starting Graph Execution for Thread: {thread_id} ---")
    async for event in graph.astream(initial_state, config=config):
        for node_name, state_update in event.items():
            print(f"Node Executed: [{node_name}]")

    current_state = await graph.aget_state(config)
    print(f"\nCurrent State Next Nodes: {current_state.next}")

    if current_state.next and "human_review" in current_state.next:
        print("\n🚨 [LangGraph Interrupt Triggered!]")
        print("Ticket is paused at human_review node.")
        tasks = current_state.tasks
        if tasks and tasks[0].interrupts:
            print("Interrupt Payload:", json.dumps(tasks[0].interrupts[0].value, indent=2))

        if args.interrupt_test:
            print("\n--- [2] Simulating Human Reviewer Approval ---")
            resume_command = Command(resume={"approved": True, "notes": "Approved via CLI probe"})
            async for event in graph.astream(resume_command, config=config):
                for node_name, state_update in event.items():
                    print(f"Resumed Node Executed: [{node_name}]")

            final_state = await graph.aget_state(config)
            print("\n✅ Final Completed State:")
            print("Action Taken:", final_state.values.get("final_action_taken"))
            print("Draft Reply:", final_state.values.get("draft_reply"))
            print("Tool Result:", json.dumps(final_state.values.get("tool_result"), indent=2))
    else:
        print("\n✅ Execution Completed Automatically (Zero Human Review Needed):")
        print("Gating Outcome:", current_state.values.get("gating_outcome"))
        print("Action Taken:", current_state.values.get("final_action_taken"))
        print("Draft Reply:", current_state.values.get("draft_reply"))


if __name__ == "__main__":
    asyncio.run(main())
