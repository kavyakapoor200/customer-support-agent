from typing import Any

import gradio as gr

from src.api.models import ReviewActionRequest, TicketIntakeRequest
from src.api.service import workflow_service


def create_gradio_ui() -> gr.Blocks:
    """Builds the dual-tab Gradio web interface."""
    with gr.Blocks(title="Customer Support Agent Portal") as demo:
        gr.Markdown(
            "# 🛡️ Customer Support AI Agent\n"
            "**System 1 Gated Decision Engine · LangGraph Human-in-the-Loop Supervision · Qdrant Policy Grounding**"
        )

        with gr.Tabs():
            # ==================================================================
            # Tab 1: Customer Support Portal
            # ==================================================================
            with gr.Tab("💬 Customer Support Portal"):
                gr.Markdown("### Submit an inquiry or support request")

                with gr.Row():
                    with gr.Column(scale=3):
                        user_input = gr.Textbox(
                            label="Your Support Request",
                            placeholder="Type your message in English, Hinglish, or Hindi (e.g. 'Mera $35 ka refund kardo bhai')...",
                            lines=3,
                        )
                        with gr.Row():
                            btn_send = gr.Button("Submit Ticket", variant="primary")
                            btn_clear = gr.Button("Clear")

                        gr.Markdown("#### Quick Test Scenarios")
                        with gr.Row():
                            btn_scen_auto = gr.Button("💰 Refund $35 (Hinglish)", size="sm")
                            btn_scen_cancel = gr.Button("❌ Cancel Plan (English)", size="sm")
                            btn_scen_high = gr.Button("⚠️ Overcharge $180 (Needs Review)", size="sm")
                            btn_scen_p0 = gr.Button("🚨 SSO Lockout (P0 Security)", size="sm")

                    with gr.Column(scale=2):
                        out_status = gr.Markdown("### Status\n*No ticket submitted yet.*")
                        out_reply = gr.Textbox(label="Agent Response", lines=4, interactive=False)
                        out_details = gr.JSON(label="Execution Details")

                async def handle_submit(text: str):
                    if not text.strip():
                        return "### Status\n*Please enter a message.*", "", {}
                    req = TicketIntakeRequest(text=text)
                    res = await workflow_service.intake_ticket(req)

                    badge = (
                        "<span class='badge-review'>⚠️ ESCALATED TO HUMAN REVIEW DESK</span>"
                        if res.requires_human_review
                        else "<span class='badge-auto'>✅ AUTO-RESOLVED</span>"
                    )

                    status_md = (
                        f"### Status: {badge}\n"
                        f"* **Ticket ID:** `{res.ticket_id}`\n"
                        f"* **Detected Action:** `{res.decision_action}` (Confidence: {res.decision_confidence:.2f})\n"
                        f"* **Detected Language:** `{res.detected_language}`\n"
                        f"* **Gating Outcome:** `{res.gating_outcome}`"
                    )

                    return status_md, res.reply or "", res.model_dump()

                btn_send.click(handle_submit, inputs=[user_input], outputs=[out_status, out_reply, out_details])
                btn_clear.click(lambda: ("", "### Status\n*Cleared.*", "", {}), outputs=[user_input, out_status, out_reply, out_details])

                # Quick Scenario wiring
                btn_scen_auto.click(lambda: "Mera renewal galti se ho gaya kal, $35 charge hua hai, refund kardo bhai please.", outputs=[user_input])
                btn_scen_cancel.click(lambda: "I want to cancel my subscription at the end of the billing cycle.", outputs=[user_input])
                btn_scen_high.click(lambda: "I was charged $180 unexpectedly for a team plan. Reverse this charge.", outputs=[user_input])
                btn_scen_p0.click(lambda: "URGENT! Entire engineering team locked out of Okta SSO right now. P0 blocker.", outputs=[user_input])

            # ==================================================================
            # Tab 2: Agent Review Desk (Human-In-The-Loop)
            # ==================================================================
            with gr.Tab("🧑‍💼 Agent Review Desk"):
                gr.Markdown("### Human-In-The-Loop Review Queue (Paused Tickets)")

                with gr.Row():
                    btn_refresh = gr.Button("🔄 Refresh Pending Queue", size="sm")

                pending_dropdown = gr.Dropdown(label="Select Pending Ticket to Inspect", choices=[])

                with gr.Group():
                    rev_info = gr.Markdown("*Select a ticket from the dropdown above to inspect details.*")
                    rev_draft = gr.Textbox(label="Customer Reply (Editable by Reviewer)", lines=4)
                    rev_notes = gr.Textbox(label="Reviewer Justification Notes", placeholder="e.g. Approved exception after reviewing payment receipt.")

                    with gr.Row():
                        btn_approve = gr.Button("✅ Approve & Execute Action", variant="primary")
                        btn_reject = gr.Button("❌ Reject & Deny Request", variant="stop")

                    rev_result = gr.Markdown("")

                # In-memory storage for dropdown ticket selection
                cached_items: dict[str, Any] = {}

                async def refresh_queue():
                    items = await workflow_service.list_pending()
                    choices = [f"{item.ticket_id} | {item.action} | {item.text[:40]}..." for item in items]
                    cached_items.clear()
                    for item, choice in zip(items, choices):
                        cached_items[choice] = item
                    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

                async def inspect_ticket(selected_choice: str):
                    if not selected_choice or selected_choice not in cached_items:
                        return "*No ticket selected.*", "", ""

                    item = cached_items[selected_choice]
                    policies_str = "\n".join([f"• **{p['title']}:** {p['content'][:150]}..." for p in item.retrieved_policies]) or "None"

                    md = (
                        f"#### Ticket: `{item.ticket_id}` (Customer: `{item.customer_id}`)\n"
                        f"* **Customer Message:** \"{item.text}\"\n"
                        f"* **Detected Language:** `{item.detected_language}`\n"
                        f"* **Model Decision:** `{item.action}` (Confidence: {item.confidence:.2f})\n"
                        f"* **Amount:** `${item.amount:.2f}`\n" if item.amount else ""
                        f"* **Review Reason:** `{item.reason}`\n\n"
                        f"**🛡️ Retrieved Policy Grounding:**\n{policies_str}"
                    )
                    return md, item.draft_reply or "", ""

                async def process_human_verdict(selected_choice: str, draft: str, notes: str, approved: bool):
                    if not selected_choice or selected_choice not in cached_items:
                        return "### ⚠️ Please select a valid ticket from the dropdown."

                    item = cached_items[selected_choice]
                    rev_req = ReviewActionRequest(
                        approved=approved,
                        edited_reply=draft if draft.strip() else None,
                        notes=notes or ("Approved by supervisor" if approved else "Rejected by supervisor"),
                    )

                    res = await workflow_service.review_ticket(item.ticket_id, rev_req)
                    verdict_str = "APPROVED & EXECUTED" if approved else "REJECTED & DENIED"
                    return f"### ✅ Verdict [{verdict_str}] Recorded for Ticket `{item.ticket_id}`\nAction: `{res.decision_action}` | Result: `{res.tool_result.get('status') if res.tool_result else 'DENIED'}`"

                btn_refresh.click(refresh_queue, outputs=[pending_dropdown])
                pending_dropdown.change(inspect_ticket, inputs=[pending_dropdown], outputs=[rev_info, rev_draft, rev_notes])
                btn_approve.click(
                    lambda sel, draft, notes: process_human_verdict(sel, draft, notes, approved=True),
                    inputs=[pending_dropdown, rev_draft, rev_notes],
                    outputs=[rev_result]
                )
                btn_reject.click(
                    lambda sel, draft, notes: process_human_verdict(sel, draft, notes, approved=False),
                    inputs=[pending_dropdown, rev_draft, rev_notes],
                    outputs=[rev_result]
                )

        return demo
