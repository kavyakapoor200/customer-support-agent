import datetime
import uuid

import gradio as gr
import httpx

from src.api.models import ReviewActionRequest, TicketIntakeRequest
from src.api.service import workflow_service
from src.core.config import get_settings


def create_gradio_ui() -> gr.Blocks:
    """Builds the dual-tab Gradio web interface with webhook diagnostics and priority triage."""
    with gr.Blocks(title="Customer Support Agent Portal") as demo:
        gr.Markdown(
            "# 🛡️ Customer Support AI Agent\n"
            "**System 1 Gated Decision Engine · Multi-tier Severity Triage (P0 / P1 / P2) · Human-in-the-Loop Supervision**"
        )

        # ======================================================================
        # Webhook Connection & Test Bar
        # ======================================================================
        settings = get_settings()
        webhook_target = settings.SLACK_WEBHOOK_URL or "Not Configured (simulated in console)"
        with gr.Accordion("🔔 Webhook Live Diagnostics", open=False):
            with gr.Row():
                gr.Markdown(f"**Target Webhook URL:** `{webhook_target}`")
                btn_test_ping = gr.Button("🚀 Send Test Alert to Webhook Now", size="sm", variant="secondary")
            webhook_ping_result = gr.Markdown("")

            async def handle_test_ping():
                cfg = get_settings()
                if not cfg.SLACK_WEBHOOK_URL:
                    return "⚠️ **SLACK_WEBHOOK_URL** is not set in `.env`!"
                try:
                    now_str = datetime.datetime.now(datetime.UTC).strftime("%H:%M:%S UTC")
                    payload = {
                        "text": f"🚨 *Manual Test Alert from Support Portal* ({now_str})",
                        "blocks": [
                            {
                                "type": "header",
                                "text": {"type": "plain_text", "text": "🔔 Webhook Verification Ping"},
                            },
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"✅ *Webhook is LIVE & CONNECTED!*\n*Time:* `{now_str}`\n*Target:* `{cfg.SLACK_WEBHOOK_URL}`\n*Status:* Ready for P0, P1, and P2 alerts.",
                                },
                            },
                        ],
                    }
                    async with httpx.AsyncClient(timeout=4.0) as client:
                        resp = await client.post(cfg.SLACK_WEBHOOK_URL, json=payload)
                        if resp.is_success:
                            return f"✅ **SUCCESS! HTTP {resp.status_code} sent to Webhook at {now_str}!**"
                        return f"❌ Webhook responded with error HTTP {resp.status_code}: {resp.text}"
                except Exception as exc:
                    return f"❌ Webhook network failure: {exc}"

            btn_test_ping.click(handle_test_ping, outputs=[webhook_ping_result])

        with gr.Tabs():
            # ==================================================================
            # Tab 1: Customer Support Portal
            # ==================================================================
            with gr.Tab("📩 Submit Ticket (Customer Portal)"):
                with gr.Row():
                    with gr.Column(scale=3):
                        user_input = gr.Textbox(
                            label="Customer Message / Query",
                            placeholder="Type your message here (English, Hinglish, French, Spanish, Hindi, etc.)...",
                            lines=5,
                        )
                        with gr.Row():
                            btn_send = gr.Button("Submit Ticket", variant="primary")
                            btn_clear = gr.Button("Clear")

                        gr.Markdown("#### Quick Test Scenarios (P0 / P1 / P2)")
                        with gr.Row():
                            btn_scen_auto = gr.Button("💰 Refund $35 (P2 Auto)", size="sm")
                            btn_scen_cancel = gr.Button("❌ Cancel Plan (P2 Auto)", size="sm")
                            btn_scen_p1 = gr.Button("⚠️ Disputed Charge $180 (P1 Review)", size="sm")
                            btn_scen_p0 = gr.Button("🚨 SSO Lockout (P0 Emergency)", size="sm")

                    with gr.Column(scale=2):
                        out_reply = gr.Textbox(label="Agent Response", lines=6, interactive=False)
                        with gr.Accordion("🛠️ Technical Details (API Payload)", open=False):
                            out_details = gr.JSON(label="Payload")

                async def handle_submit(text: str):
                    if not text.strip():
                        return "Please enter a message.", {}

                    # Automatically generate customer reference without forcing manual input
                    auto_cust_id = f"CUST-{uuid.uuid4().hex[:6].upper()}"
                    req = TicketIntakeRequest(text=text, customer_id=auto_cust_id)
                    res = await workflow_service.intake_ticket(req)

                    return res.reply or "", res.model_dump()

                btn_send.click(handle_submit, inputs=[user_input], outputs=[out_reply, out_details])
                btn_clear.click(
                    lambda: ("", "", {}),
                    outputs=[user_input, out_reply, out_details],
                )

                # Quick Scenario wiring
                btn_scen_auto.click(
                    lambda: "I was charged twice yesterday for $35.00, please refund my money.",
                    outputs=[user_input],
                )
                btn_scen_cancel.click(
                    lambda: "I want to cancel my subscription at the end of the current billing cycle.",
                    outputs=[user_input],
                )
                btn_scen_p1.click(
                    lambda: "I was charged $180 unexpectedly for a team license that I never authorized. Reverse this charge immediately.",
                    outputs=[user_input],
                )
                btn_scen_p0.click(
                    lambda: "URGENT! Entire engineering team locked out of Okta SSO right now. P0 security blocker.",
                    outputs=[user_input],
                )

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
                    rev_notes = gr.Textbox(
                        label="Reviewer Justification Notes",
                        placeholder="e.g. Approved exception after reviewing payment receipt.",
                    )

                    with gr.Row():
                        btn_approve = gr.Button("✅ Approve & Execute Action (Dispatches Tool & Webhook)", variant="primary")
                        btn_reject = gr.Button("❌ Reject & Deny Request", variant="stop")

                    rev_result = gr.Markdown("")

                async def refresh_queue():
                    pending = await workflow_service.list_pending()
                    options = [
                        f"{t.ticket_id} | [{t.priority}] | {t.action.upper()} | {t.detected_language} | {t.text[:35]}..."
                        for t in pending
                    ]
                    return gr.Dropdown(choices=options, value=options[0] if options else None)

                async def inspect_ticket(selected_opt: str | None):
                    if not selected_opt:
                        return "*No ticket selected.*", "", ""
                    t_id = selected_opt.split(" | ")[0]
                    pending = await workflow_service.list_pending()
                    item = next((t for t in pending if t.ticket_id == t_id), None)
                    if not item:
                        return "*Ticket not found.*", "", ""

                    policies_md = "\n".join([f"- **{p.get('source', 'Policy')}:** {p.get('title', '')}" for p in item.retrieved_policies])
                    info_md = (
                        f"#### Ticket: `{item.ticket_id}` (Customer: `{item.customer_id}`)\n"
                        f"* **Severity Priority:** **{item.priority}**\n"
                        f"* **Message:** \"{item.text}\"\n"
                        f"* **Action:** `{item.action}` | **Confidence:** `{item.confidence:.2f}`\n"
                        f"* **Detected Language:** `{item.detected_language}`\n"
                        f"* **Amount:** `${item.amount:.2f}`" if item.amount else "* **Amount:** N/A\n"
                        f"* **Interruption Reason:** {item.reason}\n\n"
                        f"**Grounding Policies Retrieved:**\n{policies_md}"
                    )
                    return info_md, item.draft_reply or "", ""

                async def process_human_verdict(selected_opt: str | None, draft: str, notes: str, approved: bool):
                    if not selected_opt:
                        return "⚠️ Please select a ticket first."
                    t_id = selected_opt.split(" | ")[0]
                    pending = await workflow_service.list_pending()
                    item = next((t for t in pending if t.ticket_id == t_id), None)
                    if not item:
                        return f"⚠️ Ticket `{t_id}` no longer found in pending queue."

                    rev_req = ReviewActionRequest(
                        approved=approved,
                        edited_reply=draft if draft else None,
                        notes=notes if notes else None,
                    )

                    res = await workflow_service.review_ticket(item.ticket_id, rev_req)
                    verdict_str = "APPROVED & EXECUTED" if approved else "REJECTED & DENIED"

                    slack_info = "N/A"
                    if res.tool_result and isinstance(res.tool_result, dict):
                        data = res.tool_result.get("data", {})
                        if isinstance(data, dict):
                            slack_info = data.get("slack_alert", "SENT")

                    return (
                        f"### ✅ Verdict [{verdict_str}] Recorded for Ticket `{item.ticket_id}`\n"
                        f"* **Priority:** `{res.priority}`\n"
                        f"* **Action:** `{res.decision_action}`\n"
                        f"* **Tool Status:** `{res.tool_result.get('status') if res.tool_result else 'DENIED'}`\n"
                        f"* **Webhook Dispatch:** `{slack_info}`\n"
                        f"* **Final Reply:** \"{res.reply}\""
                    )

                async def handle_approve_click(sel: str, draft: str, notes: str):
                    return await process_human_verdict(sel, draft, notes, approved=True)

                async def handle_reject_click(sel: str, draft: str, notes: str):
                    return await process_human_verdict(sel, draft, notes, approved=False)

                btn_refresh.click(refresh_queue, outputs=[pending_dropdown])
                pending_dropdown.change(inspect_ticket, inputs=[pending_dropdown], outputs=[rev_info, rev_draft, rev_notes])
                btn_approve.click(
                    handle_approve_click,
                    inputs=[pending_dropdown, rev_draft, rev_notes],
                    outputs=[rev_result],
                )
                btn_reject.click(
                    handle_reject_click,
                    inputs=[pending_dropdown, rev_draft, rev_notes],
                    outputs=[rev_result],
                )

        return demo
