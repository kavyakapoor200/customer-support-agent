import datetime
import uuid

import gradio as gr
import httpx

from src.api.models import ReviewActionRequest, TicketIntakeRequest
from src.api.service import workflow_service
from src.core.config import get_settings

CUSTOM_CSS = """
/* Enterprise Premium Theme Adjustments */
body, .gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

.ticket-panel {
    background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
    border-radius: 12px;
    border: 1px solid rgba(226, 232, 240, 0.2);
    padding: 18px;
    margin-top: 12px;
}

.metric-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 600;
}

.badge-p0 {
    background-color: #fee2e2;
    color: #b91c1c;
    border: 1px solid #f87171;
}

.badge-p1 {
    background-color: #ffedd5;
    color: #c2410c;
    border: 1px solid #fb923c;
}

.badge-p2 {
    background-color: #ecfdf5;
    color: #047857;
    border: 1px solid #34d399;
}
"""


def create_gradio_ui() -> gr.Blocks:
    """Builds the query-based two-speed support UI matching Jev experimentation."""
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    )

    with gr.Blocks(title="Two-Speed Customer Support Agent", theme=theme, css=CUSTOM_CSS) as demo:
        gr.Markdown(
            "# ⚡ Customer Support AI Agent\n"
            "**Two-Speed Architecture: Fast Calibrated System 1 Decision Engine (Choice, Score, Noul) + System 2 Open LLM Responder (Groq)**"
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
                                    "text": f"✅ *Webhook is LIVE & CONNECTED!*\n*Time:* `{now_str}`\n*Target:* `{cfg.SLACK_WEBHOOK_URL}`\n*Status:* Ready for P0 Escalation alerts.",
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
                            placeholder="Type any inquiry, question, bug report, or issue in English, Hinglish, French, Spanish, Hindi, etc...",
                            lines=6,
                        )
                        with gr.Row():
                            btn_send = gr.Button("Submit Ticket", variant="primary")
                            btn_clear = gr.Button("Clear")

                        gr.Markdown("#### Sample Customer Inquiries (from Jev Experimentation)")
                        with gr.Row():
                            btn_scen_p0 = gr.Button("🚨 Outage & Churn (P0 Escalation)", size="sm")
                            btn_scen_p1 = gr.Button("⚙️ API 403 Blocker (P1 High Urgency)", size="sm")
                            btn_scen_p2 = gr.Button("📄 VAT Invoice Copy (P2 Routine)", size="sm")

                    with gr.Column(scale=2):
                        out_reply = gr.Textbox(label="Agent Response", lines=6, interactive=False)
                        out_trace = gr.Markdown("### 📋 Pipeline Telemetry\n*Submit a query to inspect live decision stages.*")
                        with gr.Accordion("🛠️ Technical Details (API Payload)", open=False):
                            out_details = gr.JSON(label="Payload")

                async def handle_submit(text: str):
                    if not text.strip():
                        return "Please enter a message.", "*No query provided.*", {}

                    auto_cust_id = f"CUST-{uuid.uuid4().hex[:6].upper()}"
                    req = TicketIntakeRequest(text=text, customer_id=auto_cust_id)
                    res = await workflow_service.intake_ticket(req)

                    priority_str = getattr(res, "priority", "P2")
                    if priority_str == "P0":
                        p_badge = "🔴 **P0 (CRITICAL ESCALATION)**"
                    elif priority_str == "P1":
                        p_badge = "🟠 **P1 (HIGH URGENCY)**"
                    else:
                        p_badge = "🟢 **P2 (ROUTINE)**"

                    action_str = getattr(res, "triage_action", "AUTOMATED_LLM_RESPONSE")
                    if action_str == "ESCALATE_HUMAN":
                        act_badge = "🚨 **ESCALATED TO HUMAN DESK**"
                    else:
                        act_badge = "🤖 **AUTOMATED LLM RESPONSE (Groq)**"

                    dept_str = getattr(res, "department", "general")
                    urgency_val = getattr(res, "urgency_score", 0.0)
                    urgency_desc = getattr(res, "urgency_description", "Normal")
                    churn_val = getattr(res, "churn_risk_probability", 0.0)

                    trace_md = (
                        f"### 📋 Pipeline Telemetry Trace\n"
                        f"* **Detected Language:** `{res.detected_language.upper()}`\n"
                        f"* **Department (Choice):** `{dept_str.upper()}`\n"
                        f"* **Urgency (Score 0-3):** `{urgency_val:.1f}/3` — *{urgency_desc}*\n"
                        f"* **Churn Risk (Noul):** `{churn_val:.1%}` probability\n"
                        f"* **Derived Priority:** {p_badge}\n"
                        f"* **Triage Action:** {act_badge}\n"
                        f"* **Ticket Reference:** `{res.ticket_id}`"
                    )

                    return res.reply or "", trace_md, res.model_dump()

                btn_send.click(handle_submit, inputs=[user_input], outputs=[out_reply, out_trace, out_details])
                btn_clear.click(
                    lambda: ("", "### 📋 Pipeline Telemetry\n*Cleared.*", {}),
                    outputs=[user_input, out_trace, out_details],
                )

                # Quick Sample wiring from Jev Experimentation
                btn_scen_p0.click(
                    lambda: (
                        "I've had enough of your broken API! Our production system went down for 4 hours today "
                        "and we lost over $30,000 in transactions. If this is not resolved immediately, I am cancelling "
                        "our enterprise contract and instructing our legal counsel to file for SLA breach."
                    ),
                    outputs=[user_input],
                )
                btn_scen_p1.click(
                    lambda: (
                        "Hi team, our developers are blocked trying to configure webhook endpoints on the sandbox environment. "
                        "The endpoint returns HTTP 403 Forbidden even with valid bearer tokens. Can you check our account permissions?"
                    ),
                    outputs=[user_input],
                )
                btn_scen_p2.click(
                    lambda: (
                        "Hello, could you please send me a PDF copy of last month's VAT invoice for our accounting records? "
                        "Thanks so much!"
                    ),
                    outputs=[user_input],
                )

            # ==================================================================
            # Tab 2: Agent Review Desk (Human-In-The-Loop)
            # ==================================================================
            with gr.Tab("🧑‍💼 Agent Review Desk"):
                gr.Markdown("### Human-In-The-Loop Review Queue (Paused P0 Escalations)")

                with gr.Row():
                    btn_refresh = gr.Button("🔄 Refresh Pending Queue", size="sm")

                pending_dropdown = gr.Dropdown(label="Select Escalated Ticket to Inspect", choices=[])

                with gr.Group():
                    rev_info = gr.Markdown("*Select a ticket from the dropdown above to inspect details.*")
                    rev_draft = gr.Textbox(label="Customer Reply (Editable by Reviewer)", lines=4)
                    rev_notes = gr.Textbox(
                        label="Reviewer Justification Notes",
                        placeholder="e.g. Account unlocked, SLA credit issued after reviewing incident.",
                    )

                    with gr.Row():
                        btn_approve = gr.Button("✅ Approve & Send Response", variant="primary")
                        btn_reject = gr.Button("❌ Reject & Deny Request", variant="stop")

                    rev_result = gr.Markdown("")

                async def refresh_queue():
                    pending = await workflow_service.list_pending()
                    options = [
                        f"{t.ticket_id} | [{t.priority}] | {t.department.upper()} | {t.detected_language} | {t.text[:35]}..."
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

                    info_md = (
                        f"#### Escalated Ticket: `{item.ticket_id}` (Customer: `{item.customer_id}`)\n"
                        f"* **Severity Priority:** **{item.priority}**\n"
                        f"* **Message:** \"{item.text}\"\n"
                        f"* **Department:** `{item.department.upper()}`\n"
                        f"* **Urgency Score:** `{item.urgency_score:.1f}/3`\n"
                        f"* **Churn Risk:** `{item.churn_risk:.1%}`\n"
                        f"* **Detected Language:** `{item.detected_language}`\n"
                        f"* **Escalation Reason:** {item.reason}"
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
                    verdict_str = "APPROVED & DISPATCHED" if approved else "REJECTED & CLOSED"

                    return (
                        f"### ✅ Verdict [{verdict_str}] Recorded for Ticket `{item.ticket_id}`\n"
                        f"* **Priority:** `{res.priority}`\n"
                        f"* **Department:** `{res.department}`\n"
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
