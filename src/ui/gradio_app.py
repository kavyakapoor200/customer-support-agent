import datetime
from typing import Any

import gradio as gr
import httpx

from src.api.models import ReviewActionRequest, TicketIntakeRequest
from src.api.service import workflow_service
from src.core.config import get_settings


def format_trace_stepper(res: Any) -> str:
    """Formats the LangGraph state trajectory into a clean, human-readable visual stepper."""
    traj = getattr(res, "trajectory", []) or []
    if not traj:
        return "*No trajectory recorded.*"

    steps_md = []
    traj_dict = {item.get("node"): item for item in traj}

    # 1. Intake
    intake = traj_dict.get("intake_node", {})
    lang = intake.get("detected_language", getattr(res, "detected_language", "english"))
    script = intake.get("detected_script", "latin")
    amt = intake.get("extracted_amount")
    amt_str = f"${amt:.2f}" if amt is not None else "None detected"
    steps_md.append(
        f"**1️⃣ 📥 NLU & Intake Stage**\n"
        f"* **Detected Language:** `{lang.upper()}` (Script: `{script}`)\n"
        f"* **Extracted Currency Amount:** `{amt_str}`"
    )

    # 2. Decision Engine
    dec = traj_dict.get("decide_node", {})
    action = dec.get("action", getattr(res, "decision_action", "unknown"))
    conf = dec.get("confidence", getattr(res, "decision_confidence", 0.0))
    engine = dec.get("engine", "kev")
    latency = dec.get("latency_ms", "N/A")
    backend_label = "Kev-0.8B (Local System 1)" if "kev" in str(engine).lower() else f"Cloud Baseline ({engine})"
    steps_md.append(
        f"**2️⃣ 🧠 Decision Engine (System 1)**\n"
        f"* **Inference Backend:** `{backend_label}` ({latency} ms)\n"
        f"* **Classified Action:** `{action}` (Confidence: `{conf * 100:.1f}%`)"
    )

    # 3. Policy Retrieval
    ret = traj_dict.get("retrieve_policy_node", {})
    top_p = ret.get("top_policy", "Customer Service SLA Policy")
    p_count = ret.get("matched_count", 0)
    steps_md.append(
        f"**3️⃣ 📚 Policy Grounding (Qdrant Vector DB)**\n"
        f"* **Relevant Knowledge Chunks:** `{p_count}` retrieved\n"
        f"* **Top Matched Policy:** *\"{top_p}\"*"
    )

    # 4. Deterministic Guardrail Gate
    gate = traj_dict.get("gate_node", {})
    req_human = gate.get("requires_human", getattr(res, "requires_human_review", False))
    rationale = gate.get("rationale", "Rule threshold evaluated")
    gate_badge = "⚠️ <span style='color:#e67e22; font-weight:bold;'>HUMAN SUPERVISION REQUIRED</span>" if req_human else "⚡ <span style='color:#27ae60; font-weight:bold;'>AUTO-EXECUTION APPROVED</span>"
    steps_md.append(
        f"**4️⃣ 🛡️ Deterministic Guardrail Gate (Code Enforced)**\n"
        f"* **Verdict:** {gate_badge}\n"
        f"* **Enforcement Rationale:** {rationale}"
    )

    # 5. Dynamic Response Synthesis
    draft = traj_dict.get("draft_node", {})
    is_live = draft.get("live_llm", True)
    synth_lang = draft.get("language", lang)
    verify = traj_dict.get("verify_node", {})
    v_pass = verify.get("verification_passed", True)
    v_str = "✅ Invariant Compliance Passed" if v_pass else "⚠️ Guardrail Warning"
    steps_md.append(
        f"**5️⃣ ✍️ AI Response Synthesis & Compliance**\n"
        f"* **Generator:** `{'Live Dynamic Groq LLM' if is_live else 'Rule-Based Fallback'}`\n"
        f"* **Target Language:** `{synth_lang.upper()}`\n"
        f"* **Verification:** {v_str}"
    )

    # 6. Action Execution / Tool
    if req_human and not traj_dict.get("human_review_node"):
        steps_md.append(
            "**6️⃣ ⏸️ LangGraph State Machine Interrupted**\n"
            "* **Current State:** Execution paused at human checkpoint.\n"
            "* **Notification:** Real-time Webhook alert paged to support desk.\n"
            "* **Next Action:** Open the **'🧑‍💼 Agent Review Desk'** tab above to approve or reject."
        )
    else:
        exec_step = traj_dict.get("execute_node", {})
        final_act = exec_step.get("final_action", getattr(res, "decision_action", "unknown"))
        tool_res = getattr(res, "tool_result", None)
        tool_status = tool_res.get("status", "SUCCESS") if isinstance(tool_res, dict) else "COMPLETED"
        steps_md.append(
            f"**6️⃣ ⚡ Tool Execution & Webhook Dispatch**\n"
            f"* **Dispatched Tool:** `{final_act}`\n"
            f"* **Execution Result:** `{tool_status}`\n"
            f"* **Audit Log:** Recorded to PostgreSQL & OpenTelemetry"
        )

    return "\n\n---\n\n".join(steps_md)


def create_gradio_ui() -> gr.Blocks:
    """Builds the dual-tab Gradio web interface with webhook diagnostics."""
    with gr.Blocks(title="Customer Support Agent Portal") as demo:
        gr.Markdown(
            "# 🛡️ Customer Support AI Agent\n"
            "**System 1 Gated Decision Engine · LangGraph Human-in-the-Loop Supervision · Real-Time Webhook Alerting**"
        )

        # ======================================================================
        # Webhook Connection & Test Bar
        # ======================================================================
        settings = get_settings()
        webhook_target = settings.SLACK_WEBHOOK_URL or "Not Configured (simulated in console)"
        with gr.Accordion("🔔 Webhook Live Diagnostics", open=True):
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
                                "text": {"type": "plain_text", "text": "🔔 Webhook Verification Ping"}
                            },
                            {
                                "type": "section",
                                "text": {
                                    "type": "mrkdwn",
                                    "text": f"✅ *Webhook is LIVE & CONNECTED!*\n*Time:* `{now_str}`\n*Target:* `{cfg.SLACK_WEBHOOK_URL}`\n*Status:* System ready to receive P0 and Human Review escalations."
                                }
                            }
                        ]
                    }
                    async with httpx.AsyncClient(timeout=4.0) as client:
                        resp = await client.post(cfg.SLACK_WEBHOOK_URL, json=payload)
                        if resp.is_success:
                            return f"✅ **SUCCESS! HTTP {resp.status_code} sent to Webhook at {now_str}!** Check your webhook.site dashboard!"
                        else:
                            return f"❌ Webhook responded with error HTTP {resp.status_code}: {resp.text}"
                except Exception as exc:
                    return f"❌ Webhook network failure: {exc}"

            btn_test_ping.click(handle_test_ping, outputs=[webhook_ping_result])

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
                            btn_scen_auto = gr.Button("💰 Refund $35 (Auto)", size="sm")
                            btn_scen_cancel = gr.Button("❌ Cancel Plan", size="sm")
                            btn_scen_high = gr.Button("⚠️ Overcharge $180 (Needs Review)", size="sm")
                            btn_scen_p0 = gr.Button("🚨 SSO Lockout (P0 Webhook Alert)", size="sm")

                    with gr.Column(scale=3):
                        out_status = gr.Markdown("### Status\n*No ticket submitted yet.*")
                        out_reply = gr.Textbox(label="Agent Response (Dynamic AI)", lines=4, interactive=False)

                gr.Markdown("### 🔍 Live Execution Trace (Step-by-Step Pipeline)")
                out_trace = gr.Markdown("*Submit a ticket above to see the node-by-node execution trajectory.*")

                with gr.Accordion("🛠️ Raw JSON State (for Debugging)", open=False):
                    out_details = gr.JSON(label="State Dump")

                async def handle_submit(text: str):
                    if not text.strip():
                        return "### Status\n*Please enter a message.*", "", "*No ticket submitted.*", {}
                    req = TicketIntakeRequest(text=text)
                    res = await workflow_service.intake_ticket(req)

                    badge = (
                        "<span class='badge-review' style='color:#e67e22; font-weight:bold;'>⚠️ ROUTED TO AGENT REVIEW DESK (Webhook Alert Paged)</span>"
                        if res.requires_human_review
                        else "<span class='badge-auto' style='color:#27ae60; font-weight:bold;'>✅ AUTO-RESOLVED</span>"
                    )

                    status_md = (
                        f"### Status: {badge}\n"
                        f"* **Ticket ID:** `{res.ticket_id}`\n"
                        f"* **Detected Action:** `{res.decision_action}` (Confidence: {res.decision_confidence:.2f})\n"
                        f"* **Detected Language:** `{res.detected_language}`\n"
                        f"* **Gating Outcome:** `{res.gating_outcome}`"
                    )

                    trace_md = format_trace_stepper(res)
                    return status_md, res.reply or "", trace_md, res.model_dump()

                btn_send.click(handle_submit, inputs=[user_input], outputs=[out_status, out_reply, out_trace, out_details])
                btn_clear.click(lambda: ("", "### Status\n*Cleared.*", "", "*No ticket submitted.*", {}), outputs=[user_input, out_status, out_reply, out_trace, out_details])

                # Quick Scenario wiring
                btn_scen_auto.click(lambda: "Mera renewal galti se ho gaya kal, $35 charge hua hai, refund kardo bhai please.", outputs=[user_input])
                btn_scen_cancel.click(lambda: "I want to cancel my subscription at the end of the billing cycle.", outputs=[user_input])
                btn_scen_high.click(lambda: "I was charged $180 unexpectedly for a team plan. Reverse this charge.", outputs=[user_input])
                btn_scen_p0.click(lambda: "URGENT! Entire engineering team locked out of Okta SSO right now. P0 security blocker.", outputs=[user_input])

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
                        btn_approve = gr.Button("✅ Approve & Execute Action (Dispatches Tool & Webhook)", variant="primary")
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

                    slack_info = "N/A"
                    if res.tool_result and isinstance(res.tool_result, dict):
                        data = res.tool_result.get("data", {})
                        if isinstance(data, dict):
                            slack_info = data.get("slack_alert", "SENT")

                    return (
                        f"### ✅ Verdict [{verdict_str}] Recorded for Ticket `{item.ticket_id}`\n"
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

            # ==================================================================
            # Tab 3: Observability Guide (Jaeger vs Agent Traces)
            # ==================================================================
            with gr.Tab("📊 Traces & Observability Guide"):
                gr.Markdown(
                    "## 🔭 Understanding Support Agent Observability\n\n"
                    "### ❓ Why do Jaeger Traces (`http://localhost:16686`) look so complex?\n"
                    "**Jaeger** is built for **SREs and Cloud DevOps Engineers** to monitor low-level network performance, "
                    "database query times, and HTTP socket connections. It tracks every microsecond span across our FastAPI backend, "
                    "Qdrant vector store, and PostgreSQL database.\n\n"
                    "#### 💡 How to read Jaeger in 30 seconds:\n"
                    "1. Open [Jaeger UI (http://localhost:16686)](http://localhost:16686).\n"
                    "2. Under **Service**, choose `customer-support-agent` and click **Find Traces**.\n"
                    "3. Click any trace: look at the **horizontal duration bar** to see how many milliseconds each step took.\n"
                    "4. If there's ever a failure, a **red exclamation mark** will highlight exactly which HTTP call or DB query failed.\n\n"
                    "---\n\n"
                    "### 🚀 In-App Live Execution Trace (What actually matters for Agent logic)\n"
                    "For understanding the **AI Decision Pipeline**, you do **not** need to struggle with Jaeger! "
                    "Our **Live Execution Trace** in the Customer Portal tab gives you full visibility:\n\n"
                    "| Step | Component | Role in Pipeline |\n"
                    "| :--- | :--- | :--- |\n"
                    "| **1. Intake & NLU** | Language & Entity Extractor | Detects customer language (Hinglish/Hindi/English) and extracts currency amounts. |\n"
                    "| **2. Decision Engine** | Jared Palmer Kev-0.8B (Local) | System 1 single-pass probability scoring over candidate actions. |\n"
                    "| **3. Policy Grounding** | Qdrant Vector DB | Retrieves exact company refund, cancellation, and security SLA policy markdown chunks. |\n"
                    "| **4. Guardrail Gate** | Deterministic Code (YAML) | Enforces hard threshold rules (e.g. refund <= $100 auto-execute, > $100 requires supervisor). |\n"
                    "| **5. AI Synthesis** | Dynamic LLM Response | Writes empathetic, native multilingual replies grounded in retrieved policy rules. |\n"
                    "| **6. Execution & Webhook**| Action Tool / Webhook Alert | Executes database transactions and dispatches instant webhook alerts to supervisor desk. |\n"
                )

        return demo
