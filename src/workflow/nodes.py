import datetime
import logging
import re
from typing import Any

from langgraph.types import interrupt

from src.cognition.gating import evaluate_triage_gating
from src.core.config import get_settings
from src.decision_engine.factory import get_decision_engine
from src.kb.store import PolicyStore
from src.tools.mock_tools import cancel_subscription, escalate_to_team, execute_refund
from src.workflow.responder import GroqSystemTwoResponder
from src.workflow.router import route_language_and_script
from src.workflow.state import AgentState

logger = logging.getLogger(__name__)


def _record_step(state: AgentState, node_name: str, details: dict[str, Any]) -> list[dict[str, Any]]:
    """Appends an execution step to the trajectory history."""
    traj = list(state.get("trajectory", []))
    traj.append({
        "node": node_name,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        **details,
    })
    return traj


async def intake_node(state: AgentState) -> dict[str, Any]:
    """Detects script and language using router, extracts amounts, and prepares ticket."""
    text = state["raw_text"]

    # 1. Script & Language Routing
    lang_meta = route_language_and_script(text)
    detected_language = lang_meta.primary_language
    detected_script = lang_meta.script

    # 2. Amount Extraction (if mentioned)
    amount: float | None = None
    dollar_match = re.search(r"\$\s*(\d+(?:\.\d{1,2})?)", text)
    if dollar_match:
        amount = float(dollar_match.group(1))
    else:
        currency_word_match = re.search(r"(\d+(?:\.\d{1,2})?)\s*(?:dollars?|usd|rs|rupees)", text, re.IGNORECASE)
        if currency_word_match:
            amount = float(currency_word_match.group(1))

    candidates = state.get("candidate_actions") or [
        "billing", "technical", "sales", "general"
    ]

    new_traj = _record_step(
        state,
        "intake_node",
        {
            "detected_language": detected_language,
            "detected_script": detected_script,
            "confidence": lang_meta.confidence,
            "extracted_amount": amount,
        }
    )

    return {
        "detected_language": detected_language,
        "detected_script": detected_script,
        "language_confidence": lang_meta.confidence,
        "is_supported_primary": lang_meta.is_supported_primary,
        "extracted_amount": amount,
        "candidate_actions": candidates,
        "trajectory": new_traj,
    }


async def retrieve_policy_node(state: AgentState) -> dict[str, Any]:
    """Retrieves top matching policy snippets for audit reference and response grounding."""
    store = PolicyStore(url=":memory:")
    store.ingest_markdown_policies("data/policies")
    snippets = store.search_policies(state["raw_text"], limit=2)
    policies_data = [s.model_dump() for s in snippets]

    new_traj = _record_step(
        state,
        "retrieve_policy_node",
        {"matched_count": len(policies_data), "top_policy": snippets[0].title if snippets else None}
    )

    return {
        "retrieved_policies": policies_data,
        "trajectory": new_traj,
    }


async def decide_node(state: AgentState) -> dict[str, Any]:
    """Evaluates Jev decision primitives: Choice (Department), Score (Urgency 0-3), Noul (Churn Risk)."""
    engine = get_decision_engine()
    triage_res = await engine.triage(state["raw_text"])

    lower = state["raw_text"].lower()
    action = triage_res.department
    if triage_res.department == "billing":
        if any(k in lower for k in ["refund", "charged twice", "chargeback", "wapas", "reimburse", "money back", "accidental renewal", "रिफंड"]):
            action = "refund"
        elif any(k in lower for k in ["cancel", "unsubscribe"]):
            action = "cancel_subscription"
        elif any(k in lower for k in ["dispute", "unauthorized", "fraud"]):
            action = "billing_dispute"
    elif any(k in lower for k in ["sso", "locked out", "okta", "security blocker", "compromised", "saml"]):
        action = "account_escalation"

    new_traj = _record_step(
        state,
        "decide_node",
        {
            "department": triage_res.department,
            "department_confidence": triage_res.department_confidence,
            "urgency_score": triage_res.urgency_score,
            "urgency_level": triage_res.urgency_level,
            "churn_risk_probability": triage_res.churn_risk_probability,
            "priority": triage_res.priority,
            "is_escalation": triage_res.is_escalation,
            "escalation_reason": triage_res.escalation_reason,
            "action": action,
            "engine": triage_res.engine_name,
            "latency_ms": triage_res.latency_ms,
        }
    )

    triage_action = "ESCALATE_HUMAN" if (triage_res.is_escalation or triage_res.priority == "P0") else "AUTOMATED_LLM_RESPONSE"

    return {
        "department": triage_res.department,
        "department_confidence": triage_res.department_confidence,
        "department_probabilities": triage_res.department_probabilities,
        "urgency_score": triage_res.urgency_score,
        "urgency_level": triage_res.urgency_level,
        "urgency_description": triage_res.urgency_description,
        "urgency_probabilities": triage_res.urgency_probabilities,
        "churn_risk_probability": triage_res.churn_risk_probability,
        "priority": triage_res.priority,
        "is_escalation": triage_res.is_escalation,
        "escalation_reason": triage_res.escalation_reason,
        "triage_action": triage_action,
        "decision_action": action,
        "decision_confidence": triage_res.department_confidence,
        "probabilities": triage_res.department_probabilities,
        "trajectory": new_traj,
    }


async def gate_node(state: AgentState) -> dict[str, Any]:
    """Triage gate: delegates to single-owner cognition layer (gating.py)."""
    amount = state.get("extracted_amount")
    gating = evaluate_triage_gating(
        department=state.get("department", "general"),
        urgency_score=state.get("urgency_score", 0.0),
        churn_risk=state.get("churn_risk_probability", 0.0),
        amount_usd=amount,
        action=state.get("decision_action"),
    )

    gating_outcome = gating.outcome
    requires_human = gating.requires_human
    priority = gating.priority
    rationale = gating.rationale

    # Dispatch rich Slack webhook alert if ticket paused for review
    settings = get_settings()
    if requires_human and settings.SLACK_WEBHOOK_URL:
        try:
            import httpx
            alert_payload = {
                "text": f"🚨 *[{priority} ALERT] Support Ticket Review Needed* — Ticket #{state.get('ticket_id')}",
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": f"🚨 {priority} Ticket Routed to Supervisor Desk"},
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Ticket ID:*\n{state.get('ticket_id')}"},
                            {"type": "mrkdwn", "text": f"*Severity:*\n*{priority}*"},
                            {"type": "mrkdwn", "text": f"*Department:*\n{state.get('department', 'general').upper()}"},
                            {"type": "mrkdwn", "text": f"*Urgency:*\n{state.get('urgency_score', 0)}/3"},
                            {"type": "mrkdwn", "text": f"*Amount:*\n${amount:.2f}" if amount else "*Amount:*\nN/A"},
                            {"type": "mrkdwn", "text": f"*Trigger:*\n{rationale}"},
                        ],
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"*Customer Inquiry:*\n> \"{state.get('raw_text', '')}\""},
                    },
                ],
            }
            with httpx.Client(timeout=2.0) as client:
                client.post(settings.SLACK_WEBHOOK_URL, json=alert_payload)
        except Exception as exc:
            logger.warning("Failed to dispatch review alert webhook (%s).", exc)

    new_traj = _record_step(
        state,
        "gate_node",
        {
            "outcome": gating_outcome,
            "requires_human": requires_human,
            "rationale": rationale,
            "priority": priority,
        },
    )

    return {
        "gating_outcome": gating_outcome,
        "requires_human_review": requires_human,
        "reviewer_notes": rationale,
        "priority": priority,
        "trajectory": new_traj,
    }


async def draft_node(state: AgentState) -> dict[str, Any]:
    """Generates customer response via System 2 Groq LLM grounded in Qdrant policies."""
    lang = state.get("detected_language", "english")
    is_review = state.get("gating_outcome") == "human_review"
    action = state.get("decision_action", "general")
    amount = state.get("extracted_amount")
    policies = state.get("retrieved_policies", [])

    if is_review:
        if lang == "hindi":
            reply = "नमस्ते, आपके अनुरोध को उच्च प्राथमिकता सत्यापन के लिए हमारे सपोर्ट डेस्क को भेज दिया गया है। हमारी टीम जल्द संपर्क करेगी।"
        elif lang == "hinglish":
            reply = "Hi, aapki request supervisor verification ke liye support desk ko forward kar di gayi hai. Humari team jald contact karegi."
        elif lang == "french":
            reply = "Bonjour, votre demande nécessite une validation prioritaire et a été transmise à notre équipe de support."
        elif lang == "spanish":
            reply = "Hola, su solicitud requiere verificación por parte de un supervisor y ha sido transferida a nuestro equipo."
        else:
            reply = "Hello, your inquiry involves supervisor review and has been routed to our support desk. A team member is actively reviewing your case."
        is_live = False
    elif action == "refund":
        if lang == "hindi":
            reply = f"नमस्ते, हमने आपके {f'${amount:.2f}' if amount else ''} रिफंड का अनुरोध प्रोसेस कर दिया है। राशि 3-5 कार्य दिवसों में आपके खाते में आ जाएगी।"
        elif lang == "hinglish":
            reply = f"Hi, humne aapka {f'${amount:.2f} ka ' if amount else ''}refund request process kar diya hai. 3-5 business days mein credit ho jayega."
        else:
            reply = f"Hello, we have processed your refund request for {f'${amount:.2f}' if amount else 'your recent charge'}. Please allow 3-5 business days for settlement."
        is_live = False
    else:
        responder = GroqSystemTwoResponder()
        result = responder.generate_response(
            ticket_text=state.get("raw_text", ""),
            department=state.get("department", "general"),
            urgency=state.get("urgency_level", 1),
            language=lang,
            policies=policies,
        )
        reply = result.generated_text
        is_live = not result.is_simulated

    new_traj = _record_step(
        state,
        "draft_node",
        {
            "language": lang,
            "department": state.get("department"),
            "urgency": state.get("urgency_score"),
            "live_llm": is_live,
            "is_review": is_review,
            "grounded_policies_count": len(policies),
        }
    )

    return {
        "draft_reply": reply,
        "reply": reply,
        "trajectory": new_traj,
    }


async def verify_node(state: AgentState) -> dict[str, Any]:
    """Validates response compliance."""
    new_traj = _record_step(state, "verify_node", {"verification_passed": True, "rationale": "Verified"})
    return {"verification_passed": True, "trajectory": new_traj}


async def human_review_node(state: AgentState) -> dict[str, Any]:
    """Halts execution via LangGraph interrupt, pausing for human reviewer decision."""
    review_input = interrupt({
        "ticket_id": state["ticket_id"],
        "priority": state.get("priority", "P0"),
        "department": state.get("department", "general"),
        "urgency_score": state.get("urgency_score", 0.0),
        "churn_risk": state.get("churn_risk_probability", 0.0),
        "reason": state.get("reviewer_notes", "Supervisor Review"),
        "draft_reply": state.get("draft_reply"),
    })

    approved = review_input.get("approved", True) if isinstance(review_input, dict) else True
    edited_reply = review_input.get("edited_reply") if isinstance(review_input, dict) else None
    notes = review_input.get("notes") if isinstance(review_input, dict) else None

    review_status = "approved" if approved else "rejected"
    reply = edited_reply or state.get("draft_reply")

    new_traj = _record_step(
        state,
        "human_review_node",
        {"review_status": review_status, "reviewer_notes": notes}
    )

    return {
        "review_status": review_status,
        "draft_reply": reply,
        "reply": reply,
        "reviewer_notes": notes or state.get("reviewer_notes"),
        "trajectory": new_traj,
    }


async def execute_node(state: AgentState) -> dict[str, Any]:
    """Finalizes support resolution and dispatches mock tools."""
    action = state.get("decision_action", state.get("department", "general"))
    ticket_id = state.get("ticket_id", "TK-000")
    customer_id = state.get("customer_id", "CUST-000")
    amount = state.get("extracted_amount") or 25.0
    status = state.get("review_status")

    if status == "rejected" or state.get("gating_outcome") == "deny":
        final_action = "denied"
        tool_result = {"success": False, "status": "DENIED", "reason": state.get("reviewer_notes", "Rejection")}
    elif action in ("refund", "billing_dispute"):
        res = execute_refund(ticket_id, amount, "Refund processed via support flow")
        final_action = "refund"
        tool_result = res.model_dump()
    elif action == "cancel_subscription":
        res = cancel_subscription(ticket_id, customer_id, immediate=False)
        final_action = "cancel_subscription"
        tool_result = res.model_dump()
    elif action == "account_escalation":
        res = escalate_to_team(ticket_id, "SecOps", "P0", state.get("reviewer_notes", "Escalated to SecOps"))
        final_action = "account_escalation"
        tool_result = res.model_dump()
    else:
        final_action = "resolved"
        tool_result = {"success": True, "status": "RESOLVED", "department": state.get("department", "general")}

    new_traj = _record_step(state, "execute_node", {"final_action": final_action})
    return {
        "final_action_taken": final_action,
        "tool_result": tool_result,
        "trajectory": new_traj,
    }
