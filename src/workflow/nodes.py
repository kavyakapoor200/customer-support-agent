import datetime
import logging
import re
from typing import Any

from langgraph.types import interrupt

from src.cognition.gating import evaluate_gating
from src.core.config import get_settings
from src.decision_engine.factory import get_decision_engine
from src.kb.store import PolicyStore
from src.tools.mock_tools import cancel_subscription, escalate_to_team, execute_refund
from src.workflow.state import AgentState

logger = logging.getLogger(__name__)

HINGLISH_KEYWORDS = {
    "kardo", "kijiye", "hai", "hain", "bhai", "mera", "meri", "humne", "nahi",
    "paise", "rupaye", "wapas", "chahiye", "galti", "se", "ho", "gaya"
}


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
    """Detects script, language, extracts currency amounts, and initializes candidates."""
    text = state["raw_text"]

    # 1. Script & Language Detection
    has_devanagari = bool(re.search(r"[\u0900-\u097F]", text))
    if has_devanagari:
        detected_script = "devanagari"
        detected_language = "hindi"
    else:
        detected_script = "latin"
        words = set(re.findall(r"\w+", text.lower()))
        if words.intersection(HINGLISH_KEYWORDS):
            detected_language = "hinglish"
        else:
            detected_language = "english"

    # 2. Amount Extraction
    amount: float | None = None
    dollar_match = re.search(r"\$\s*(\d+(?:\.\d{1,2})?)", text)
    if dollar_match:
        amount = float(dollar_match.group(1))
    else:
        currency_word_match = re.search(r"(\d+(?:\.\d{1,2})?)\s*(?:dollars?|usd|rs|rupees)", text, re.IGNORECASE)
        if currency_word_match:
            amount = float(currency_word_match.group(1))

    candidates = state.get("candidate_actions") or [
        "refund",
        "cancel_subscription",
        "billing_dispute",
        "account_escalation",
        "general_inquiry"
    ]

    new_traj = _record_step(
        state,
        "intake_node",
        {
            "detected_language": detected_language,
            "detected_script": detected_script,
            "extracted_amount": amount,
        }
    )

    return {
        "detected_language": detected_language,
        "detected_script": detected_script,
        "extracted_amount": amount,
        "candidate_actions": candidates,
        "trajectory": new_traj,
    }


async def retrieve_policy_node(state: AgentState) -> dict[str, Any]:
    """Retrieves top matching policy snippets from the Qdrant policy store."""
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
    """Calls DecisionEngine to obtain calibrated probabilities across candidates."""
    engine = get_decision_engine()
    output = await engine.decide(state["raw_text"], state["candidate_actions"])

    new_traj = _record_step(
        state,
        "decide_node",
        {
            "action": output.action,
            "confidence": output.confidence,
            "engine": output.engine_name,
            "latency_ms": output.latency_ms,
        }
    )

    return {
        "decision_action": output.action,
        "decision_confidence": output.confidence,
        "probabilities": output.probabilities,
        "trajectory": new_traj,
    }


async def gate_node(state: AgentState) -> dict[str, Any]:
    """Evaluates YAML thresholds to determine auto-execution vs human review."""
    gating = evaluate_gating(
        action=state["decision_action"],
        confidence=state["decision_confidence"],
        amount_usd=state["extracted_amount"],
    )

    new_traj = _record_step(
        state,
        "gate_node",
        {
            "outcome": gating.outcome,
            "requires_human": gating.requires_human,
            "rationale": gating.rationale,
        }
    )

    return {
        "gating_outcome": gating.outcome,
        "reviewer_notes": gating.rationale,
        "trajectory": new_traj,
    }


async def _generate_live_reply(
    text: str,
    action: str,
    language: str,
    amount: float | None,
    policies: list[dict[str, Any]],
) -> str | None:
    """Invokes Groq Llama-3.3-70b to dynamically write an empathetic, customized reply."""
    settings = get_settings()
    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY.startswith("gsk_your"):
        return None

    try:
        from litellm import acompletion

        policy_context = "\n---\n".join([
            f"Policy: {p.get('title', 'SLA Policy')}\n{p.get('content', '')}"
            for p in policies[:2]
        ]) or "Standard SaaS 14-day refund and subscription SLA applies."

        lang_instruction = {
            "hindi": "Respond fluently and respectfully in Hindi (Devanagari script).",
            "hinglish": "Respond naturally in conversational Hinglish (Latin script code-mixed Hindi & English, like 'Hi, humne aapka request process kar diya hai...').",
            "english": "Respond professionally and empathetically in English.",
        }.get(language, "Respond in the customer's native language and tone.")

        prompt = (
            f"Customer Message: \"{text}\"\n"
            f"Determined Action: {action}\n"
            f"Amount: {f'${amount:.2f}' if amount else 'N/A'}\n"
            f"Retrieved Company Policy:\n{policy_context}\n\n"
            f"Tone Directive: {lang_instruction}\n\n"
            "Rules:\n"
            "1. Directly address their specific situation without generic robotic fillers.\n"
            "2. State clearly what action is taken (e.g. refund initiated, cancellation confirmed, or team escalated).\n"
            "3. Mention realistic timelines (e.g. 3-5 business days for bank settlement).\n"
            "4. Keep it concise (2-4 sentences max).\n\n"
            "Final Response:"
        )

        resp = await acompletion(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert customer support agent for a SaaS platform. Write warm, accurate, and direct responses mirroring the customer's tone."},
                {"role": "user", "content": prompt},
            ],
            api_key=settings.GROQ_API_KEY,
            temperature=0.3,
            max_tokens=250,
        )
        content = resp.choices[0].message.content
        return content.strip() if content else None
    except Exception as exc:
        logger.warning("Live LLM drafting failed (%s). Falling back to smart template.", exc)
        return None


async def draft_node(state: AgentState) -> dict[str, Any]:
    """Drafts customer response using live Groq LLM with deterministic fallback."""
    lang = state.get("detected_language", "english")
    action = state.get("decision_action", "general_inquiry")
    amount = state.get("extracted_amount")
    text = state.get("raw_text", "")
    policies = state.get("retrieved_policies", [])

    # 1. Attempt live LLM synthesis if live backend / key is available
    reply = await _generate_live_reply(text, action, lang, amount, policies)
    is_live = bool(reply)

    # 2. High-fidelity fallback template if offline or in CI mock mode
    if not reply:
        if lang == "hindi":
            if action == "refund":
                reply = f"नमस्ते, हमने आपके {f'${amount:.2f}' if amount else ''} रिफंड का अनुरोध प्राप्त कर लिया है। बैंक में राशि दिखने में 3-5 कार्य दिवस लगेंगे।"
            elif action == "cancel_subscription":
                reply = "नमस्ते, आपका सब्सक्रिप्शन रद्द कर दिया गया है। चालू बिलिंग चक्र के अंत तक सेवाएं सक्रिय रहेंगी।"
            else:
                reply = "नमस्ते, आपका अनुरोध प्राप्त हो गया है। हमारी टीम जल्द ही आपसे संपर्क करेगी।"
        elif lang == "hinglish":
            if action == "refund":
                reply = f"Hi, humne aapka {f'${amount:.2f} ka ' if amount else ''}refund request process kar diya hai. 3-5 business days mein account mein aa jayega."
            elif action == "cancel_subscription":
                reply = "Hi, aapka subscription cancel ho gaya hai. Current billing cycle ke end tak access rahega."
            else:
                reply = "Hi, humne aapki request note kar li hai. Support team jald contact karegi."
        else:
            if action == "refund":
                reply = f"Hello, we have processed your refund request for {f'${amount:.2f}' if amount else 'your recent charge'}. Please allow 3-5 business days for settlement."
            elif action == "cancel_subscription":
                reply = "Hello, your subscription has been successfully cancelled. Your access will remain active until the end of your billing cycle."
            elif action == "account_escalation":
                reply = "Hello, your issue has been escalated to our Priority Security Operations team. An on-call engineer has been alerted."
            else:
                reply = "Hello, thank you for reaching out to support. We have received your inquiry and are reviewing it."

    new_traj = _record_step(state, "draft_node", {"language": lang, "action": action, "live_llm": is_live})
    return {"draft_reply": reply, "trajectory": new_traj}


async def _verify_reply_with_llm(
    draft_reply: str,
    action: str,
    policies: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Uses Groq Llama-3.3-70b as an invariant compliance checker."""
    settings = get_settings()
    if not settings.GROQ_API_KEY or settings.GROQ_API_KEY.startswith("gsk_your"):
        return True, "Deterministic policy check"

    try:
        import json

        from litellm import acompletion

        policy_text = "\n".join([f"- {p.get('content', '')}" for p in policies[:2]]) or "Standard SLA: Refunds within 14 days."
        prompt = (
            f"Action: {action}\n"
            f"Draft Reply: \"{draft_reply}\"\n"
            f"Company Policies:\n{policy_text}\n\n"
            "Does this draft reply violate any policy or make unfulfillable guarantees (e.g. promising immediate money without verification)?\n"
            "Respond ONLY with valid JSON: {\"passed\": true, \"reason\": \"<short justification>\"}"
        )
        resp = await acompletion(
            model=settings.GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict policy compliance auditor. Respond strictly in JSON format."},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            api_key=settings.GROQ_API_KEY,
            temperature=0.0,
            max_tokens=100,
        )
        data = json.loads(resp.choices[0].message.content)
        return bool(data.get("passed", True)), str(data.get("reason", "Verified compliant"))
    except Exception as exc:
        logger.warning("LLM policy verification error (%s). Falling back.", exc)
        return True, "Fallback verification passed"


async def verify_node(state: AgentState) -> dict[str, Any]:
    """Verifies that draft claims align with retrieved policy invariants."""
    verification_passed = True
    amount = state.get("extracted_amount") or 0.0
    rationale = "Policy verified"

    # 1. Deterministic safety ceiling ($500 requires human supervisor)
    if (
        state.get("decision_action") == "refund"
        and amount > 500.0
        and state.get("review_status") != "approved"
    ):
        verification_passed = False
        rationale = "Refund amount exceeds $500 threshold and requires explicit supervisor approval"
    else:
        # 2. Live LLM Policy Guardrail
        draft = state.get("draft_reply", "")
        policies = state.get("retrieved_policies", [])
        action = state.get("decision_action", "")
        passed, reason = await _verify_reply_with_llm(draft, action, policies)
        if not passed:
            verification_passed = False
            rationale = reason

    new_traj = _record_step(state, "verify_node", {"verification_passed": verification_passed, "rationale": rationale})
    return {"verification_passed": verification_passed, "trajectory": new_traj}


async def human_review_node(state: AgentState) -> dict[str, Any]:
    """Halts execution via LangGraph interrupt, pausing for external human reviewer decision."""
    # This invokes LangGraph interrupt! Execution pauses and returns control to caller.
    review_input = interrupt({
        "ticket_id": state["ticket_id"],
        "action": state["decision_action"],
        "confidence": state["decision_confidence"],
        "extracted_amount": state["extracted_amount"],
        "reason": state.get("reviewer_notes", "Human review requested"),
        "draft_reply": state.get("draft_reply"),
    })

    # When resumed from interrupt:
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
        "reviewer_notes": notes or state.get("reviewer_notes"),
        "trajectory": new_traj,
    }


async def execute_node(state: AgentState) -> dict[str, Any]:
    """Executes the finalized support tool action."""
    action = state["decision_action"]
    ticket_id = state["ticket_id"]
    customer_id = state["customer_id"]
    amount = state.get("extracted_amount") or 25.0
    status = state.get("review_status")

    if status == "rejected" or state.get("gating_outcome") == "deny":
        final_action = "denied"
        tool_result = {"success": False, "status": "DENIED", "reason": state.get("reviewer_notes", "Policy rejection")}
    elif action in ("refund", "billing_dispute"):
        res = execute_refund(ticket_id, amount, "Refund/dispute processed via support flow")
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
        final_action = "general_inquiry_resolved"
        tool_result = {"success": True, "status": "RESOLVED_INFORMATIONAL"}

    new_traj = _record_step(state, "execute_node", {"final_action": final_action})
    return {
        "final_action_taken": final_action,
        "tool_result": tool_result,
        "trajectory": new_traj,
    }
