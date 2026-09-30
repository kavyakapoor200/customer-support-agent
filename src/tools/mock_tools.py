"""Core tool execution logic and sandbox simulation."""
import hashlib
import uuid

from src.tools.models import AuditEntry, ToolResult


def _generate_tx_hash(tool: str, ticket_id: str, params: dict) -> str:
    """Generates a reproducible unique audit transaction hash."""
    raw = f"{tool}:{ticket_id}:{sorted(params.items())}:{uuid.uuid4()}"
    return f"tx_{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:12]}"


def execute_refund(
    ticket_id: str,
    amount_usd: float,
    reason: str,
    max_amount_usd: float | None = None,
    authorized_by_human: bool = False,
) -> ToolResult:
    """Executes a financial refund against a customer ticket."""
    if amount_usd <= 0.0:
        return ToolResult(
            success=False,
            tool_name="refund",
            transaction_id="tx_failed",
            audit_entry={},
            error=f"Refund amount must be strictly greater than $0.00, got: {amount_usd}",
        )

    if max_amount_usd is None:
        try:
            from src.core.thresholds import load_thresholds

            policy = load_thresholds().get_policy("refund")
            max_amount_usd = policy.max_auto_amount_usd
        except Exception:
            max_amount_usd = 50.0

    if not authorized_by_human and max_amount_usd > 0.0 and amount_usd > max_amount_usd:
        return ToolResult(
            success=False,
            tool_name="refund",
            transaction_id="tx_failed",
            audit_entry={},
            error=f"Refund amount ${amount_usd:.2f} exceeds configured maximum ceiling of ${max_amount_usd:.2f}",
        )

    tx_hash = _generate_tx_hash("refund", ticket_id, {"amount": amount_usd, "reason": reason})
    audit = AuditEntry(
        tool_name="refund",
        action_type="FINANCIAL_REIMBURSEMENT",
        ticket_id=ticket_id,
        input_parameters={"amount_usd": amount_usd, "reason": reason},
        output_status="SUCCESS",
        transaction_hash=tx_hash,
    )

    return ToolResult(
        success=True,
        tool_name="refund",
        transaction_id=tx_hash,
        data={
            "refund_amount_usd": round(amount_usd, 2),
            "status": "COMPLETED",
            "estimated_settlement_days": "3-5 business days",
            "reason": reason,
        },
        audit_entry=audit.model_dump(),
    )


def cancel_subscription(ticket_id: str, customer_id: str, immediate: bool = False) -> ToolResult:
    """Cancels a customer subscription (immediate or end of billing cycle)."""
    tx_hash = _generate_tx_hash("cancel_subscription", ticket_id, {"immediate": immediate})
    effective_date = (
        "Immediately"
        if immediate
        else "At the end of current billing cycle (30 days access retained)"
    )

    audit = AuditEntry(
        tool_name="cancel_subscription",
        action_type="SUBSCRIPTION_TERMINATION",
        ticket_id=ticket_id,
        input_parameters={"customer_id": customer_id, "immediate": immediate},
        output_status="SUCCESS",
        transaction_hash=tx_hash,
    )

    return ToolResult(
        success=True,
        tool_name="cancel_subscription",
        transaction_id=tx_hash,
        data={
            "customer_id": customer_id,
            "status": "CANCELLED",
            "effective_date": effective_date,
            "data_retention_days": 30,
        },
        audit_entry=audit.model_dump(),
    )


def escalate_to_team(ticket_id: str, target_team: str, priority: str, notes: str) -> ToolResult:
    """Escalates a support ticket to internal teams (e.g. SecOps, BillingOps)."""
    valid_priorities = {"P0", "P1", "P2", "P3"}
    normalized_priority = priority.upper()
    if normalized_priority not in valid_priorities:
        return ToolResult(
            success=False,
            tool_name="escalate_to_team",
            transaction_id="tx_failed",
            audit_entry={},
            error=f"Invalid priority '{priority}'. Must be one of: {sorted(valid_priorities)}",
        )

    tx_hash = _generate_tx_hash("escalate_to_team", ticket_id, {"team": target_team, "priority": normalized_priority})
    
    # Slack Webhook Integration
    from src.core.config import get_settings
    settings = get_settings()
    slack_status = "NOT_CONFIGURED"
    
    if settings.SLACK_WEBHOOK_URL:
        try:
            import httpx
            slack_payload = {
                "name": f"🚨 [{normalized_priority}] Escalation #{ticket_id}: {notes[:60]}",
                "price": 0,
                "qty": 1,
                "text": f"🚨 *Support Escalation [{normalized_priority}]* — Ticket #{ticket_id}",
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": f"🚨 Support Escalation: {normalized_priority}"}
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Ticket ID:*\n{ticket_id}"},
                            {"type": "mrkdwn", "text": f"*Target Team:*\n{target_team}"},
                            {"type": "mrkdwn", "text": f"*Priority:*\n{normalized_priority}"},
                            {"type": "mrkdwn", "text": f"*Notes:*\n{notes}"}
                        ]
                    }
                ]
            }
            with httpx.Client(timeout=3.0) as client:
                resp = client.post(settings.SLACK_WEBHOOK_URL, json=slack_payload)
                slack_status = "SENT" if resp.is_success else f"FAILED_{resp.status_code}"
        except Exception as exc:
            slack_status = f"ERROR_{type(exc).__name__}"
    else:
        slack_status = "SIMULATED"

    audit = AuditEntry(
        tool_name="escalate_to_team",
        action_type="INTERNAL_ESCALATION",
        ticket_id=ticket_id,
        input_parameters={"target_team": target_team, "priority": normalized_priority, "notes": notes},
        output_status="PAGED" if normalized_priority == "P0" else "QUEUED",
        transaction_hash=tx_hash,
    )

    return ToolResult(
        success=True,
        tool_name="escalate_to_team",
        transaction_id=tx_hash,
        data={
            "escalated_to": target_team,
            "priority": normalized_priority,
            "pager_status": "PAGED_ONCALL" if normalized_priority == "P0" else "NOT_PAGED",
            "slack_alert": slack_status,
            "notes": notes,
        },
        audit_entry=audit.model_dump(),
    )
