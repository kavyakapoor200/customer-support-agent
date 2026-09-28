"""CLI probe for testing tools directly."""
import argparse
import json

from src.tools.mock_tools import cancel_subscription, escalate_to_team, execute_refund


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe support tools directly.")
    parser.add_argument("--tool", choices=["refund", "cancel", "escalate"], required=True)
    parser.add_argument("--ticket-id", default="TK-TEST-101")
    parser.add_argument("--amount", type=float, default=49.00)
    parser.add_argument("--reason", default="Customer requested refund within 14-day SLA")
    parser.add_argument("--customer-id", default="CUST-8812")
    parser.add_argument("--immediate", action="store_true")
    parser.add_argument("--team", default="SecOps")
    parser.add_argument("--priority", default="P0")
    parser.add_argument("--notes", default="Automated test escalation")

    args = parser.parse_args()

    if args.tool == "refund":
        res = execute_refund(args.ticket_id, args.amount, args.reason)
    elif args.tool == "cancel":
        res = cancel_subscription(args.ticket_id, args.customer_id, args.immediate)
    else:
        res = escalate_to_team(args.ticket_id, args.team, args.priority, args.notes)

    print(json.dumps(res.model_dump(), indent=2))


if __name__ == "__main__":
    main()
