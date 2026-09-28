"""CLI probe for testing DecisionEngine backends directly."""
import argparse
import asyncio
import json

from src.decision_engine.factory import get_decision_engine

DEFAULT_CANDIDATES = [
    "refund",
    "cancel_subscription",
    "billing_dispute",
    "account_escalation",
    "general_inquiry"
]


async def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Test DecisionEngine backends directly.")
    parser.add_argument("--text", type=str, required=True, help="Ticket message to classify.")
    parser.add_argument("--backend", type=str, default="mock", choices=["mock", "kev", "jev", "groq"])
    parser.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES, help="Candidate actions.")

    args = parser.parse_args()

    engine = get_decision_engine(args.backend)
    result = await engine.decide(args.text, args.candidates)

    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    asyncio.run(run_cli())
