"""CLI probe for testing Policy Knowledge Base retrieval."""
import argparse

from src.kb.store import PolicyStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Query Policy Knowledge Base.")
    parser.add_argument("--query", type=str, required=True, help="Support policy query.")
    parser.add_argument("--limit", type=int, default=3, help="Max results to return.")
    parser.add_argument("--policies-dir", type=str, default="data/policies")

    args = parser.parse_args()

    store = PolicyStore(url=":memory:")
    count = store.ingest_markdown_policies(args.policies_dir)
    print(f"Ingested {count} policy sections into memory.")

    results = store.search_policies(args.query, limit=args.limit)

    for idx, snippet in enumerate(results, 1):
        print(f"\n[{idx}] {snippet.title} (Match Score: {snippet.score})")
        print(f"Policy ID: {snippet.policy_id}")
        print(f"Content:\n{snippet.content[:250]}...")


if __name__ == "__main__":
    main()
