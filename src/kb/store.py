"""Policy Knowledge Base store wrapping Qdrant (in-memory and remote)."""
import hashlib
import math
import re
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from src.core.config import get_settings
from src.kb.models import PolicySnippet

VECTOR_DIM = 128
COLLECTION_NAME = "support_policies"


def _deterministic_embedding(text: str, dim: int = VECTOR_DIM) -> list[float]:
    """Generates a deterministic, normalized term-frequency embedding vector.

    Extracts word tokens and character n-grams to populate hashed vector bins,
    followed by L2 normalization. Guarantees cosine similarity without requiring
    heavy PyTorch or remote API downloads in tests and offline environments.
    """
    vector = [0.0] * dim
    words = re.findall(r"\w+", text.lower())

    for word in words:
        # Single word hash
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % dim
        vector[h] += 1.0

        # Subword 3-grams for morphology
        for i in range(len(word) - 2):
            trigram = word[i : i + 3]
            th = int(hashlib.md5(trigram.encode("utf-8")).hexdigest(), 16) % dim
            vector[th] += 0.5

    # L2 Normalization
    magnitude = math.sqrt(sum(x * x for x in vector))
    if magnitude > 0:
        vector = [round(x / magnitude, 5) for x in vector]
    else:
        vector = [0.0] * dim
        vector[0] = 1.0
    return vector


class PolicyStore:
    """Manages ingestion and semantic retrieval of policy documents using Qdrant."""

    def __init__(self, url: str | None = None) -> None:
        target_url = url or get_settings().QDRANT_URL
        if target_url == ":memory:":
            self.client = QdrantClient(":memory:")
        else:
            self.client = QdrantClient(url=target_url)

        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Creates the Qdrant collection if it does not already exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        if not exists:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(
                    size=VECTOR_DIM,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    def ingest_markdown_policies(self, directory: str | Path = "data/policies") -> int:
        """Parses markdown files in the directory and ingests sections into Qdrant.

        Returns:
            Total number of ingested policy chunks.
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise FileNotFoundError(f"Policies directory not found: {dir_path.resolve()}")

        points: list[qmodels.PointStruct] = []
        point_id = 1

        for md_file in sorted(dir_path.glob("*.md")):
            content = md_file.read_text(encoding="utf-8")
            # Split sections by '## ' headers
            sections = re.split(r"\n(?=##\s+)", content)
            file_title = md_file.stem.replace("_", " ").title()

            for idx, sec in enumerate(sections):
                sec = sec.strip()
                if not sec:
                    continue

                lines = sec.splitlines()
                header = lines[0].lstrip("#").strip() if lines else file_title
                chunk_id = f"{md_file.stem}_{idx + 1}"
                vector = _deterministic_embedding(sec)

                payload: dict[str, Any] = {
                    "policy_id": chunk_id,
                    "title": header,
                    "content": sec,
                    "source_file": md_file.name,
                }

                points.append(
                    qmodels.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                )
                point_id += 1

        if points:
            self.client.upsert(collection_name=COLLECTION_NAME, points=points)

        return len(points)

    def search_policies(self, query: str, limit: int = 3) -> list[PolicySnippet]:
        """Searches for relevant policy chunks using semantic similarity."""
        query_vector = _deterministic_embedding(query)

        # Use search or query_points based on client version
        try:
            results = self.client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=limit,
            )
        except AttributeError:
            # Fallback for newer Qdrant API
            response = self.client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=limit,
            )
            results = response.points

        snippets: list[PolicySnippet] = []
        for hit in results:
            payload = hit.payload or {}
            # Normalize cosine score to [0, 1] range
            normalized_score = max(0.0, min(1.0, (hit.score + 1.0) / 2.0)) if hit.score is not None else 0.5
            snippets.append(
                PolicySnippet(
                    policy_id=str(payload.get("policy_id", "unknown")),
                    title=str(payload.get("title", "Policy Rule")),
                    content=str(payload.get("content", "")),
                    score=round(normalized_score, 4),
                    metadata={"source_file": payload.get("source_file")},
                )
            )

        return snippets
