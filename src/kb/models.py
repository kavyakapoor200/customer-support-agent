"""Data models for Policy Knowledge Base retrieval."""
from typing import Any

from pydantic import BaseModel, Field


class PolicySnippet(BaseModel):
    """Retrieved policy snippet matched against a customer query."""
    policy_id: str = Field(..., description="Unique identifier of the policy chunk.")
    title: str = Field(..., description="Title or section heading of the policy.")
    content: str = Field(..., description="Markdown text content of the policy rule.")
    score: float = Field(..., ge=0.0, le=1.0, description="Semantic similarity match score.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional policy metadata.")
