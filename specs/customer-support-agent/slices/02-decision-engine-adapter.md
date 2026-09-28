# Slice 02: DecisionEngine Adapter (Kev, Jev, Mock, LLM Baseline)

## 1. Contract Unlocked
Provides a swappable, calibrated decision interface adhering to the TypeSafe `/v1/systemone` contract, enabling sub-50ms System 1 classification with deterministic CI mock fallback.

## 2. API Seam & Module Ownership
* **Module:** `src/decision_engine/`
* **Base Contract (`src/decision_engine/base.py`):**
  ```python
  class DecisionOutput(BaseModel):
      action: str
      confidence: float
      probabilities: dict[str, float]
      raw_scores: dict[str, float]
      engine_name: str
      latency_ms: float

  class BaseDecisionEngine(ABC):
      @abstractmethod
      async def decide(self, text: str, candidate_actions: list[str]) -> DecisionOutput:
          pass
  ```
* **Implementations (`src/decision_engine/backends/`):**
  * `MockDecisionEngine`: Deterministic keyword/hash-based calibrated probabilities for fast, zero-weight test & CI runs.
  * `KevDecisionEngine`: Connects to local Kev-0.8B running on `http://localhost:11434` or TypeSafe server; handles `/v1/systemone` payload formatting.
  * `JevDecisionEngine`: Remote API backend with bearer token auth.
  * `GroqBaselineEngine`: Fallback/eval judge using LiteLLM structured outputs (`llama-3.3-70b-versatile`).
* **Factory (`src/decision_engine/factory.py`):**
  * `get_decision_engine(backend: str | None = None) -> BaseDecisionEngine`

## 3. What the Human Can Run or See
* Run the standalone CLI probe:
  `python -m src.decision_engine.cli --text "Please refund my payment" --backend mock`
  Outputs formatted JSON showing action, confidence, and latency breakdown.

## 4. Verification Gates & Tests
* `tests/test_decision_engine.py`:
  * Tests that `MockDecisionEngine` returns valid probability distributions that sum to 1.0 ($\pm 1e-4$).
  * Asserts option-order invariance (permuting candidates produces consistent action).
  * Validates graceful fallback behavior when remote engines are unreachable.
* Command gate: `pytest tests/test_decision_engine.py` passes 100%.

## 5. Delegated Implementer Discretion
* Internal caching of HTTP client sessions (`httpx.AsyncClient`).
* Logging format for decision latency metrics.

## 6. What Must Stay Green
* Tests must run completely offline using `MockDecisionEngine`.
* Output model must always return normalized probabilities ($0.0 \le p \le 1.0$).

## 7. Non-blocking Checkpoint
Review candidate actions taxonomy: `["refund", "cancel_subscription", "billing_dispute", "account_escalation", "general_inquiry"]`. Proceed with this taxonomy if unedited.
