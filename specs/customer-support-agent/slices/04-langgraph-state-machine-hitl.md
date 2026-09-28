# Slice 04: LangGraph State Machine & Human-In-The-Loop Gating

## 1. Contract Unlocked
Implements the core orchestration graph with deterministic gating, Postgres checkpointing, and non-blocking human-in-the-loop (HITL) execution interrupts.

## 2. API Seam & Module Ownership
* **Module:** `src/workflow/`
* **State Contract (`src/workflow/state.py`):**
  * `AgentState` TypedDict containing strictly JSON-serializable primitives:
    ```python
    class AgentState(TypedDict):
        ticket_id: str
        customer_id: str
        raw_text: str
        detected_language: str       # "english" | "hinglish" | "hindi"
        detected_script: str         # "latin" | "devanagari"
        extracted_amount: float | None
        decision_action: str
        decision_confidence: float
        probabilities: dict[str, float]
        retrieved_policies: list[dict[str, str]]
        gating_outcome: str          # "auto_execute" | "human_review" | "clarify" | "deny"
        review_status: str | None    # "pending" | "approved" | "rejected"
        reviewer_notes: str | None
        draft_reply: str | None
        verification_passed: bool
        final_action_taken: str | None
    ```
* **Graph Builder (`src/workflow/graph.py`):**
  * Nodes:
    1. `intake_node`: Unicode script detection & metadata tagging.
    2. `retrieve_policy_node`: Fetches top-2 policy chunks from `PolicyStore`.
    3. `decide_node`: Calls `DecisionEngine` for calibrated probabilities.
    4. `gate_node`: Applies `config/thresholds.yaml` rules.
    5. `human_review_node`: Invokes `interrupt({"ticket_id": ..., "reason": "Requires human sign-off"})`.
    6. `draft_node`: Generates response via Groq with tone/language mirroring instruction.
    7. `verify_node`: Compares draft claims against policy invariants (Noul-style verification).
    8. `execute_node`: Invokes corresponding mock tool.
    9. `log_trajectory_node`: Persists full trajectory to PostgreSQL.
* **Checkpointer:**
  * Uses `AsyncPostgresSaver` in production/docker; `MemorySaver` in headless tests.

## 3. What the Human Can Run or See
* Run the graph probe:
  `python -m src.workflow.cli --ticket "Accidentally renewed annual plan, charge was $45"`
  Outputs step-by-step trace showing auto-execution path.
* Run the interrupt probe:
  `python -m src.workflow.cli --ticket "Refund needed for $150 charge" --interrupt-test`
  Demonstrates graph halting at `human_review_node`, waiting for external resumption.

## 4. Verification Gates & Tests
* `tests/test_workflow.py`:
  * Tests the complete auto-execute path (<$50, high confidence -> status: completed).
  * Tests the HITL interrupt path (>$50 or confidence < 0.90 -> graph yields interrupt state).
  * Tests state resumption (simulated human approval resumes graph and completes execution).
  * Tests that `AgentState` contains zero unpicklable/unserializable objects.
* Command gate: `pytest tests/test_workflow.py` passes 100%.

## 5. Delegated Implementer Discretion
* Prompt structure for the Groq draft generator.
* Postgres table naming (`trajectories`, `tickets`, `audit_logs`).

## 6. What Must Stay Green
* Tests must run with `MemorySaver` without requiring a running Postgres instance.
* Interrupt resumption must preserve state continuity without dropping context.

## 7. Non-blocking Checkpoint
Review LangGraph node sequence. If no edits proposed, proceed with standard intake-to-audit flow.
