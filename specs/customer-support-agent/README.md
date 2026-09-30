# Customer Support Agent — System Specification

## Next Agent Prompt

> **Instructions for the next agent:**
> 1. Read this master spec and the target slice file in `specs/customer-support-agent/slices/` before writing any code.
> 2. Pick up at the exact next incomplete slice in the **Global Slice Checklist** below.
> 3. Each slice must be fully verified and runnable before checking it off and advancing to the next.
> 4. Keep all state types strictly primitive (`str`, `float`, `int`, `bool`, `list`, `dict`) to guarantee 100% JSON-serializable LangGraph Postgres checkpoints.
> 5. Before concluding your pass, update this **Next Agent Prompt** section with:
>    - Current Status and Last Updated date.
>    - Completed items in the checklist.
>    - The exact next slice pickup point.

* **Current Status:** All 6 Slices complete and verified. 35/35 tests passing, 0 lint errors, 96.0% eval accuracy.
* **Last Updated:** 2026-09-29
* **Next Pickup Point:** Production ready. Ready for GitHub push.
* **Active Blockers / Warnings:** Zero.

### Global Slice Checklist
- [x] **Slice 01: Project Skeleton, Config & Docker Foundation** (`01-project-skeleton-and-config.md`)
- [x] **Slice 02: DecisionEngine Adapter (Kev, Jev, Mock, LLM Baseline)** (`02-decision-engine-adapter.md`)
- [x] **Slice 03: Policy KB (Qdrant) & Sandboxed Operational Tools** (`03-policy-kb-and-mock-tools.md`)
- [x] **Slice 04: LangGraph State Machine & Human-In-The-Loop Gating** (`04-langgraph-state-machine-hitl.md`)
- [x] **Slice 05: Interfaces (FastAPI & Gradio Dual-Portal)** (`05-fastapi-and-gradio-ui.md`)
- [x] **Slice 06: Synthetic Eval Harness, Benchmarks & GitHub Actions CI** (`06-eval-harness-and-ci.md`)

---

## 1. Goal & Product Vision

An enterprise-grade, calibrated **Customer Support Agent with Deterministic Gating and Human Supervision**.

* **Core Axiom:** *"Model scores, code decides"* — Large Language Models do not execute actions autonomously. Decisions (refund, cancel, escalate) are driven by calibrated probabilities from specialized small System 1 models (Kev-0.8B) checked against strict YAML thresholds.
* **Human-in-the-Loop:** High-risk actions ($> \$50$) or borderline confidence ($0.60 - 0.89$) trigger a LangGraph execution interrupt, routing the ticket to a human support agent on a Gradio review desk.
* **Audience:**
  * **End Customers:** Fast, polite resolutions in their native language/tone (English, Hinglish, Hindi) via an interactive chat widget.
  * **Support Agents:** Transparent review queue showing model confidence scores, retrieved policy citations, and editable draft responses.
  * **Engineering Leads / Recruiters:** Clean architecture, instant headless CI (`pytest eval/` runs in <30s without GPU/weights), OpenTelemetry tracing to Jaeger, and reproducible calibration benchmarks.

---

## 2. Seven-Layer Architectural Seams

```
┌─ 7. CONTROL / OPS ─────────────────────────────────────────┐
│ OpenTelemetry spans ➔ console & Jaeger · audit log of every │
│ gated action · pytest eval harness (accuracy, ECE, latency, │
│ flip rate, cost/1k) · GitHub Actions CI on push            │
└─────────────────────────────────────────────────────────────┘
┌─ 6. INFERENCE ─────────────────────────────────────────────┐
│ DecisionEngine adapter:                                    │
│   ├── MockBackend (deterministic, zero-weight CI default)   │
│   ├── KevBackend (Jared Palmer's Kev-0.8B local / Ollama)   │
│   ├── JevBackend (pluggable remote /v1/systemone contract)  │
│   └── GroqBaselineBackend (LiteLLM structured output)      │
│ Text Generation: Groq (llama-3.3-70b-versatile via LiteLLM) │
└─────────────────────────────────────────────────────────────┘
┌─ 5. TOOLS / ENVIRONMENT ───────────────────────────────────┐
│ Qdrant local vector store (policy KB in data/policies/*.md)│
│ Sandboxed operational tools: refund, cancel, escalate      │
└─────────────────────────────────────────────────────────────┘
┌─ 4. MEMORY / STATE ────────────────────────────────────────┐
│ PostgreSQL (Local Docker default / Supabase 1-click env)   │
│ Tables: tickets, decision_trajectories, audit_logs         │
│ Checkpointer: AsyncPostgresSaver                           │
└─────────────────────────────────────────────────────────────┘
┌─ 3. COGNITION ─────────────────────────────────────────────┐
│ Typed decisions (Choice, Score, Noul)                      │
│ Threshold rules in config/thresholds.yaml                  │
│ Code decides: auto_execute / human_review / clarify / deny │
│ Noul verification: draft response checked against policy   │
└─────────────────────────────────────────────────────────────┘
┌─ 2. RUNTIME / ORCHESTRATION ───────────────────────────────┐
│ LangGraph state machine:                                   │
│ intake ➔ route ➔ decide ➔ gate ➔ draft ➔ verify ➔ send/esc  │
│ HITL interrupts on review or P0 actions                    │
└─────────────────────────────────────────────────────────────┘
┌─ 1. INTERFACE / CHANNEL ───────────────────────────────────┐
│ FastAPI intake endpoints (/api/v1/tickets, /api/v1/resume) │
│ Review queue and state management endpoints                │
│ Gradio Dual-Tab Portal mounted at /ui                      │
│   ├── Tab 1: Customer Chatbot                              │
│   └── Tab 2: Agent Review Desk                             │
│ Universal Multilingual & Hinglish language/tone mirroring  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Single-Owner Invariants (Refactor-Clean Rules)

1. **Threshold Logic Ownership:** All threshold evaluation lives strictly in `src/cognition/gating.py`. No other layer may hardcode float comparisons or decide `auto` vs `review`.
2. **State Serialization Invariant:** LangGraph `AgentState` contains **only primitive JSON-serializable types** (`str`, `float`, `int`, `bool`, `list`, `dict`). Tracers, clients, or connection pools must never be stored in the state graph.
3. **Database Portability:** All database operations access PostgreSQL through a standard `DATABASE_URL`. Zero proprietary Supabase SDK code inside domain logic.
4. **Offline / CI Immunity:** The application must boot, run tests, and execute complete LangGraph traversals with `MOCK_DECISION_ENGINE=1` and `MOCK_LLM=1` without external network access or GPU.
5. **Memory Cap (Mac M1 8GB):** Total Docker footprint must not exceed 2.5 GB RAM. Qdrant embedded/docker memory limit = 256MB, Postgres = 128MB, Jaeger = 128MB.

---

## 4. Scrollback Audit (Captured Decisions & Rejected Alternatives)

* **Decision Engine Mocking for CI:** Rejected heavy Kev model download in GitHub Actions runners. Adopted deterministic mock mode with calibrated probability output for <30s CI passes.
* **Domain Selection:** SaaS Billing, Subscriptions, and Account Access chosen over E-Commerce to cleanly map onto `refund`, `cancel`, and `escalate` tools with clear 14-day SLA policies.
* **Contamination Guard:** Public datasets (Banking77, AG News) strictly forbidden in evals because Kev-0.8B may have seen them during pre-training. A new 100-ticket synthetic set will be generated in Slice 06.
* **Language Support:** Rejected rigid Latin vs Devanagari translation bottleneck. Adopted Universal Multilingual pass-through with prompt-level tone/language mirroring (customer writes Hinglish -> agent replies in Hinglish).
* **UI Architecture:** Selected Gradio Dual-Tab mounted directly at `/ui` on FastAPI to avoid node/npm frontend dependencies while providing distinct Customer and Agent views.
