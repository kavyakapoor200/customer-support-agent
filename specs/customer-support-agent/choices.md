# Choices Ledger — Customer Support Agent

This document audits the architectural choices and implementation shortcuts made across Slices 01 through 06. Every entry is written in the ELI5 register: a walked scenario, terms of art defined in place, and a clear verdict distinguishing between test-optimized mocks and production intelligence.

---

## 1. Needs-User Review (Ranked by Confidence Ascending)

### Choice 01: Pure Template Drafting vs Live Groq LLM Response Synthesis

- **When:** Slice 04 / Slice 05 (`src/workflow/nodes.py:draft_node`)
- **The Choice:** When drafting the customer-facing message, the workflow completely ignores the live LLM (Groq) and instead selects a static, hardcoded f-string template (e.g., hardcoded `"Hi Rahul, humne aapka refund process kar diya..."`).
  - _Concrete Walked Scenario:_ Customer "Priya" writes: _"Hello, I accidentally bought two annual seats for team Alpha on invoice #9401, please cancel the duplicate seat."_ Groq correctly classifies the action as `refund` with 0.95 confidence. But when generating the reply, the system does not send Priya's context or invoice number to Groq. Instead, it hits a Python `if/elif` block and outputs: `"Hi Rahul, humne aapka refund request process kar diya hai."` The user sees a hardcoded response calling them "Rahul" and immediately realizes the LLM never actually wrote the reply.
  - _Unbuilt Alternative:_ If `GROQ_API_KEY` is present, `draft_node` calls `litellm.acompletion` with the customer's actual query, the retrieved policy chunk from Qdrant, and tone guidelines to dynamically synthesize a customized, natural response (falling back to templates only when offline in CI).
- **The Gap:** The spec mentioned _"Layer 6: Text Generation: Groq (llama-3.3-70b-versatile via LiteLLM)"_, but did not specify an explicit offline-fallback switch for the text generation node during fast headless test runs.
- **The Reach:** Dictates whether the agent feels like a real, intelligent AI assistant or a rigid chatbot script.
- **Verdict:** **UNSOUND** (Confidence: **LOW** — The implementer optimized for 2-second CI execution, creating the hollow mock feeling the user noticed).
- **Corrected Decision:** Wire `draft_node` to check if `settings.GROQ_API_KEY` is valid; if so, stream/generate a personalized response using Groq `llama-3.3-70b-versatile`; fall back to templates strictly when `GROQ_API_KEY` is empty or backend is `mock`.

---

### Choice 02: MD5 Hashed Term-Frequency Vectors vs Real Embedding Model

- **When:** Slice 03 (`src/kb/store.py:_deterministic_embedding`)
- **The Choice:** Instead of downloading an embedding model (like `all-MiniLM-L6-v2` or using OpenAI/Groq embeddings), the vector database hashes words using MD5 into 128 buckets.
  - _Concrete Walked Scenario:_ In `PolicyStore`, policy markdown files are indexed into Qdrant. To compute vector coordinates without downloading a 500MB PyTorch dependency on Mac M1 or hitting rate limits, `_deterministic_embedding` hashes each word string modulo 128. While this matches exact keywords, it has zero conceptual semantic understanding (e.g., "money back" does not match "refund" unless the exact stem is present).
  - _Unbuilt Alternative:_ Use a lightweight sentence-transformer or OpenAI `text-embedding-3-small` for true semantic vector similarity.
- **The Gap:** The hardware constraint (Mac M1 8GB RAM, <2.5GB Docker footprint) and the requirement for fast CI (<30s) pushed the implementer away from local Torch neural models.
- **The Reach:** Limits the vector database to bag-of-words keyword retrieval rather than genuine semantic deep retrieval.
- **Verdict:** **NEEDS-USER** (Confidence: **MEDIUM**).
- **Provisional Call:** Keep MD5 deterministic vectors as the default for tests/CI, but provide a pluggable embedding provider (e.g. FastEmbed or remote embeddings) when live semantic search is needed.

---

### Choice 03: Regex Keyword Language Classifier vs LLM Language & Intent Parser

- **When:** Slice 04 (`src/workflow/nodes.py:_detect_language`)
- **The Choice:** Language detection uses character range checking (`\u0900` - `\u097F` for Devanagari) and a hardcoded list of Hinglish marker words (`["mera", "bhai", "karo", "nahi"]`) instead of asking the model.
  - _Concrete Walked Scenario:_ A customer writes an ambiguous sentence or mixed English-Spanish inquiry. If it lacks the hardcoded Hindi/Hinglish vocabulary words, it defaults to English.
  - _Unbuilt Alternative:_ Pass language detection into the System 1 / Groq prompt so the model returns detected dialect and tone alongside intent.
- **The Gap:** Slice 04 requested "universal multilingual & Hinglish pass-through" without specifying whether detection happens in Python or inside the LLM prompt.
- **The Reach:** Constrains multi-lingual adaptation to predefined keyword dictionaries.
- **Verdict:** **SOUND for latency, UNSOUND for edge cases** (Confidence: **MEDIUM**).
- **Corrected Decision:** Retain regex as a fast zero-latency pre-filter, but allow the Groq prompt to return `"language"` in its JSON payload.

---

## 2. Unsound Choices (Redo Decisions)

### Choice 04: Dummy Float Comparison in Policy Verification Node (`verify_node`)

- **When:** Slice 04 (`src/workflow/nodes.py:verify_node`)
- **The Choice:** The policy verification node only checks `if amount > 500.0`. It does not inspect the draft text or compare it against the retrieved policy rules.
  - _Concrete Walked Scenario:_ The customer requests a refund after 45 days. The retrieved policy clearly states the SLA is 14 days. But because the amount is $50 (< $500), `verify_node` returns `verification_passed: True`! It never checks whether the draft promises something against company policy.
  - _Unbuilt Alternative:_ The Noul verification step runs a fast prompt checking: _"Does this draft promise a refund outside the 14-day SLA found in policy chunk X?"_
- **The Gap:** The spec defined "Noul verification" conceptually, but did not define the prompt schema.
- **The Reach:** Allows the agent to issue responses that contradict the Qdrant retrieved policy text if the dollar amount is small.
- **Verdict:** **UNSOUND** (Confidence: **LOW**).
- **Corrected Decision:** If a live LLM is configured, `verify_node` checks the draft against the retrieved policy constraints using an LLM verification prompt; fall back to deterministic amount bounds when in mock mode.

---

## 3. Sound Choices (Architecture the User Owns)

### Choice 05: Mock Payment & Cancellation Tools with Audit Hashing

- **When:** Slice 03 (`src/tools/mock_tools.py`)
- **The Choice:** Operational tools (`execute_refund`, `cancel_subscription`) generate realistic transaction hashes (`tx_...`) and audit logs in-memory rather than connecting to live Stripe/Chargebee billing APIs.
  - _Concrete Walked Scenario:_ An approved refund outputs `{"success": true, "transaction_id": "tx_519983620149", ...}`. No real bank accounts are debited.
  - _Why Sound:_ Enterprise customer support prototypes should never execute real financial debits without a sandbox Stripe secret key. The MCP tool interfaces are standard and can be swapped for real Stripe SDK calls with zero changes to the LangGraph workflow.
- **Verdict:** **SOUND** (Confidence: **HIGH**).

### Choice 06: MemorySaver Fallback for LangGraph Interrupts

- **When:** Slice 04 / Slice 05 (`src/api/service.py`)
- **The Choice:** Uses `MemorySaver` in-memory checkpointer when PostgreSQL is offline, switching to `AsyncPostgresSaver` when `DATABASE_URL` is active.
- **Why Sound:** Allows developers on laptops without running Docker daemons to test the full Human-in-the-Loop interrupt and resume cycle without database crashes.
- **Verdict:** **SOUND** (Confidence: **HIGH**).
