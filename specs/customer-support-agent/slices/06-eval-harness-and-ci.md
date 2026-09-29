# Slice 06: Synthetic Eval Harness, Benchmarks & GitHub Actions CI

## 1. Contract Unlocked
Delivers the automated evaluation harness, statistical calibration metrics (ECE), latency/cost benchmarking, and the GitHub Actions CI pipeline ensuring a permanent green passing badge on the repository.

## 2. API Seam & Module Ownership
* **Module:** `eval/` and `.github/workflows/`
* **Synthetic Evaluation Dataset (`eval/data/saas_tickets_eval.json`):**
  * 100 curated synthetic SaaS support tickets:
    * 50 English, 35 Hinglish, 15 Hindi (Devanagari).
    * Categories: `refund`, `cancel_subscription`, `billing_dispute`, `account_escalation`, `general_inquiry`.
    * Ground truth: `expected_action`, `ground_truth_policy_id`, `should_auto_execute: bool`.
* **Eval Metric Runners (`eval/`):**
  * `eval/metrics/calibration.py`: Computes Expected Calibration Error (ECE) across binned confidence scores.
  * `eval/metrics/invariance.py`: Tests option-order flip rate by permuting candidate action list $[A, B, C]$ vs $[C, B, A]$.
  * `eval/metrics/threshold_sweep.py`: Computes precision vs human review rate across confidence thresholds from $0.50$ to $0.99$.
  * `eval/test_eval_suite.py`: Pytest suite asserting:
    * Classification Accuracy $\ge 90\%$.
    * Option-order flip rate $\le 5\%$.
    * Zero false-positive auto-refunds on flagged dispute/fraud tickets.
* **Observability Setup (`src/telemetry/tracer.py`):**
  * Configures OpenTelemetry SDK with OTLP exporter to Jaeger (`OTEL_EXPORTER_OTLP_ENDPOINT`).
  * Injects custom attributes into every LangGraph node span: `decision.latency_ms`, `decision.confidence`, `gating.outcome`.
* **GitHub Actions Pipeline (`.github/workflows/ci.yml`):**
  * Runs on `push` and `pull_request` to `main`.
  * Installs `uv`, syncs virtual environment, runs `ruff check`, and executes `pytest tests/ eval/`.
  * Runs in < 45 seconds with zero external network or GPU requirements.

## 3. What the Human Can Run or See
* Run the eval suite locally:
  `pytest eval/ -v --tb=short`
* Generate the evaluation markdown report:
  `python -m eval.generate_report`
  Produces `eval/EVAL_REPORT.md` containing formatted markdown tables for README integration:
  * Accuracy & ECE breakdown
  * Flip-rate invariance table
  * P50 / P95 decision latency comparison
  * Estimated cost per 1,000 tickets

## 4. Verification Gates & Tests
* `pytest tests/` passes 100%.
* `pytest eval/` passes 100%.
* GitHub Actions workflow passes locally via `act` or dry-run validation.

## 5. Delegated Implementer Discretion
* Exact bin count for ECE calculation (default: 10 bins).
* Visual table formatting inside `EVAL_REPORT.md`.

## 6. What Must Stay Green
* CI pipeline must never depend on live external APIs (Groq, Supabase, or Ollama). All CI runs default to deterministic mock mode.

## 7. Non-blocking Checkpoint
Review eval report columns and metrics. Proceed with default accuracy, ECE, latency, and flip-rate metrics.
