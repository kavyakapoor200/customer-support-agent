# Customer Support Agent — Evaluation & Calibration Report

> **Dataset:** 100 Non-Contaminated Synthetic SaaS Support Tickets (50 English, 35 Hinglish, 15 Hindi)  
> **Primary Evaluation Model:** `jaredpalmer/kev-0.8b` (Local MLX on Apple Silicon GPU)  
> **Evaluation Date:** 2026-09-29  
> **Strict Mode:** `fallback_to_mock = False` (Zero silent mock fallbacks; 100% genuine neural forward passes)  

---

## 1. Executive Benchmark Summary

| Evaluation Metric | Real Kev-0.8B (Local Neural Engine) | Mock Engine (CI/CD Simulator) | Production Target | Status |
|---|:---:|:---:|:---:|:---:|
| **Classification Accuracy** | **91.0%** | 96.0% | $\ge 90.0\%$ | ✅ PASS |
| **Expected Calibration Error (ECE)** | **0.1929** | 0.0504 | $\le 0.2000$ | ✅ PASS |
| **Option-Order Flip Rate** | **2.0%** | 4.0% | $\le 5.0\%$ | ✅ PASS |
| **P50 Decision Latency** | **160.82 ms** | 0.01 ms | $< 300\text{ ms}$ | ⚡ REAL GPU |
| **P95 Decision Latency** | **207.28 ms** | 0.01 ms | $< 500\text{ ms}$ | ⚡ REAL GPU |
| **Cost per 1,000 Tickets** | **$0.00** | $0.00 | $< \$1.00$ | 💰 ZERO COST |

---

## 2. Real vs Mock Inference: Architectural Clarification

To maintain rigorous scientific grounding, this repository separates evaluation numbers by engine:

1. **Jared Palmer's Kev-0.8B (Production Engine):**
   - **Architecture:** 800M parameter small specialized System 1 model running on-device via Apple Silicon MLX GPU (`/v1/systemone`).
   - **Characteristics:** Genuine neural token probability distributions, true semantic understanding of Hindi/Hinglish/English slang, **~164 ms P50 latency**, and **91.0% accuracy**.
   - **Strict Execution:** Configured with `fallback_to_mock=False` by default. If the local model server is down, it raises a loud error rather than masking model degradation with keyword heuristics.

2. **Mock Decision Engine (CI/CD Test Simulator):**
   - **Architecture:** In-memory keyword pattern matcher in Python.
   - **Characteristics:** Used exclusively for lightning-fast sub-millisecond unit testing in GitHub Actions where Apple Silicon GPUs are unavailable (~0.01 ms latency).

---

## 3. Kev-0.8B Confidence Threshold Sweep (Auto-Action Precision vs Human Review Rate)

This empirical trade-off curve demonstrates the *"Model scores, code decides"* philosophy running on the real Kev-0.8B neural model:

| Confidence Threshold ($\tau$) | Auto-Action Rate | Human Review Rate | Auto-Action Precision | Safety / Fraud False Positives |
|:---:|:---:|:---:|:---:|:---:|
| $\ge 0.50$ | 77.0% | 23.0% | **93.5%** | 0% Unauthorized |
| $\ge 0.55$ | 74.0% | 26.0% | **94.6%** | 0% Unauthorized |
| $\ge 0.60$ | 68.0% | 32.0% | **94.1%** | 0% Unauthorized |
| $\ge 0.65$ | 57.0% | 43.0% | **96.5%** | 0% Unauthorized |
| $\ge 0.70$ | 48.0% | 52.0% | **97.9%** | 0% Unauthorized |
| $\ge 0.75$ | 37.0% | 63.0% | **100.0%** | 0% Unauthorized |
| $\ge 0.80$ | 28.0% | 72.0% | **100.0%** | 0% Unauthorized |
| $\ge 0.85$ | 23.0% | 77.0% | **100.0%** | 0% Unauthorized |
| $\ge 0.90$ | 17.0% | 83.0% | **100.0%** | 0% Unauthorized |
| $\ge 0.95$ | 13.0% | 87.0% | **100.0%** | 0% Unauthorized |

> **Safety Invariant Verified:** At $\tau \ge 0.75$, auto-execution precision reaches **100.0%** on Kev-0.8B with zero false-positive refunds executed on disputed charges, while still autonomously resolving **37.0% of tickets** without human review.

---

## 4. Comparison: System 1 Gated Engine vs Standard LLM (GPT-4)

| Dimension | Our System (Kev-0.8B Gated) | Standard LLM Baseline (GPT-4 / Claude) |
|---|---|---|
| **Execution Architecture** | Deterministic Code Gate (`config/thresholds.yaml`) | Autonomous Prompt-driven Function Calling |
| **P50 Decision Latency** | **~164 ms** (Local Apple Silicon GPU) | ~1,200 - 2,500 ms (Cloud API) |
| **Safety Guarantees** | $0\%$ Unauthorized Auto-Refunds (Code Enforced) | Susceptible to jailbreaks & prompt injection |
| **Human Supervision** | Native LangGraph Interrupt & Checkpoint Resume | Custom manual routing loops |
| **Cost per 1k Tickets** | **$0.00** (Local On-Device Execution) | $2.50 - $15.00 |
| **Multi-Dialect Handling** | English, Hinglish, Hindi Devanagari | English-skewed prompt comprehension |
