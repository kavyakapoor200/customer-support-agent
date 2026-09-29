# Customer Support Agent — Evaluation & Calibration Report

> **Dataset:** 100 Synthetic SaaS Support Tickets (50 English, 35 Hinglish, 15 Hindi)  
> **Held-Out Calibration Set:** 30 Synthetic Tickets (Disjoint split used strictly for temperature scaling)  
> **Primary Evaluation Model:** `jaredpalmer/kev-0.8b` (Local MLX on Apple Silicon GPU)  
> **Evaluation Date:** 2026-09-29  
> **Strict Mode:** `fallback_to_mock = False` (Zero silent mock fallbacks; 100% genuine neural forward passes)  

---

## 1. Executive Benchmark Summary

*Note: In Table 1, Raw ECE is computed on uncalibrated raw softmax confidences ($T=1.0$), while Calibrated ECE is computed on runtime temperature-scaled confidences ($T=0.65$).*

| Evaluation Metric | Real Kev-0.8B (Local Neural Engine) | Mock Engine (CI/CD Simulator) | Production Target | Status |
|---|:---:|:---:|:---:|:---:|
| **Classification Accuracy** | **91.0%** | 96.0% | $\ge 90.0\%$ | ✅ PASS |
| **Raw Expected Calibration Error (ECE)** | **0.1929** | 0.0504 | $\le 0.1500$ | ⚠️ EXCEEDS TARGET (Raw Softmax) |
| **Calibrated ECE (Temperature Scaled)** | **0.0739** ($T=0.65$) | N/A (Linear Heuristic) | $\le 0.1500$ | ✅ PASS (Calibrated) |
| **Option-Order Flip Rate** | **2.0%** | 4.0% | $\le 5.0\%$ | ✅ PASS |
| **P50 Decision Latency** | **162.94 ms** | 0.01 ms | $< 50.0\text{ ms}$ | ⚠️ EXCEEDS TARGET (Local On-Device GPU) |
| **P95 Decision Latency** | **203.91 ms** | 0.03 ms | $< 150.0\text{ ms}$ | ⚠️ EXCEEDS TARGET (Local On-Device GPU) |
| **Cost per 1,000 Tickets** | **$0.00** | $0.00 | $< \$1.00$ | 💰 ZERO COST |

---

## 2. Target Analysis & Engineering Rationale

The benchmark table restores the original strict production targets and transparently surfaces the misses and post-hoc engineering solutions:

1. **Latency Targets ($< 50.0\text{ ms}$ P50, $< 150.0\text{ ms}$ P95):**
   - **Original Target Context:** Formulated around in-memory keyword matching heuristics or embedding vector lookups, which execute in sub-millisecond time.
   - **On-Device Neural Reality:** Running full neural forward passes of an 800M parameter model (`jaredpalmer/kev-0.8b`) locally via Apple Silicon MLX GPU on unbatched single inputs takes **162.94 ms P50** and **203.91 ms P95**.
   - **Justification & Trade-off:** While exceeding the 50 ms target, 162.94 ms is **10× to 15× faster** than cloud LLM APIs (~1,200–2,500 ms), completely eliminates cloud API token fees, keeps sensitive customer data on-premise, and has zero network dependency. Sub-50 ms neural inference would require 4-bit INT4 quantization or continuous batching.

2. **Calibration Target ($\le 0.1500$ ECE):**
   - **Original Target Context:** Required for reliable confidence-based gating thresholds ($\tau$).
   - **Raw Softmax Limitation (Underconfidence):** The raw model is significantly **underconfident** on both splits:
     - **Calibration Split (N=30):** Mean raw confidence of **77.5%** vs **83.3%** empirical accuracy.
     - **Evaluation Split (N=100):** Mean raw confidence of **73.4%** vs **91.0%** empirical accuracy.
     - This gap yields a raw ECE of **0.1929** (exceeding the 0.1500 target).
   - **Temperature Scaling Solution ($T < 1.0$):** Because the raw model is underconfident, post-hoc **Temperature Scaling** with $T < 1.0$ ($T = 0.65$, fitted on the held-out calibration split minimizing NLL) sharpens the probability distribution. This lifts average winning confidence into alignment with empirical accuracy, reducing evaluation ECE to **0.0739** (passing the $\le 0.1500$ target) with **zero test label leakage** and zero change to predicted action classifications.
   - **Sample Size Limitation:** The temperature parameter ($T = 0.65$) was fitted on a small sample of 30 synthetic calibration tickets. While 30 samples is standard for a 1D scalar parameter without overfitting, production systems should fit calibration across $\ge 200$ tickets to capture nuanced dialectal variances.

---

## 3. Real vs Mock Inference: Architectural Clarification

1. **Jared Palmer's Kev-0.8B (Production Engine):**
   - **Architecture:** 800M parameter specialized System 1 model running on-device via Apple Silicon MLX GPU (`/v1/systemone`).
   - **Characteristics:** Genuine neural token probability distributions, true semantic understanding of Hindi/Hinglish/English slang, **162.94 ms P50 latency**, and **91.0% accuracy**.
   - **Strict Execution:** Configured with `fallback_to_mock=False` by default. If the local model server is down, it raises an explicit `RuntimeError` rather than silently masking model degradation with keyword heuristics.
   - **Runtime Calibration:** Integrated directly into `KevDecisionEngine.decide()` using configured $T = 0.65$.

2. **Mock Decision Engine (CI/CD Test Simulator):**
   - **Architecture:** In-memory keyword pattern matcher in Python.
   - **Characteristics:** Used exclusively for lightning-fast sub-millisecond unit testing in GitHub Actions where Apple Silicon GPUs are unavailable (~0.01 ms latency).

---

## 4. Confidence Threshold Sweep with 95% Confidence Intervals (Wilson Score)

*Note: The threshold sweep below is evaluated directly through the runtime `KevDecisionEngine(temperature=0.65)` path using calibrated confidences.*

| Confidence Threshold ($\tau$) | Auto Count | Auto Rate | Review Count | Review Rate | Auto Precision | 95% CI (Wilson) | Disputed Auto-Refunds |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $\ge 0.50$ | 79 | 79.0% | 21 | 21.0% | **92.4%** | [84.4%, 96.5%] | 1 unauthorized |
| $\ge 0.55$ | 77 | 77.0% | 23 | 23.0% | **93.5%** | [85.7%, 97.2%] | 1 unauthorized |
| $\ge 0.60$ | 74 | 74.0% | 26 | 26.0% | **93.2%** | [85.1%, 97.1%] | 1 unauthorized |
| $\ge 0.65$ | 73 | 73.0% | 27 | 27.0% | **93.2%** | [84.9%, 97.0%] | 1 unauthorized |
| $\ge 0.70$ | 70 | 70.0% | 30 | 30.0% | **94.3%** | [86.2%, 97.8%] | 0 unauthorized |
| $\ge 0.75$ | 63 | 63.0% | 37 | 37.0% | **93.7%** | [84.8%, 97.5%] | 0 unauthorized |
| $\ge 0.80$ | 57 | 57.0% | 43 | 43.0% | **94.7%** | [85.6%, 98.2%] | 0 unauthorized |
| $\ge 0.85$ | 42 | 42.0% | 58 | 58.0% | **100.0%** | [91.6%, 100.0%] | 0 unauthorized |
| $\ge 0.90$ | 32 | 32.0% | 68 | 68.0% | **100.0%** | [89.3%, 100.0%] | 0 unauthorized |
| $\ge 0.95$ | 22 | 22.0% | 78 | 78.0% | **100.0%** | [85.1%, 100.0%] | 0 unauthorized |

> **Sample Safety Observation (N=100 evaluation tickets):** At $\tau \ge 0.85$, 0 false-positive auto-refunds were observed on disputed charges within this 100-ticket evaluation sample (42 tickets auto-actioned with 100.0% precision, 95% Wilson CI: [91.6%, 100.0%]).

---

## 5. Comparison: System 1 Gated Engine vs Standard LLM Baseline

| Dimension | Our System (Kev-0.8B Gated) | Standard LLM Baseline (GPT-4 / Cloud API - Estimated)* |
|---|---|---|
| **Execution Architecture** | Deterministic Code Gate (`config/thresholds.yaml`) | Autonomous Prompt-driven Function Calling |
| **P50 Decision Latency** | **162.94 ms** (Local Apple Silicon GPU) | ~1,200 - 2,500 ms (Cloud API)* |
| **Safety Guarantees** | 0 observed at $\tau \ge 0.70$ in 100-ticket synthetic sample (1 observed at $\tau \le 0.65$) | Susceptible to jailbreaks & prompt injection |
| **Human Supervision** | Native LangGraph Interrupt & Checkpoint Resume | Custom manual routing loops |
| **Cost per 1k Tickets** | **$0.00** (Local On-Device Execution) | $2.50 - $15.00* |
| **Multi-Dialect Handling** | English, Hinglish, Hindi Devanagari | English-skewed prompt comprehension |

\* *Note: Standard LLM Baseline figures (~1,200 - 2,500 ms round-trip latency, $2.50 - $15.00/1k tickets) are industry reference estimates for cloud GPT-4 / Claude class models rather than directly measured local runs.*
