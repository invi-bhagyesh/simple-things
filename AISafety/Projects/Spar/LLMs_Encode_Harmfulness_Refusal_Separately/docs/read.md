# Adaptive Safety Calibration via Representation Engineering

## Overview

Large Language Models (LLMs) frequently exhibit **over-refusal** , rejecting benign user requests that merely _sound_ harmful, while still remaining vulnerable to sophisticated jailbreaks. Recent work has shown that LLMs encode **harmfulness** and **refusal behavior** as **distinct internal representations** , localized at different token positions in the forward pass.

This project builds on the findings of _LLMs Encode Harmfulness and Refusal Separately_ (Zhao et al., NeurIPS 2025) and moves beyond static analysis to implement a **runtime representation-engineering system** called **Adaptive Safety Calibration** .

Instead of globally suppressing refusal mechanisms, we introduce a **context-aware control policy** that:

- reads the model’s internal belief about harmfulness,
- conditionally modulates refusal behavior,
- reduces over-refusal **without disabling safety** for genuinely dangerous prompts.

This project is **training-free** , data-efficient, and designed to be extensible into parameter-efficient adapters.

---

## Key Idea

LLMs internally:

- encode **harmfulness** at the last token of the user instruction (`t_inst`),
- encode **refusal behavior** at the final post-instruction token (`t_post-inst`).

However, refusal is not always causally downstream of harmfulness, leading to:

- over-refusal on benign prompts,
- brittle safety under jailbreak attacks.

We exploit this separation to build a **piece-wise representation control system** that enforces the correct causal ordering:

> _Refusal should activate **only if** internal harmfulness is high._

---

## Methodology

### 1. Dataset Categorization (Inference)

Using the provided inference scripts, prompts are categorized into four behaviorally distinct regimes:

| Category          |     | Description                                    |     |
| ----------------- | --- | ---------------------------------------------- | --- |
| Accepted Harmless | ✅  | Normal helpful responses (e.g. Alpaca)         |     |
| Refused Harmless  | ✅  | Over-refusal cases (e.g. XSTest)               |     |
| Refused Harmful   |     | Correct safety behavior (e.g. AdvBench, CATQA) |     |
| Accepted Harmful  |     | Successful jailbreaks (e.g. GCG, Persuasion)   |     |

This stratification explicitly isolates **over-refusal** as a target failure mode.

---

### 2. Representation Reading (LAT)

Rather than using a simple mean-difference direction, we apply **Linear Artificial Tomography (LAT)** to extract a **robust Harmfulness Vector** :

- Hidden states are extracted from a **middle transformer layer** (typically 13–14).
- Activations are taken at the **instruction boundary token** (`t_inst`).
- LAT yields a **scalar harmfulness score** per example, robust to adversarial suffixes.

This converts harmfulness from a qualitative concept into a **measurable latent variable**.

---

### 3. Representation Control (Piece-wise Operator)

At inference time, we apply a **conditional intervention policy** :

1. Measure harmfulness alignment at `t_inst`.
2. If harmfulness is **above a threshold** :
   - Allow refusal behavior to proceed normally.
3. If harmfulness is **below the threshold** :
   - Apply a _negative shift_ along the **Refusal Vector** at `t_post-inst`,
   - Forcing compliance and correcting over-refusal.

Formally, the intervention is:

[
\Delta a =
\begin{cases}
0 & \text{if } \langle a_{t_{inst}}, v_{harm} \rangle > \tau
-\alpha \cdot v_{refuse} & \text{otherwise}
\end{cases}
]

This ensures refusal behavior is **gated by internal belief** , not surface-level prompt features.

---

### 4. Evaluation

We evaluate performance using a **Trade-off Score** :

[
\text{Trade-off Score} = \frac{1}{2}(\text{Compliance Rate on XSTest} + \text{Safety Score on AdvBench})
]

A successful system:

- significantly increases compliance on over-refusal benchmarks,
- while maintaining or improving safety on harmful benchmarks.

---

## Why This Matters

### Conceptual Contribution

- Moves from _representation analysis_ to **representation engineering** .
- Enforces a **causal relationship** between internal belief and external behavior.
- Demonstrates that refusal can be calibrated without retraining.

### Practical Advantages

- **Training-free** : no fine-tuning required.
- **Data-efficient** : high-quality vectors from ~64 instruction pairs.
- **Fast iteration** : inference-time hooks allow rapid experimentation.

### Robustness Insight

We find that many jailbreaks suppress refusal behavior **without changing the model’s internal harmfulness belief** . This project explicitly exploits that mismatch.

---

## Extensions

This framework naturally supports further research directions:

- **LoRRA integration** : compress the piece-wise operator into a low-rank adapter (~168 MB), removing inference-time hooks.
- **Category-specific harmfulness** : learn multiple harmfulness subspaces for fine-grained risk control.
- **Trajectory-level calibration** : extend harmfulness tracking across token generation.
- **Training diagnostics** : track harmfulness representations across SFT or RLHF checkpoints.

---

## Repository Structure (Planned)

```
adaptive-safety-calibration/
├── data/
│   ├── alpaca/
│   ├── xstest/
│   ├── advbench/
│   └── jailbreaks/
├── extraction/
│   ├── extract_hidden_states.py
│   └── lat_probe.py
├── steering/
│   ├── piecewise_operator.py
│   └── hooks.py
├── evaluation/
│   ├── metrics.py
│   └── run_eval.py
├── scripts/
│   ├── run_inference.sh
│   └── run_calibration.sh
└── README.md
```

---

## Summary

**Adaptive Safety Calibration** demonstrates that safety failures in LLMs are not inevitable trade-offs but often arise from misaligned internal control logic. By reading and respecting the model’s own latent belief about harmfulness, we can reduce over-refusal while preserving robust safety guarantees.

This project serves as a minimal, principled example of **representation engineering for AI safety** .
