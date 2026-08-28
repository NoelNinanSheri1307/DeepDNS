# DeepDNS Multi-View Fusion Architecture

## 1. Executive Summary & Objective
The **DeepDNS Multi-View Network** fuses two independent representations of DNS network traffic:
1. **Behavioral View ($z_{beh} \in \mathbb{R}^{64}$)**: Extracted by the Temporal GRU over 12 causal statistical channels, capturing inter-arrival dynamics and traffic volume persistence across an observation horizon $K \in [5, 30]$.
2. **Lexical View ($z_{lex} \in \mathbb{R}^{128}$)**: Extracted by the Character-CNN over raw domain character sequences, capturing orthographic and entropy signatures.

By combining both views through late fusion, DeepDNS prevents adversarial timing evasion (e.g. adding random delays to break cadence) while maintaining low false alarm rates.

---

## 2. Multi-View Architecture Specification

```
Behavioral Sequence (B, K, 12)                 Lexical Tokens Sequence (B, K, 128)
             │                                                 │
             ▼                                                 ▼
   Temporal GRU Layer (12 -> 64)                     Character-CNN Encoder (128)
             │                                                 │
             ▼                                                 ▼
  Behavioral State z_beh in R^64                    Lexical Embedding z_lex in R^128
             │                                                 │
             ▼                                                 ▼
  Linear(64, 64) + LayerNorm + ReLU                 Linear(128, 64) + LayerNorm + ReLU
             │                                                 │
             └───────────────────────┬─────────────────────────┘
                                     │
                                     ▼
                      Concatenated Views in R^128
                                     │
                                     ▼
                      Linear(128, 64) + LayerNorm + ReLU + Dropout(0.1)
                                     │
                                     ▼
                      Fused Representation z_fuse in R^64
                                     │
                                     ▼
                      Linear(64, 2) ──► Softmax ──► Class Probabilities [P(0), P(1)]
```

---

## 3. Parameter Counts & Module Breakdown

| Module | Component Layers | Parameter Count | Output Tensor Shape |
| :--- | :--- | :--- | :--- |
| **Temporal GRU** | `nn.GRU(12, 64)` + Linear Head | 17,218 params | $z_{beh} \in \mathbb{R}^{B \times K \times 64}$ |
| **Character-CNN** | `Embedding(45, 32)` + 3x Conv1D + Projection | 29,826 params | $z_{lex} \in \mathbb{R}^{B \times K \times 128}$ |
| **MultiView Fusion Head** | Behavioral Proj (4,288) + Lexical Proj (8,384) + Fusion MLP (8,384) + Classifier (130) | 21,186 params | $z_{fuse} \in \mathbb{R}^{B \times K \times 64}$, Logits $\in \mathbb{R}^{B \times K \times 2}$ |
| **Total Unified Model** | — | **68,230 Parameters** ($\approx 272.9\text{ KB}$) | Fits entirely within RTX 2050 L2 Cache/VRAM |

---

## 4. Causality & Information Isolation Guarantees

1. **Strict Horizon Isolation**:
   At horizon $K$, the fused prediction $p_K$ is computed exclusively from $[x_1 \dots x_K]$ and $[w_1 \dots w_K]$.
2. **Empirical Causality Proof**:
   Unit test `test_multiview_causality_no_future_leakage` injects extreme random noise into behavioral features and character tokens for future steps $K+1 \dots 30$. Step predictions $p_1 \dots p_K$ and representations $z_{fuse}^{(1\dots K)}$ remain strictly unchanged ($< 10^{-6}$ error).
3. **Forbidden Feature Elimination**:
   Neither view accepts labels, split metadata, capture IDs, or post-event statistics.

---

## 5. Ablation Support

The architecture provides native routing switches via the `mode` parameter:
- `mode="both"`: Full Dual-View Late Fusion ($z_{beh} + z_{lex}$).
- `mode="behavioral_only"`: Routes exclusively through the Temporal GRU branch.
- `mode="lexical_only"`: Routes exclusively through the Character-CNN branch.
