# DeepDNS Temporal GRU Architecture & Sequence Specification

## 1. Executive Summary & Purpose
The **Temporal GRU** serves as the temporal evidence accumulation backbone for DeepDNS. Point-wise baseline classifiers (Rule-Based, Random Forest, MLP) operate on single isolated DNS queries ($K=1$) and suffer from a severe False Positive Rate ceiling ($\sim 16.3\%$–$68.2\%$). The Temporal GRU resolves this by maintaining a recurrent hidden state $h_t$ across expanding sequence windows ($K \in [5, 30]$), allowing persistent exfiltration signals to accumulate over time while filtering out transient benign structural noise.

---

## 2. Sequence Construction & Causal Guarantees

### Stream Definition
For each capture session, observations are ordered chronologically:
$$X = [x_1, x_2, \dots, x_K] \in \mathbb{R}^{B \times K \times 12}$$
where each observation $x_t$ contains the 12 approved causal features:
1. `inter_arrival_time` ($\ln(1 + \max(\Delta t_i, 0))$)
2. `fqdn_length`
3. `subdomain_length`
4. `char_entropy`
5. `digit_ratio`
6. `uppercase_ratio`
7. `special_ratio`
8. `label_count`
9. `max_label_length`
10. `avg_label_length`
11. `has_subdomain`
12. `payload_len`

### Mathematical Causality Guarantee
The network enforces strict forward temporal causality:
$$h_t = \text{GRU}(h_{t-1}, x_t), \quad p_t = \text{Softmax}(W_c h_t + b_c)$$
The representation $h_k$ and prediction $p_k$ at step $k$ depend exclusively on past observations $[x_1 \dots x_k]$ and are mathematically invariant to future inputs $[x_{k+1} \dots x_K]$. This has been empirically verified via unit test `test_causal_no_future_leakage`.

---

## 3. PyTorch Architecture Specification

```
   DNS Observation Stream x_t in R^12 (t = 1 ... K)
                         │
                         ▼
        ┌──────────────────────────────────┐
        │  Causal GRU Layer                │
        │  - input_dim: 12                 │
        │  - hidden_dim: 64                │
        │  - num_layers: 1 (batch_first)   │
        └────────────────┬─────────────────┘
                         │
                         ▼
            Hidden State Sequence h_t in R^64
                         │
                         ▼
        ┌──────────────────────────────────┐
        │  Classification Head             │
        │  - Linear(64, 32)                │
        │  - ReLU()                        │
        │  - Dropout(p=0.1)                │
        │  - Linear(32, 2)                 │
        └────────────────┬─────────────────┘
                         │
                         ▼
        Step Logits (B, K, 2) ──► Softmax ──► Step Probas p_t in [0, 1]
```

### Tensor Dimensions & Input/Output Mapping
| Component | Tensor Shape | Data Type | Description |
| :--- | :--- | :--- | :--- |
| **Input Sequence $X$** | `(Batch_Size, 30, 12)` | `torch.float32` | Padded causal feature matrix ($K_{max}=30$). |
| **Sequence Lengths** | `(Batch_Size,)` | `torch.long` | Valid unpadded horizon length $k \in [5, 30]$. |
| **Hidden States $h_{1\dots K}$** | `(Batch_Size, 30, 64)` | `torch.float32` | Sequential recurrent hidden embeddings. |
| **Step Logits** | `(Batch_Size, 30, 2)` | `torch.float32` | Logits evaluated at every step $t = 1 \dots 30$. |
| **Final Logits** | `(Batch_Size, 2)` | `torch.float32` | Logits evaluated at the valid horizon length $k$. |
| **Streaming Step** | `(Batch_Size, 12)` + $h_{t-1}$ | `torch.float32` | Online single-query update returning $(p_t, h_t)$. |

---

## 4. Sequence Padding & Length Masking Strategy
- Observation windows shorter than $K_{max} = 30$ (e.g. $k=5, 6, \dots, 29$) are right-padded with zeros to shape `(30, 12)`.
- The network indexes final predictions at step $k-1$ using actual sequence lengths (`batch_indices, seq_lens - 1`), ensuring zero-padded trailing observations do not corrupt sequence predictions.

---

## 5. Training & Multi-Horizon Evaluation Interface

### Training Protocol
- **Optimizer**: AdamW ($\text{lr} = 10^{-3}$, $\text{weight\_decay} = 10^{-4}$)
- **Loss**: Binary Cross-Entropy on final horizon logits
- **Batch Size**: 128 (memory footprint $< 300\text{ MB}$ on RTX 2050)
- **Hardware Acceleration**: Automatic CUDA device selection (`cuda` if available else `cpu`).

### Multi-Horizon Evaluation Interface
The evaluation engine evaluates the model across discrete horizon checkpoints:
$$K \in \{5, 10, 15, 20, 25, 30\}$$
For each horizon length $K$, the engine extracts predictions $p_K$ and reports:
- Accuracy, Precision, Recall / TPR, Specificity / TNR, F1-Score
- False Positive Rate (FPR), False Negative Rate (FNR)
- ROC-AUC, PR-AUC, Confusion Matrix ($TP, FP, TN, FN$)
- Mean alert probability and observation budget consumed.

---

## 6. Execution Command for Full Training

> [!NOTE]
> The Temporal GRU infrastructure has been implemented and validated via unit tests. Full training on the 52,714 training sequences has **NOT** been executed yet in adherence to execution constraints.

To execute training and multi-horizon evaluation on your RTX 2050 GPU:
```powershell
python scripts/train_temporal_gru.py
```

*Optional CLI parameters:*
- `--epochs 10`: Training epochs.
- `--batch_size 128`: Mini-batch size.
- `--hidden_dim 64`: GRU hidden units.
- `--step_size 10`: Stride between sequence window starts.

---

## 7. Next Architectural Integrations
With the Temporal GRU foundation established, subsequent stages will integrate:
1. **Multi-View Fusion**: Combining the GRU behavioral embedding $z_{beh} \in \mathbb{R}^{64}$ with the Character-CNN lexical embedding $z_{lex} \in \mathbb{R}^{128}$.
2. **Deep Ensemble Uncertainty Engine**: $M=3$ diverse Temporal GRU models to measure predictive entropy $H(p)$ and ensemble disagreement $\sigma^2_e$.
3. **Adaptive Evidence Controller**: Dynamic stopping threshold policy $\tau(k)$ for early exit under high certainty.
