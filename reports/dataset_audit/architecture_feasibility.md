# Architecture Component Feasibility Assessment

| Component | Feasibility Status | Input Required | Supported by Data? | Recommendation |
| :--- | :--- | :--- | :--- | :--- |
| **Character-Level CNN** | **FEASIBLE (via DNS Threats)** | Raw character ASCII sequence (L <= 128) | Yes in `dns_threats`, No in `cic_bell` | **KEEP**: Pre-train on `dns_threats` or use Tabular Lexical MLP on `cic_bell`. |
| **Behavioral Encoder** | **FEASIBLE** | 12-dim causal feature vector | Yes (Stateless CIC-Bell) | **KEEP**: Lightweight 2-layer MLP with LayerNorm. |
| **Temporal GRU** | **FEASIBLE** | Sequence X in R^{B x K x D} | Yes (Monotonic timestamps in CIC-Bell) | **KEEP**: Single or 2-layer GRU with hidden dim 64. |
| **Multi-View Fusion** | **FEASIBLE** | Lexical embedding z_lex + Behavioral z_beh | Yes | **KEEP**: Gated cross-attention or bilinear fusion. |
| **Deep Ensemble Uncertainty** | **FEASIBLE** | Ensemble of M=3 lightweight models | Yes | **KEEP**: Fast inference on RTX 2050 (under 5ms/window). |
| **Adaptive Evidence Controller** | **FEASIBLE (Core Novelty)** | Entropy H(p), Ensemble Variance sigma^2_e, Step k | Yes | **KEEP**: Dynamic stopping threshold tau_k = alpha * exp(-beta * k). |
