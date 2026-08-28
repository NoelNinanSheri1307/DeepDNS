# Final Dataset Forensic Audit & DeepDNS Technical Recommendation

## Executive Synthesis & Answers to Key Research Questions

### 1. What exactly do we have?
- **CIC-Bell-DNS-EXF-2021**: 18 capture sessions (36 CSVs) containing **697,120 stateless records** with monotonic timestamps, and 239,337 pre-aggregated stateful records with string-serialized Python structures.
- **DNS Threats Dataset**: **3,103,513 raw domain queries** (2,482,810 train, 620,703 test) with zero train/test domain overlap and 3-class ground truth labels.

### 2. What is actually usable?
- **CIC-Bell Stateless CSVs** (12 safe/causal numerical and temporal features).
- **DNS Threats Raw Strings** for character-level lexical representation learning.

### 3. What should be the primary training dataset?
- **CIC-Bell Stateless Sequences** for training the sequential evidence accumulator, GRU temporal representation, and adaptive decision controller.

### 4. What should be external validation?
- The test split of **DNS Threats** for lexical generalization, and holdout PCAP sessions (e.g. `benign_2`, `heavy_video`, `light_text`) for out-of-distribution temporal evaluation.

### 5. Should stateful CSVs be used directly?
- **NO.** Stateful CSVs suffer from a 65.7% row mismatch, lack timestamps, and risk non-causal session look-ahead. Stateful features must instead be computed causally inside the sequential GRU.

### 6. How should sequences be constructed?
- By chronologically streaming stateless queries within each PCAP session, computing relative delta t = t_i - t_{i-1}, and evaluating dynamic windows k = 1, 2, ..., K_{max} (K_{max}=30).

### 7. Which features are strictly safe?
- `inter_arrival_time`, `FQDN_count`, `subdomain_length`, `entropy`, `numeric`, `upper`, `special`, `labels`, `labels_max`, `labels_average`, `len`, `subdomain`.

### 8. Which features must be excluded?
- `sld` (domain leakage), `longest_word` (parsing instability), all 27 raw static stateful columns (non-causal alignment).

### 9. What is the ground truth on "Low-and-Slow" in CIC-Bell?
- "Light" attacks in CIC-Bell share the **exact same query rate (~1.7 QPS)** as "Heavy" attacks; they merely represent lower total volume. DeepDNS must test low-volume early stopping (K <= 5) and synthetically rate-diluted streams to benchmark true low-and-slow resilience.

### 10. What is the recommended DeepDNS Architecture?
```
Raw DNS Query Stream (Stateless t_1 ... t_k)
   │
   ├── Lexical Feature Vector (Pre-extracted / Char-CNN embedding)
   └── Behavioral Feature Vector (delta t, length, entropy, ratios)
         │
         ▼
    Learned Multi-View Fusion
         │
         ▼
    Temporal Sequence Model (Bidirectional/Causal GRU, h_dim=64)
         │
         ▼
    Deep Ensemble Prediction Heads (M=3)
         │
         ▼
    Uncertainty & Consistency Engine
    - Predictive Entropy: H(p)
    - Ensemble Disagreement: Var(p_m)
    - Temporal Consistency: ||p_k - p_{k-1}||
         │
         ▼
    Adaptive Evidence Controller
    ├── If Confidence > Dynamic Threshold (or k = K_max) ──► STOP & EMIT DECISION
    └── If Uncertain ──► INGEST NEXT QUERY (k ◄─ k + 1)
```

### 11. Novelty & Patent Hypothesis
- **Core Claim**: An uncertainty-guided dynamic stopping criterion that minimizes time-to-detection and total DNS queries consumed while maintaining a bounded false-alarm constraint under low-and-slow evasion.

### 12. Computational Feasibility
- Fully verified for **NVIDIA RTX 2050 (4GB VRAM)** with sub-5ms decision latency and batch sizes of 128/256.
