# Cross-Dataset Compatibility & Synergies

## Structural Comparison

| Dimension | CIC-Bell DNS Exfiltration 2021 | DNS Threats Dataset |
| :--- | :--- | :--- |
| **Primary Data Type** | Temporal Network Telemetry & Statistical Features | Static Domain String Lexical Data |
| **Timestamps** | Microsecond resolution ISO8601 | None |
| **Raw Domain Names** | None (Pre-computed summary stats only) | Full raw string URLs/domains |
| **Sequential Structure** | Intrinsic PCAP packet streams | Independent I.I.D. domain samples |
| **Primary Suitability** | Temporal GRU & Adaptive Evidence Controller | Pre-trained Character-level CNN Lexical Feature Extractor |

## Recommended Cross-Dataset Experimental Protocol
> [!TIP]
> **Decoupled Two-Tier Architecture**:
> 1. **Tier 1 (Lexical Representation)**: Train a Character-level CNN on `dns_threats` (2,482,810 raw domains) to learn a robust lexical embedding z_{lex}.
> 2. **Tier 2 (Temporal Evidence Accumulator)**: Train the Behavioral Feature Encoder + GRU + Adaptive Controller on `CIC-Bell` sequential streams.
> 3. **Zero-Shot / External Robustness Validation**: Evaluate whether the decision controller generalizes across both datasets without catastrophic domain drift.
