# DNS Threats Dataset Forensic Audit

## Overview
The `dns_threats` dataset is a large-scale static lexical domain repository provided in compressed format (`.csv.gz`).

### Dataset Statistics
- **Training Set**: 2,482,810 rows, 2 columns (`domain`, `class`).
- **Testing Set**: 620,703 rows, 2 columns (`domain`, `class`).
- **Total Samples**: 3,103,513 domains.
- **Train/Test Domain Overlap**: **0 (0.00% leakage)**.

### Multiclass Label Distribution
| Class ID | Label Semantic | Train Count | Train % | Test Count | Test % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **0** | Benign (Legitimate Domains) | 944,106 | 38.03% | 236,072 | 38.03% |
| **1** | Malicious / Tunneling / DGA / Phishing | 1,532,263 | 61.71% | 383,072 | 61.71% |
| **2** | Suspicious / Specialized Threat | 6,441 | 0.26% | 1,559 | 0.26% |

### Domain String Lexical Properties
- **Train Mean Length**: 14.23 chars (Range: 1 – 65).
- **Test Mean Length**: 18.33 chars (Range: 4 – 253).

## Key Capabilities & Limitations
1. **Raw Domain Presence**: Unlike CIC-Bell, `dns_threats` provides full, raw domain strings, making it **100% suitable for Character-level CNN / Char-Embedding experiments**.
2. **Lack of Temporal / Network Telemetry**: It contains no timestamps, packet sizes, query types, or session groups. It cannot be used for temporal GRU sequence accumulation on its own.
