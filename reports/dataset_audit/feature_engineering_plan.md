# Causal Feature Engineering Plan

## Verified Causal Feature Set for DeepDNS Streaming

| Feature Name | Source Column | Derivation / Transformation | Causal Status | Risk Level |
| :--- | :--- | :--- | :--- | :--- |
| `inter_arrival_time` | `timestamp` | delta t_i = t_i - t_{i-1} (log-scaled) | Strictly Causal | **SAFE** |
| `fqdn_length` | `FQDN_count` | Robust standard scaling | Strictly Causal | **SAFE** |
| `subdomain_length` | `subdomain_length` | Numerical standardization | Strictly Causal | **SAFE** |
| `char_entropy` | `entropy` | Value normalization [0, 5] | Strictly Causal | **SAFE** |
| `digit_ratio` | `numeric`, `FQDN_count` | numeric / (FQDN_count + epsilon) | Strictly Causal | **SAFE** |
| `uppercase_ratio` | `upper`, `FQDN_count` | upper / (FQDN_count + epsilon) | Strictly Causal | **SAFE** |
| `special_ratio` | `special`, `FQDN_count` | special / (FQDN_count + epsilon) | Strictly Causal | **SAFE** |
| `label_count` | `labels` | Standard scaling | Strictly Causal | **SAFE** |
| `max_label_length` | `labels_max` | Standard scaling | Strictly Causal | **SAFE** |
| `avg_label_length` | `labels_average` | Standard scaling | Strictly Causal | **SAFE** |
| `has_subdomain` | `subdomain` | Binary pass-through {0, 1} | Strictly Causal | **SAFE** |
| `payload_len` | `len` | Standard scaling | Strictly Causal | **SAFE** |

## Features Excluded to Prevent Leakage
- `sld`: Excluded (contains fixed testbed domain hashes).
- `longest_word`: Excluded (unstable string parsing artifact in raw CSV).
- All 27 raw `stateful_features-*.csv` columns: Excluded in favor of internal GRU hidden state recurrence.
