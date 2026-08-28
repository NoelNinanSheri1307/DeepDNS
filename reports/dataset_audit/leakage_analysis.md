# Data Leakage & Feature Risk Forensic Report

## Overview
We audited all 15 stateless and 27 stateful features for target leakage, non-causal dependencies, laboratory artifacts, and artificial discriminators.

## Feature Risk Categorization Summary
- **SAFE**: 11 features (Suitable for online, causal sequential modeling).
- **SUSPICIOUS**: 6 features (Require normalization, clipping, or verification).
- **LEAKAGE-PRONE**: 25 features (Stateful window summaries, set-string dumps, or lab artifacts).

## Key Risk Findings
1. **Stateful Batch Leakage**:
   - All stateful features (`ttl_variance`, `distinct_domains`, `distinct_ip`) reflect post-capture aggregations. Using them directly assumes knowledge of past and future packets in the session window.
2. **Lab Environment Artifacts**:
   - `sld` in stateless data has low unique cardinalities in attack files because the attack was hosted on a fixed synthetic test domain (e.g. `tunnel.example.com`).
   - If a model learns `sld` directly, it will overfit to the testbed domain rather than learning tunneling mechanics!
3. **Timestamp Target Independence**:
   - Absolute timestamps must NEVER be fed as raw numerical features (e.g. epoch float) to avoid the model simply classifying attack date/time. Only relative inter-arrival times (delta t = t_i - t_{i-1}) are causally valid.

## Feature Risk Table
See machine-readable artifact: `reports/dataset_audit/feature_risk_matrix.csv`.
