# DeepDNS Canonical Inference Engine Documentation

## 1. Architecture Overview

The **DeepDNS Canonical Inference Engine** converts the trained multi-view neural networks and the Adaptive Evidence Controller into an enterprise-ready, callable Python module.

```
DNS Query Stream (DNSStream / DataFrame / Dict List)
                    │
                    ▼
       Causal Preprocessing Pipeline
      ┌─────────────┴─────────────┐
      ▼                           ▼
12 Causal Features         Character Tokens
  (Normalized R^12)         (ASCII Indices V=45, L=128)
      │                           │
      ▼                           ▼
Temporal GRU Network       Character-CNN Encoder
  (z_beh in R^64)            (z_lex in R^128)
      └─────────────┬─────────────┘
                    ▼
         Dual-View Fusion Head
         (z_fuse in R^64 -> Logits R^2)
                    │
                    ▼
     Sequential Step Probabilities P(Attack)
                    │
                    ▼
       Adaptive Evidence Controller (AEC)
        ├── K=5:  Crosses tau_attack / tau_benign? ──> [EMIT EARLY DECISION]
        ├── K=10: Crosses tau_attack / tau_benign? ──> [EMIT EARLY DECISION]
        ├── K=15: Crosses tau_attack / tau_benign? ──> [EMIT EARLY DECISION]
        ├── K=20: Crosses tau_attack / tau_benign? ──> [EMIT EARLY DECISION]
        ├── K=25: Crosses tau_attack / tau_benign? ──> [EMIT EARLY DECISION]
        └── K=30: Forced Bounded Decision ───────────> [EMIT BOUNDED DECISION]
```

---

## 2. Input Contract & Data Structures

### `DNSObservation`
Represents a single observed DNS query:
- `timestamp`: Query timestamp string or float.
- `domain_name`: Fully Qualified Domain Name (FQDN) or SLD string.
- `fqdn_count`, `subdomain_length`, `entropy`, `numeric`, `upper`, `special`, etc.: Raw stateless numerical features.

### `DNSStream`
A container representing an ordered observation sequence belonging to a host or client session.

### `DetectionResult`
The structured dataclass returned by `engine.detect(stream)`:
- `decision`: `"ATTACK"` or `"BENIGN"`.
- `confidence`: float in $[0.0, 1.0]$.
- `attack_probability`: float in $[0.0, 1.0]$.
- `benign_probability`: float in $[0.0, 1.0]$ ($1 - p_{\text{attack}}$).
- `stopping_horizon`: Horizon $K \in \{5, 10, 15, 20, 25, 30\}$ at which observation stopped.
- `max_horizon`: $30$.
- `observations_consumed`: Equal to `stopping_horizon`.
- `observations_saved`: $30 - \text{stopping\_horizon}$.
- `query_savings_pct`: Percentage of queries saved vs static $K=30$ buffering.
- `is_early_decision`: True if $K < 30$.
- `stop_reason`: Explanatory string (e.g. `"ATTACK_THRESHOLD_MET (p=0.9850 >= 0.95)"`).
- `evidence_history`: Map of $\{K \to P(\text{Attack})\}$.

---

## 3. Supported Modes & Checkpoint Resolution

```python
from src.inference import DeepDNSInferenceEngine, InferenceMode

# 1. In-Distribution Mode (Standard Multi-View)
engine = DeepDNSInferenceEngine(mode=InferenceMode.IN_DISTRIBUTION)
# Uses: multiview_both.pt + feature_scaler.json
# Calibrated Thresholds: tau_attack = 0.95, tau_benign = 0.15

# 2. Out-of-Distribution Mode (LOMO / Unseen Modalities)
engine_ood = DeepDNSInferenceEngine(mode=InferenceMode.OOD)
# Uses: multiview_ood_both.pt + feature_scaler_lomo.json
# Calibrated Thresholds: tau_attack = 0.80, tau_benign = 0.01

# 3. Behavioral-Only Ablation Mode
engine_beh = DeepDNSInferenceEngine(mode=InferenceMode.BEHAVIORAL_ONLY)
# Uses: multiview_behavioral_only.pt

# 4. Lexical-Only Ablation Mode
engine_lex = DeepDNSInferenceEngine(mode=InferenceMode.LEXICAL_ONLY)
# Uses: multiview_lexical_only.pt
```

---

## 4. Usage Examples

### A. Batch / Complete Sequence Inference
```python
from src.inference import DeepDNSInferenceEngine, DNSStream, DNSObservation
import pandas as pd

# Load engine
engine = DeepDNSInferenceEngine(mode="in_distribution")

# From DataFrame
df = pd.read_csv("data/raw/cic_bell_dns_exf_2021/stateless/stateless_features-light_text.pcap.csv", nrows=30)
result = engine.detect(df)

print(f"Decision:   {result.decision}")
print(f"Confidence: {result.confidence:.2%}")
print(f"Stopped at: K={result.stopping_horizon} queries (Saved {result.query_savings_pct:.1f}% queries)")
print(f"Reason:     {result.stop_reason}")
```

### B. Incremental Streaming Inference
```python
stream = DNSStream(stream_id="host_192.168.1.50")

# Progressively feed DNS queries as they arrive
for query_dict in incoming_dns_traffic:
    result, status = engine.process_observation(stream, query_dict)
    
    if status == "CONTINUE":
        # Keep waiting for more queries
        continue
    else:
        # Decision reached (stopped early or forced at K=30)
        print(f"Alert: {result.decision} triggered at horizon K={result.stopping_horizon}")
        break
```

---

## 5. Running the Test Suite

```powershell
python -m pytest -v tests/test_inference_engine.py
```
*(All 67 repository tests pass deterministically with 100% reproducibility).*
