# DeepDNS Baseline Model Suite Architecture

## 1. Executive Overview
The DeepDNS Baseline Model Suite establishes reference benchmarks for DNS exfiltration and tunneling detection before deploying the full adaptive multi-view architecture. All baselines consume the identical, verified 12-dimensional causal feature representation derived during the data-engineering phase.

---

## 2. Baseline Models Specification

### Baseline 1: Rule-Based Deterministic Detector (`RuleBasedDetector`)
- **Location**: [`src/models/rule_based.py`](file:///c:/Users/VICTUS/deepdns/src/models/rule_based.py)
- **Role**: Serves as the domain-heuristic / RFC-style baseline (standard SOC signature/threshold detection).
- **Core Mechanism**:
  A DNS observation is classified as an Attack ($y=1$) if it triggers $\ge M$ violations among 5 structural threshold rules:
  1. $\text{Entropy} \ge \theta_{entropy}$ (Default 3.20)
  2. $\text{FQDN Length} \ge \theta_{fqdn}$ (Default 30.0)
  3. $\text{Subdomain Length} \ge \theta_{subdomain}$ (Default 12.0)
  4. $\text{Digit Ratio} \ge \theta_{digit}$ (Default 0.25)
  5. $\text{Payload Length} \ge \theta_{payload}$ (Default 50.0)
- **Threshold Tuning**:
  Thresholds are tunable strictly on the **Training partition** ($\theta_i$ set to the 25th percentile of positive attack queries). Thresholds are never fit on validation or test sets.

---

### Baseline 2: Random Forest Classifier (`RandomForestBaseline`)
- **Location**: [`src/models/random_forest.py`](file:///c:/Users/VICTUS/deepdns/src/models/random_forest.py)
- **Role**: Strong non-linear tabular benchmark capturing feature interactions across the 12 causal channels.
- **Hyperparameters**:
  - `n_estimators`: 100 decision trees (configurable via `--rf_trees`)
  - `max_depth`: 16 (bounded to prevent memorization of noise)
  - `min_samples_split`: 5
  - `min_samples_leaf`: 2
  - `class_weight`: `"balanced"` (inversely proportional to class frequencies)
  - `random_state`: 42 (deterministic reproducibility)
  - `n_jobs`: -1 (multithreaded CPU parallelization)
- **Capabilities**:
  - Outputs binary predictions and calibrated class probability distributions $[P(0), P(1)]$.
  - Generates Gini feature importance rankings mapped to `CAUSAL_FEATURE_NAMES`.
  - Serializes to disk via `joblib`.

---

### Baseline 3: Lightweight Multi-Layer Perceptron (`MLPClassifierWrapper`)
- **Location**: [`src/models/mlp_baseline.py`](file:///c:/Users/VICTUS/deepdns/src/models/mlp_baseline.py)
- **Role**: Neural representation benchmark matching the behavioral encoder dimension ($d=128$) of the planned DeepDNS architecture.
- **PyTorch Network Architecture**:
  ```
  Input Tensor (B, 12)
        │
        ▼
  Linear(12, 64)
        │
        ▼
  LayerNorm(64)
        │
        ▼
     ReLU()
        │
        ▼
  Dropout(p=0.1)
        │
        ▼
  Linear(64, 32)
        │
        ▼
     ReLU()
        │
        ▼
  Linear(32, 2) ──► Output Logits (B, 2) ──► Softmax Probas (B, 2)
  ```
- **Training Hyperparameters**:
  - Optimizer: `AdamW` ($\text{lr} = 10^{-3}$, $\text{weight\_decay} = 10^{-4}$)
  - Loss Function: `nn.CrossEntropyLoss()`
  - Batch Size: 256
  - Epochs: 10
  - Hardware Execution: GPU accelerated on NVIDIA RTX 2050 (fallback to CPU).

---

## 3. Common Baseline Interface Contract

All three baselines implement a consistent interface:

```python
class BaselineModel:
    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> "BaselineModel":
        """Fits parameters/thresholds strictly on training data."""
        ...

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Emits binary predictions (0 or 1)."""
        ...

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Emits class probabilities of shape (N, 2)."""
        ...

    def save(self, file_path: Union[str, Path]) -> None:
        """Serializes model weights and configuration to disk."""
        ...

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "BaselineModel":
        """Loads serialized model from disk."""
        ...
```

---

## 4. Input Features & Channel Mapping

Input tensors must strictly adhere to the 12 causal channels defined in `src/data/schema.py::CAUSAL_FEATURE_NAMES`:

| Index | Feature Name | Source / Derivation |
| :--- | :--- | :--- |
| 0 | `inter_arrival_time` | $\ln(1 + \max(\Delta t_i, 0))$ |
| 1 | `fqdn_length` | Total character count of FQDN |
| 2 | `subdomain_length` | Length of subdomain labels |
| 3 | `char_entropy` | Shannon character entropy |
| 4 | `digit_ratio` | $\text{numeric} / (\text{FQDN\_count} + 10^{-6})$ |
| 5 | `uppercase_ratio` | $\text{upper} / (\text{FQDN\_count} + 10^{-6})$ |
| 6 | `special_ratio` | $\text{special} / (\text{FQDN\_count} + 10^{-6})$ |
| 7 | `label_count` | Count of dot-separated labels |
| 8 | `max_label_length` | Maximum label length |
| 9 | `avg_label_length` | Average label length |
| 10 | `has_subdomain` | Binary indicator $\{0, 1\}$ |
| 11 | `payload_len` | Packet payload byte length |

**Forbidden Inputs**: All 3 models enforce immediate exception raising (`ValueError: Data Leakage Breach`) if forbidden columns (`sld`, `longest_word`, raw timestamps, `capture_id`, etc.) are passed.
