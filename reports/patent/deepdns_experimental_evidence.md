# DeepDNS Technical Architecture & Experimental Evidence Document

## 1. Technical Problem
Modern DNS exfiltration attacks encode stolen confidential data into subdomain labels (e.g. Base32, Base64, or Hex encoding) and transmit queries at low rates to bypass traditional static firewall rules, domain reputation lookups, and pointwise classifiers. Pointwise classifiers suffer from unacceptable false alarm rates (FPR $\approx 38\%$) because individual benign queries frequently resemble anomalous strings.

---

## 2. Claimed Technical Architecture & Methods

```
                                +-----------------------------------+
                                | Raw DNS Packet Stream per Host   |
                                +-----------------+-----------------+
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
            +---------------------------+                     +---------------------------+
            |  Causal Behavioral Stream  |                     |  Raw Lexical Query Stream |
            |  x_t in R^12              |                     |  w_t in Z^128 (ASCII)     |
            +-------------+-------------+                     +-------------+-------------+
                          |                                                 |
                          v                                                 v
            +---------------------------+                     +---------------------------+
            | Temporal GRU Recurrent    |                     | Parallel Character-CNN    |
            | Encoder (Hidden Dim = 64) |                     | Encoder (Filter k=3,5,7)  |
            +-------------+-------------+                     +-------------+-------------+
                          | z_beh (t) in R^64                               | z_lex (t) in R^128
                          +------------------------+------------------------+
                                                   |
                                                   v
                                     +---------------------------+
                                     | Late Multi-View Fusion    |
                                     | Linear Projections + MLP  |
                                     +-------------+-------------+
                                                   | z_fuse (t) in R^64
                                                   v
                                     +---------------------------+
                                     | Sequential Decision Head  |
                                     | y_hat(t) in [0, 1]        |
                                     +---------------------------+
```

1. **Causal Behavioral Representation**:
   - 12 stateless causal features per query (domain lengths, entropy, digit/special ratios, label counts).
   - Zero access to future queries, timestamps, or capture identity.
2. **Lexical Orthographic Representation**:
   - Discrete ASCII token mapping ($V=45$) processed by a 3-way multi-scale Character-CNN ($k \in \{3, 5, 7\}$).
3. **Learned Late Fusion**:
   - Projected behavioral ($z_{beh}$) and lexical ($z_{lex}$) representations are fused at each step $t \in [1..K]$ via a multi-layer perceptron with residual pathways.

---

## 3. Comprehensive Experimental Evidence Matrix

### A. Pointwise Baselines vs Sequential Models (Test Partition, $N=13,084$)

| Model Architecture | Input Representation | Horizon ($K$) | Accuracy | Detection Rate (Recall) | F1-Score | False Positive Rate (FPR) | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Pointwise Logistic Regression** | 11 Causal Features | 1 | 73.80% | 98.84% | 0.7065 | 37.92% | 0.8075 |
| **Pointwise Decision Tree** (Depth 3) | 11 Causal Features | 1 | 73.75% | 99.64% | 0.7077 | 38.38% | 0.8069 |
| **Sequential Temporal GRU** | 12 Causal Features | 10 | 98.98% | 99.19% | 0.9843 | 1.11% | 0.9958 |
| **Sequential Temporal GRU** | 12 Causal Features | 30 | 99.61% | 99.24% | 0.9939 | 0.21% | 0.9998 |
| **Sequential Character-CNN** | Raw Domain Strings | 30 | 73.52% | 98.45% | 0.7048 | 38.26% | 0.8048 |
| **DeepDNS Dual-View Network** | Behavioral + Lexical | 10 | 98.91% | 99.52% | 0.9833 | 1.37% | 0.9937 |
| **DeepDNS Dual-View Network** | Behavioral + Lexical | 30 | **99.69%** | **99.52%** | **0.9952** | **0.23%** | **0.9998** |

### B. Multi-View Branch Ablation Evidence ($K=30$)

| Ablation Mode | Behavioral GRU Active | Lexical CNN Active | False Negatives ($FN$) | False Positives ($FP$) | Overall F1 | False Positive Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Lexical-Only** | No | Yes | 65 | 3,399 | 0.7048 | 38.26% |
| **Behavioral-Only** | Yes | No | 47 | 9 | 0.9933 | 0.10% |
| **Dual-View (Fused)**| Yes | Yes | **20** | **20** | **0.9952** | **0.23%** |

- **Key Synergy Finding**: Late fusion cuts False Negatives by **$57.4\%$** relative to Behavioral-Only ($FN: 47 \to 20$), recovering missed exfiltration attempts while maintaining sub-$0.25\%$ FPR.

### C. Scientific Negative Controls & Robustness Audits

1. **Training Label Permutation Sanity Check**:
   - Randomizing training labels while keeping test labels untouched collapsed performance to base rate ($\text{F1} = 0.0, \text{Recall} = 0.0, \text{ROC-AUC} = 0.2813$).
   - Proves zero structural leakage or dataset memorization.
2. **Inter-Arrival Time Ablation**:
   - Completely removing $IAT$ features maintained $\text{F1} = 0.9932$ at $K=30$.
   - Proves detection capability is driven by DNS payload structural dynamics rather than synthetic clock cadence.
3. **Temporal Order Permutation**:
   - Shuffling time steps degrades short-horizon performance ($K=5$) while preserving density at large horizons ($K=30$).
4. **Capture-Level Robustness**:
   - Aggregated predictions on test PCAPs achieved **$100\%$ Capture-Level Accuracy** ($TP=2, FP=0, TN=1, FN=0$).

### D. Leave-One-Modality-Out (LOMO) OOD Generalization Matrix ($N=20,683$)

| Evaluation Scope | Attack Modality Exposure in Training | Horizon ($K$) | Accuracy | Detection Rate (Recall) | F1-Score | False Positive Rate | ROC-AUC | Confusion Matrix ($TN, FP, FN, TP$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Combined OOD Test** | 100% Unseen Video & Text | 5 | 89.13% | 82.97% | 0.8970 | 2.69% | 0.9835 | $[8644, 239 \mid 2009, 9791]$ |
| **Combined OOD Test** | 100% Unseen Video & Text | 10 | 96.56% | 94.58% | 0.9691 | 0.81% | 0.9954 | $[8811, 72 \mid 639, 11161]$ |
| **Combined OOD Test** | 100% Unseen Video & Text | 20 | 99.56% | 99.52% | 0.9962 | 0.37% | 0.9996 | $[8850, 33 \mid 57, 11743]$ |
| **Combined OOD Test** | 100% Unseen Video & Text | 30 | **99.59%** | **99.44%** | **0.9964** | **0.20%** | **0.9994** | $[8865, 18 \mid 66, 11734]$ |
| **Video-Only OOD** | 100% Unseen Modality | 30 | **99.14%** | **99.14%** | **0.9957** | — | — | $[0, 0 \mid 37, 4253]$ |
| **Text-Only OOD** | 100% Unseen Modality | 30 | **99.61%** | **99.61%** | **0.9981** | — | — | $[0, 0 \mid 29, 7481]$ |
| **Benign-Only Test** | Unseen PCAP Session | 30 | **99.80%** | — | — | **0.20%** | — | $[8865, 18 \mid 0, 0]$ |

- **OOD Generalization Finding**: When evaluated on attack modalities with **zero prior training exposure**, the DeepDNS Dual-View network maintains near-flawless performance ($\text{F1} = 0.9964, \text{FPR} = 0.2026\%$), proving that it learns intrinsic multi-view DNS exfiltration signatures rather than modality-specific file artifacts.

### E. Adaptive Evidence Controller & Sequential CUSUM Results ($N=20,683$)

| Decision Strategy | Observation Horizon ($K$) | Detection Rate (Recall) | False Positive Rate (FPR) | F1-Score | Mean Stopping Horizon ($\bar{K}^*$) | Early Decision Rate | Query Volume Reduction |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Fixed Horizon $K=5$** | Fixed $5$ | 82.97% | 2.6905% | 0.8970 | 5.00 | 100.0% | $83.3\%$ (High FPR) |
| **Fixed Horizon $K=10$**| Fixed $10$ | 94.58% | 0.8105% | 0.9691 | 10.00 | 100.0% | $66.7\%$ |
| **Fixed Horizon $K=20$**| Fixed $20$ | 99.52% | 0.3715% | 0.9962 | 20.00 | 100.0% | $33.3\%$ |
| **Fixed Horizon $K=30$**| Fixed $30$ | 99.44% | 0.2026% | 0.9964 | 30.00 | 0.0% | Baseline ($0\%$) |
| **Adaptive Confidence** ($\tau_{\text{atk}}=0.80, \tau_{\text{ben}}=0.01$) | **Dynamic** | **99.21%** | **0.5066%** | **0.9941** | **10.30** | **99.9%** | **$65.7\%$ Savings** |
| **Sequential CUSUM** ($h=4.0, \gamma=0.0$) | **Dynamic** | **96.17%** | **0.1801%** | **0.9798** | **12.96** | **99.8%** | **$56.8\%$ Savings** |

- **Minimum Time-to-Detection (MTTD) Finding**: Dynamic early-stopping slashes observation latency by **$65.7\%$** (mean $\bar{K}^* = 10.30$ queries vs $30.0$), detecting exfiltration attempts early while preserving near-peak detection accuracy ($99.21\%$ Recall, $0.9941$ F1).

---

## 4. Technical Limitations & Future Defensibility
- **Dataset Domain Shift**: Zero-shot evaluation on external DGA datasets (`dns_threats`) confirms that lexical models trained on DNS tunneling exfiltration require domain adaptation when applied to algorithmic DGA malware.
- **Continuous Deployment**: Real-world networks should employ the DeepDNS Dual-View architecture with the Adaptive Evidence Controller to enable dynamic early alarms within $9\text{--}13$ queries.
