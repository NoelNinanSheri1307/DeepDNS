# DeepDNS Baseline Evaluation Protocol

## 1. Experimental Setup & Partitioning Protocol

Evaluation is conducted strictly according to the session-grouped partition defined in `reports/data_pipeline/split_manifest.json`:

```
                       ┌──────────────────────────────────────────────────────────┐
                       │  CIC-Bell Stateless Captures (18 PCAP Sessions, 757,211 rows)  │
                       └────────────────────────────┬─────────────────────────────┘
                                                    │
             ┌──────────────────────────────────────┼──────────────────────────────────────┐
             │                                      │                                      │
             ▼                                      ▼                                      ▼
┌──────────────────────────┐           ┌──────────────────────────┐           ┌──────────────────────────┐
│      TRAIN PARTITION     │           │   VALIDATION PARTITION   │           │      TEST PARTITION      │
│  12 Captures (69.20%)    │           │   3 Captures (13.62%)    │           │   3 Captures (17.18%)    │
│     524,027 Rows         │           │     103,119 Rows         │           │     130,065 Rows         │
├──────────────────────────┤           ├──────────────────────────┤           ├──────────────────────────┤
│ - heavy_audio            │           │ - heavy_image (attack)   │           │ - heavy_video (attack)   │
│ - heavy_compressed       │           │ - light_audio (attack)   │           │ - light_text  (attack)   │
│ - heavy_exe              │           │ - benign_heavy_2 (benign)│           │ - benign_2    (benign)   │
│ - heavy_text             │           └──────────────────────────┘           └──────────────────────────┘
│ - benign_heavy_1, 3      │
│ - light_compressed, exe  │
│ - light_image, video     │
│ - light_benign, benign_1 │
└──────────────────────────┘
```

### Partitioning Invariants
1. **Zero Session Overlap**: No `capture_id` exists in more than one partition.
2. **Scaler Parameter Isolation**: `FeatureScaler` statistics (mean $\mu$, variance $\sigma^2$) are computed exclusively on the 524,027 training rows. Validation and test vectors are normalized using the stored training scaler.

---

## 2. Evaluation Metrics Suite

For each baseline model, metrics are recorded on both Validation and Test splits using [`src/evaluation/metrics.py::compute_classification_metrics()`](file:///c:/Users/VICTUS/deepdns/src/evaluation/metrics.py):

| Metric | Mathematical Formula | SOC Security Significance |
| :--- | :--- | :--- |
| **Accuracy** | $(TP + TN) / (TP + TN + FP + FN)$ | General classification accuracy across balanced streams. |
| **Precision** | $TP / (TP + FP)$ | Confidence that an emitted alert is a genuine exfiltration event. |
| **Recall (TPR)** | $TP / (TP + FN)$ | Proportion of total exfiltration queries successfully intercepted. |
| **Specificity (TNR)** | $TN / (TN + FP)$ | Proportion of benign background traffic correctly ignored. |
| **F1-Score** | $2 \cdot (\text{Precision} \cdot \text{Recall}) / (\text{Precision} + \text{Recall})$ | Harmonic mean balancing alert accuracy against detection completeness. |
| **FPR (False Positive Rate)** | $FP / (FP + TN)$ | **Primary Operational Constraint**: Must remain near zero to avoid alert fatigue in SOCs. |
| **FNR (False Negative Rate)** | $FN / (FN + TP)$ | Proportion of missed tunneling queries ($1 - \text{Recall}$). |
| **ROC-AUC** | $\int_0^1 \text{TPR}(\text{FPR}^{-1}(t)) dt$ | Threshold-independent discrimination power. |
| **PR-AUC** | $\sum_n (R_n - R_{n-1}) P_n$ | Area under Precision-Recall curve under class imbalance. |

---

## 3. Protocol for Running Baseline Training

### Hardware Target
- **GPU**: NVIDIA RTX 2050 (4GB VRAM) / Intel/AMD CPU.
- **VRAM Utilization**: $< 400\text{ MB}$ (MLP batch size = 256).
- **RAM Utilization**: $\approx 1.8\text{ GB}$.
- **Execution Time Estimate**: $\approx 45\text{ seconds}$ on RTX 2050 / multi-core CPU.

### Execution Command
To execute baseline training and evaluation:
```powershell
python scripts/train_baselines.py
```

Optional CLI parameters:
- `--rf_trees 100`: Number of estimators in Random Forest.
- `--mlp_epochs 10`: Number of training epochs for MLP.
- `--mlp_batch_size 256`: Mini-batch size for PyTorch data loader.
- `--mlp_lr 0.001`: AdamW learning rate.
- `--max_train_samples 10000`: Fast debugging subsample per capture.

### Generated Artifacts
- Models saved to: `data/processed/models/`
  - `rule_based_detector.json`
  - `random_forest_baseline.joblib`
  - `mlp_baseline.pt`
- Evaluation results saved to: `reports/model_baselines/baseline_results.json`
