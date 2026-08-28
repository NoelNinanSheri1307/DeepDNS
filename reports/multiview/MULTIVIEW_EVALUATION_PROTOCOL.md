# DeepDNS Multi-View Evaluation Protocol

## 1. Experimental Setup & Partitioning Protocol
Evaluation is executed strictly on the Session-Grouped partitions defined in `reports/data_pipeline/split_manifest.json`:
- **Train Partition (12 Captures, 52,714 Sequence Windows)**: Models are fit exclusively on this stream.
- **Validation Partition (3 Captures, 10,390 Sequence Windows)**: Hyperparameter and checkpoint selection.
- **Test Partition (3 Captures, 13,084 Sequence Windows)**: Final out-of-distribution evaluation.

---

## 2. Multi-Horizon Evaluation Points
The evaluation pipeline evaluates performance at discrete observation checkpoints:
$$K \in \{5, 10, 15, 20, 25, 30\}$$

For each horizon length $K$, the engine extracts predictions $p_K$ and records:
- **Accuracy**, **Precision**, **Recall / TPR**, **Specificity / TNR**, **F1-Score**
- **False Positive Rate (FPR)**, **False Negative Rate (FNR)**
- **ROC-AUC**, **PR-AUC**, and **Confusion Matrix** ($TP, FP, TN, FN$)

---

## 3. Ablation Protocol Matrix

To evaluate the contribution of each view independently:

| Experiment Name | Model Configuration | CLI Flag | Scientific Objective |
| :--- | :--- | :--- | :--- |
| **Dual-View Fusion** | Temporal GRU + Character-CNN | `--mode both` | Full DeepDNS multi-modal model. |
| **Behavioral Only** | Temporal GRU only | `--mode behavioral_only` | Measures temporal recurrence without lexical text. |
| **Lexical Only** | Character-CNN only | `--mode lexical_only` | Measures raw character orthography without recurrence. |

---

## 4. Execution Command for Full Training (Manual User Execution)

> [!NOTE]
> The Multi-View Fusion infrastructure is implemented and verified via unit tests. Full training on the 52,714 sequences has **NOT** been run yet.

To execute training and multi-horizon evaluation on your RTX 2050 GPU:
```powershell
python scripts/train_multiview.py --mode both
```

*To run specific ablations:*
- Behavioral only: `python scripts/train_multiview.py --mode behavioral_only`
- Lexical only: `python scripts/train_multiview.py --mode lexical_only`
