"""
Evaluation metrics for DeepDNS security and anomaly detection.

Provides standard classification metrics with explicit security-focused metrics
including False Positive Rate (FPR), False Negative Rate (FNR), Precision, Recall,
F1, ROC-AUC, PR-AUC, and confusion matrices.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


@dataclass
class EvaluationReport:
    """Immutable container for comprehensive model evaluation metrics."""
    model_name: str
    dataset_split: str
    total_samples: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    fpr: float
    fnr: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    tp: int
    fp: int
    tn: int
    fn: int

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "dataset_split": self.dataset_split,
            "total_samples": self.total_samples,
            "accuracy": round(self.accuracy, 6),
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "specificity": round(self.specificity, 6),
            "f1": round(self.f1, 6),
            "fpr": round(self.fpr, 6),
            "fnr": round(self.fnr, 6),
            "roc_auc": round(self.roc_auc, 6) if self.roc_auc is not None else None,
            "pr_auc": round(self.pr_auc, 6) if self.pr_auc is not None else None,
            "confusion_matrix": {
                "tp": self.tp,
                "fp": self.fp,
                "tn": self.tn,
                "fn": self.fn,
            },
        }

    def summary_table(self) -> str:
        """Returns formatted ASCII table of metrics."""
        roc_str = f"{self.roc_auc:.4f}" if self.roc_auc is not None else "N/A"
        pr_str = f"{self.pr_auc:.4f}" if self.pr_auc is not None else "N/A"
        return (
            f"=========================================================\n"
            f"EVALUATION REPORT: {self.model_name.upper()} ({self.dataset_split.upper()})\n"
            f"=========================================================\n"
            f"Total Samples: {self.total_samples:,}\n"
            f"---------------------------------------------------------\n"
            f"Accuracy:    {self.accuracy * 100:.2f}%\n"
            f"Precision:   {self.precision * 100:.2f}%\n"
            f"Recall:      {self.recall * 100:.2f}% (Detection Rate)\n"
            f"F1-Score:    {self.f1:.4f}\n"
            f"ROC-AUC:     {roc_str}\n"
            f"PR-AUC:      {pr_str}\n"
            f"---------------------------------------------------------\n"
            f"FPR (False Positive Rate): {self.fpr * 100:.4f}% (Target: Low)\n"
            f"FNR (False Negative Rate): {self.fnr * 100:.4f}%\n"
            f"---------------------------------------------------------\n"
            f"Confusion Matrix: [TN={self.tn:,}, FP={self.fp:,} | FN={self.fn:,}, TP={self.tp:,}]\n"
            f"========================================================="
        )


def compute_classification_metrics(
    y_true: Union[np.ndarray, List[int]],
    y_pred: Union[np.ndarray, List[int]],
    y_prob: Optional[Union[np.ndarray, List[float]]] = None,
    model_name: str = "model",
    dataset_split: str = "test",
) -> EvaluationReport:
    """
    Computes standard and cybersecurity-specific metrics from predictions.

    Args:
        y_true: Ground truth binary targets (0 = Benign, 1 = Attack).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Optional predicted probabilities for positive class (Class 1).
        model_name: Identifier for model being evaluated.
        dataset_split: Name of dataset partition (e.g. 'val', 'test').

    Returns:
        EvaluationReport dataclass containing all computed metrics.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: y_true ({len(y_t)}) vs y_pred ({len(y_p)})")

    if len(y_t) == 0:
        raise ValueError("Cannot compute metrics on empty arrays.")

    # Confusion matrix elements
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    total = len(y_t)
    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    rec = float(recall_score(y_t, y_p, zero_division=0))
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    # Security metrics
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # Probability based metrics
    roc_auc = None
    pr_auc = None
    if y_prob is not None:
        y_pr = np.asarray(y_prob, dtype=float)
        # Verify if both classes are present in y_true
        if len(np.unique(y_t)) > 1:
            try:
                roc_auc = float(roc_auc_score(y_t, y_pr))
                pr_auc = float(average_precision_score(y_t, y_pr))
            except Exception:
                pass

    return EvaluationReport(
        model_name=model_name,
        dataset_split=dataset_split,
        total_samples=total,
        accuracy=acc,
        precision=prec,
        recall=rec,
        specificity=specificity,
        f1=f1,
        fpr=fpr,
        fnr=fnr,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
    )
