"""
Evaluation package for DeepDNS.
"""

from src.evaluation.metrics import (
    EvaluationReport,
    compute_classification_metrics,
)

__all__ = [
    "EvaluationReport",
    "compute_classification_metrics",
]
