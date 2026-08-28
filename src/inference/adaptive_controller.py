"""
Adaptive Evidence Controller for DeepDNS Sequential Inference.

Dynamically controls the observation horizon K in [K_min, K_max] to make
early, high-confidence detection decisions while minimizing the number of
DNS queries required before emitting an alert or confirming benign traffic.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


@dataclass
class AdaptiveDecision:
    """Represents a single sequential classification decision by the Adaptive Controller."""
    predicted_label: int
    probability_attack: float
    stopping_horizon: int
    is_early_stop: bool
    step_probabilities: Dict[int, float]
    stopping_reason: str


@dataclass
class AdaptiveEvaluationResult:
    """Aggregated evaluation metrics for an adaptive controller evaluation run."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    fpr: float
    fnr: float
    mean_stopping_horizon: float
    median_stopping_horizon: float
    horizon_decision_percentages: Dict[int, float]
    early_decision_percentage: float
    confusion_matrix: Dict[str, int]
    total_samples: int


class AdaptiveEvidenceController:
    """
    Inference-time sequential evidence controller for DeepDNS Multi-View models.
    
    Observes predictions across sequential horizons K in [K_min, ..., K_max]
    and terminates observation as soon as evidence crosses pre-calibrated confidence bounds.
    """

    def __init__(
        self,
        tau_attack: float = 0.90,
        tau_benign: float = 0.05,
        horizons: Optional[List[int]] = None,
        min_horizon: int = 5,
        max_horizon: int = 30,
    ):
        """
        Args:
            tau_attack: Probability threshold above which an ATTACK decision is emitted early.
            tau_benign: Probability threshold below which a BENIGN decision is emitted early.
            horizons: List of discrete evaluation horizons (default: [5, 10, 15, 20, 25, 30]).
            min_horizon: Minimum observation length before any decision can be made.
            max_horizon: Maximum observation length at which a decision is forced.
        """
        self.tau_attack = tau_attack
        self.tau_benign = tau_benign
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        self.horizons = horizons or [5, 10, 15, 20, 25, 30]

        # Sanity validation
        if not (0.5 <= self.tau_attack <= 1.0):
            raise ValueError(f"tau_attack must be in [0.5, 1.0], got {self.tau_attack}")
        if not (0.0 <= self.tau_benign <= 0.5):
            raise ValueError(f"tau_benign must be in [0.0, 0.5], got {self.tau_benign}")

    def evaluate_sequence_probabilities(
        self,
        step_probabilities: np.ndarray,
    ) -> AdaptiveDecision:
        """
        Evaluates a single sequence's sequential attack probabilities over horizons.

        Args:
            step_probabilities: 1D array of shape (T,) or (max_horizon,) containing
                                P(Attack) at each observation step t in [1..T].

        Returns:
            AdaptiveDecision containing stopping horizon, predicted label, and stats.
        """
        prob_history: Dict[int, float] = {}

        for k in self.horizons:
            if k > len(step_probabilities):
                break
            
            # Step index is 0-indexed (step k -> index k-1)
            p_k = float(step_probabilities[k - 1])
            prob_history[k] = p_k

            # Early Attack Detection Rule
            if p_k >= self.tau_attack:
                return AdaptiveDecision(
                    predicted_label=1,
                    probability_attack=p_k,
                    stopping_horizon=k,
                    is_early_stop=(k < self.max_horizon),
                    step_probabilities=prob_history,
                    stopping_reason=f"ATTACK_THRESHOLD_MET (p={p_k:.4f} >= {self.tau_attack})",
                )

            # Early Benign Confirmation Rule
            if p_k <= self.tau_benign:
                return AdaptiveDecision(
                    predicted_label=0,
                    probability_attack=p_k,
                    stopping_horizon=k,
                    is_early_stop=(k < self.max_horizon),
                    step_probabilities=prob_history,
                    stopping_reason=f"BENIGN_THRESHOLD_MET (p={p_k:.4f} <= {self.tau_benign})",
                )

        # Forced Decision at max_horizon (or latest observed horizon)
        final_k = min(self.max_horizon, len(step_probabilities))
        final_p = float(step_probabilities[final_k - 1])
        prob_history[final_k] = final_p
        forced_label = 1 if final_p >= 0.5 else 0

        return AdaptiveDecision(
            predicted_label=forced_label,
            probability_attack=final_p,
            stopping_horizon=final_k,
            is_early_stop=False,
            step_probabilities=prob_history,
            stopping_reason=f"FORCED_HORIZON_REACHED (p={final_p:.4f})",
        )

    def evaluate_batch(
        self,
        batch_step_probabilities: np.ndarray,
        ground_truth_labels: np.ndarray,
    ) -> Tuple[List[AdaptiveDecision], AdaptiveEvaluationResult]:
        """
        Evaluates a batch of sequences and computes comprehensive adaptive metrics.

        Args:
            batch_step_probabilities: Array of shape (N, T) with P(Attack) at each step.
            ground_truth_labels: 1D array of shape (N,) with true binary labels.

        Returns:
            Tuple of (decisions list, AdaptiveEvaluationResult summary).
        """
        decisions: List[AdaptiveDecision] = []
        preds: List[int] = []
        stopping_horizons: List[int] = []
        horizon_counts: Dict[int, int] = {k: 0 for k in self.horizons}

        for i in range(len(batch_step_probabilities)):
            dec = self.evaluate_sequence_probabilities(batch_step_probabilities[i])
            decisions.append(dec)
            preds.append(dec.predicted_label)
            stopping_horizons.append(dec.stopping_horizon)
            if dec.stopping_horizon in horizon_counts:
                horizon_counts[dec.stopping_horizon] += 1
            else:
                horizon_counts[dec.stopping_horizon] = 1

        y_true = np.asarray(ground_truth_labels, dtype=int)
        y_pred = np.asarray(preds, dtype=int)

        # Compute Confusion Matrix
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        total = len(y_true)

        acc = (tp + tn) / total if total > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        mean_k = float(np.mean(stopping_horizons))
        median_k = float(np.median(stopping_horizons))
        horizon_pcts = {k: round(count / total * 100, 2) for k, count in horizon_counts.items()}
        early_pct = round(sum(count for k, count in horizon_counts.items() if k < self.max_horizon) / total * 100, 2)

        summary = AdaptiveEvaluationResult(
            accuracy=round(acc * 100, 2),
            precision=round(prec * 100, 2),
            recall=round(rec * 100, 2),
            f1_score=round(f1, 4),
            fpr=round(fpr * 100, 4),
            fnr=round(fnr * 100, 4),
            mean_stopping_horizon=round(mean_k, 2),
            median_stopping_horizon=round(median_k, 2),
            horizon_decision_percentages=horizon_pcts,
            early_decision_percentage=early_pct,
            confusion_matrix={"TP": tp, "FP": fp, "TN": tn, "FN": fn},
            total_samples=total,
        )

        return decisions, summary

    @classmethod
    def calibrate_from_validation(
        cls,
        val_step_probabilities: np.ndarray,
        val_labels: np.ndarray,
        target_max_fpr: float = 0.005,
        target_min_recall: float = 0.98,
        horizons: Optional[List[int]] = None,
    ) -> "AdaptiveEvidenceController":
        """
        Calibrates optimal (tau_attack, tau_benign) thresholds strictly on validation data.
        
        Searches candidate thresholds on validation sequences to find the pair that
        minimizes mean stopping horizon while satisfying FPR <= target_max_fpr.
        """
        candidate_tau_atk = [0.80, 0.85, 0.90, 0.92, 0.95, 0.98]
        candidate_tau_ben = [0.01, 0.02, 0.05, 0.10, 0.15]

        best_controller = None
        best_f1 = -1.0
        best_mean_k = 999.0

        for t_atk in candidate_tau_atk:
            for t_ben in candidate_tau_ben:
                ctrl = cls(tau_attack=t_atk, tau_benign=t_ben, horizons=horizons)
                _, res = ctrl.evaluate_batch(val_step_probabilities, val_labels)

                # Select configuration that satisfies safety constraints and maximizes efficiency
                if res.fpr / 100.0 <= target_max_fpr and res.recall / 100.0 >= target_min_recall:
                    if res.mean_stopping_horizon < best_mean_k:
                        best_mean_k = res.mean_stopping_horizon
                        best_f1 = res.f1_score
                        best_controller = ctrl

        # Fallback to standard robust defaults if validation bounds are too strict
        if best_controller is None:
            best_controller = cls(tau_attack=0.90, tau_benign=0.05, horizons=horizons)

        return best_controller
