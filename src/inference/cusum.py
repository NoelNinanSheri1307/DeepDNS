"""
Sequential Cumulative Sum (CUSUM) Statistical Change-Point Detector for DeepDNS.

Tracks sequential log-likelihood ratios over time to detect shifts from
benign background DNS traffic to malicious exfiltration tunnels.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from src.inference.adaptive_controller import AdaptiveDecision, AdaptiveEvaluationResult


class SequentialCUSUMDetector:
    """
    Two-sided Sequential CUSUM Evidence Detector for DNS Streams.
    
    Accumulates positive log-likelihood evidence for attacks and negative
    evidence for benign flows across observation horizons.
    """

    def __init__(
        self,
        h_attack: float = 4.0,
        h_benign: float = 4.0,
        drift: float = 0.0,
        horizons: Optional[List[int]] = None,
        min_horizon: int = 5,
        max_horizon: int = 30,
        eps: float = 1e-6,
    ):
        """
        Args:
            h_attack: Decision threshold for positive CUSUM (Attack Alarm).
            h_benign: Decision threshold for negative CUSUM (Benign Confirmation).
            drift: Reference drift parameter (allowance / reference value).
            horizons: Evaluation horizons grid (default: [5, 10, 15, 20, 25, 30]).
            min_horizon: Minimum horizon before early stopping is permitted.
            max_horizon: Maximum horizon where decision is forced.
            eps: Numerical stability constant for log odds computation.
        """
        self.h_attack = h_attack
        self.h_benign = h_benign
        self.drift = drift
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        self.horizons = horizons or [5, 10, 15, 20, 25, 30]
        self.eps = eps

    def compute_log_odds(self, prob: float) -> float:
        """Converts probability p in [0, 1] to log-odds score."""
        p_clamped = np.clip(prob, self.eps, 1.0 - self.eps)
        return float(np.log(p_clamped / (1.0 - p_clamped)))

    def evaluate_sequence(
        self,
        step_probabilities: np.ndarray,
    ) -> AdaptiveDecision:
        """
        Runs sequential CUSUM evidence accumulation over a single sequence.

        Args:
            step_probabilities: Array of shape (T,) with P(Attack) at each step.

        Returns:
            AdaptiveDecision with CUSUM stopping horizon and prediction.
        """
        s_pos = 0.0
        s_neg = 0.0
        prob_history: Dict[int, float] = {}

        for k in self.horizons:
            if k > len(step_probabilities):
                break

            p_k = float(step_probabilities[k - 1])
            prob_history[k] = p_k

            # Compute instantaneous log-likelihood ratio / log-odds increment
            log_odds = self.compute_log_odds(p_k)
            score_atk = log_odds - self.drift
            score_ben = -log_odds - self.drift

            # Two-sided CUSUM recursion
            s_pos = max(0.0, s_pos + score_atk)
            s_neg = max(0.0, s_neg + score_ben)

            # Check CUSUM stopping criteria if past min_horizon
            if k >= self.min_horizon:
                if s_pos >= self.h_attack:
                    return AdaptiveDecision(
                        predicted_label=1,
                        probability_attack=p_k,
                        stopping_horizon=k,
                        is_early_stop=(k < self.max_horizon),
                        step_probabilities=prob_history,
                        stopping_reason=f"CUSUM_ATTACK_ALARM (S+={s_pos:.2f} >= {self.h_attack})",
                    )

                if s_neg >= self.h_benign:
                    return AdaptiveDecision(
                        predicted_label=0,
                        probability_attack=p_k,
                        stopping_horizon=k,
                        is_early_stop=(k < self.max_horizon),
                        step_probabilities=prob_history,
                        stopping_reason=f"CUSUM_BENIGN_ALARM (S-={s_neg:.2f} >= {self.h_benign})",
                    )

        # Forced stopping at max_horizon
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
            stopping_reason=f"CUSUM_FORCED_HORIZON (S+={s_pos:.2f}, S-={s_neg:.2f})",
        )

    def evaluate_batch(
        self,
        batch_step_probabilities: np.ndarray,
        ground_truth_labels: np.ndarray,
    ) -> Tuple[List[AdaptiveDecision], AdaptiveEvaluationResult]:
        """Evaluates batch of sequences using CUSUM detector."""
        decisions: List[AdaptiveDecision] = []
        preds: List[int] = []
        stopping_horizons: List[int] = []
        horizon_counts: Dict[int, int] = {k: 0 for k in self.horizons}

        for i in range(len(batch_step_probabilities)):
            dec = self.evaluate_sequence(batch_step_probabilities[i])
            decisions.append(dec)
            preds.append(dec.predicted_label)
            stopping_horizons.append(dec.stopping_horizon)
            if dec.stopping_horizon in horizon_counts:
                horizon_counts[dec.stopping_horizon] += 1
            else:
                horizon_counts[dec.stopping_horizon] = 1

        y_true = np.asarray(ground_truth_labels, dtype=int)
        y_pred = np.asarray(preds, dtype=int)

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
