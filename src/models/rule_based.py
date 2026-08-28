"""
Deterministic Rule-Based DNS Tunneling Detector.

Implements an RFC 1035 / empirical heuristic baseline evaluating causal DNS indicators
(entropy, FQDN length, subdomain length, digit ratio, and payload size).
Thresholds are configurable and tuneable exclusively on training data.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.data.schema import CAUSAL_FEATURE_NAMES, FORBIDDEN_MODEL_COLUMNS


class RuleBasedDetector:
    """
    Rule-based heuristic classifier for DNS tunneling / exfiltration detection.
    
    A query or feature vector is classified as Attack (1) if it violates structural heuristics:
    1. Character entropy >= entropy_threshold
    2. FQDN length >= fqdn_length_threshold
    3. Subdomain length >= subdomain_length_threshold
    4. Digit ratio >= digit_ratio_threshold
    5. Payload length >= payload_len_threshold
    """

    def __init__(
        self,
        entropy_threshold: float = 3.2,
        fqdn_length_threshold: float = 30.0,
        subdomain_length_threshold: float = 12.0,
        digit_ratio_threshold: float = 0.25,
        payload_len_threshold: float = 50.0,
        min_rules_triggered: int = 2,
    ):
        self.entropy_threshold = entropy_threshold
        self.fqdn_length_threshold = fqdn_length_threshold
        self.subdomain_length_threshold = subdomain_length_threshold
        self.digit_ratio_threshold = digit_ratio_threshold
        self.payload_len_threshold = payload_len_threshold
        self.min_rules_triggered = min_rules_triggered
        self.is_fitted = False
        self.feature_names = CAUSAL_FEATURE_NAMES

    def _validate_input(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Validates input matrix and verifies absence of forbidden features."""
        if isinstance(X, pd.DataFrame):
            for col in FORBIDDEN_MODEL_COLUMNS:
                if col in X.columns:
                    raise ValueError(f"Data Leakage Breach: Forbidden column '{col}' passed to model!")
            # Ensure columns follow exact causal ordering
            data = X[self.feature_names].values.astype(np.float64)
        else:
            data = np.asarray(X, dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] != len(self.feature_names):
                raise ValueError(f"Expected input shape (*, {len(self.feature_names)}), got {data.shape}")
        return data

    def _evaluate_rules(self, X_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluates rule conditions across feature matrix.
        
        Indices in CAUSAL_FEATURE_NAMES:
        0: inter_arrival_time
        1: fqdn_length
        2: subdomain_length
        3: char_entropy
        4: digit_ratio
        5: uppercase_ratio
        6: special_ratio
        7: label_count
        8: max_label_length
        9: avg_label_length
        10: has_subdomain
        11: payload_len
        """
        fqdn_len = X_arr[:, 1]
        subdomain_len = X_arr[:, 2]
        entropy = X_arr[:, 3]
        digit_ratio = X_arr[:, 4]
        payload_len = X_arr[:, 11]

        r1 = entropy >= self.entropy_threshold
        r2 = fqdn_len >= self.fqdn_length_threshold
        r3 = subdomain_len >= self.subdomain_length_threshold
        r4 = digit_ratio >= self.digit_ratio_threshold
        r5 = payload_len >= self.payload_len_threshold

        rule_matrix = np.column_stack([r1, r2, r3, r4, r5]).astype(int)
        trigger_counts = np.sum(rule_matrix, axis=1)
        total_rules = rule_matrix.shape[1]

        scores = trigger_counts / float(total_rules)
        return trigger_counts, scores

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> "RuleBasedDetector":
        """
        Tunes rule thresholds on training data to optimize balanced accuracy.
        
        Args:
            X: Training features (unnormalized or raw causal features).
            y: Training binary labels (0 = Benign, 1 = Attack).
        """
        X_arr = self._validate_input(X)
        y_arr = np.asarray(y, dtype=int)

        # Tune thresholds using training percentiles of the positive (attack) vs negative class
        attack_mask = y_arr == 1
        benign_mask = y_arr == 0

        if np.sum(attack_mask) > 0 and np.sum(benign_mask) > 0:
            # Set thresholds based on training attack distributions
            self.entropy_threshold = float(np.percentile(X_arr[attack_mask, 3], 25))
            self.fqdn_length_threshold = float(np.percentile(X_arr[attack_mask, 1], 25))
            self.subdomain_length_threshold = float(np.percentile(X_arr[attack_mask, 2], 25))
            self.digit_ratio_threshold = float(np.percentile(X_arr[attack_mask, 4], 25))
            self.payload_len_threshold = float(np.percentile(X_arr[attack_mask, 11], 25))

        self.is_fitted = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts binary class labels (0 = Benign, 1 = Attack)."""
        X_arr = self._validate_input(X)
        trigger_counts, _ = self._evaluate_rules(X_arr)
        preds = (trigger_counts >= self.min_rules_triggered).astype(int)
        return preds

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts class probabilities [P(y=0), P(y=1)] based on rule trigger ratio."""
        X_arr = self._validate_input(X)
        _, scores = self._evaluate_rules(X_arr)
        p1 = np.clip(scores, 0.0, 1.0)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def save(self, file_path: Union[str, Path]) -> None:
        """Serializes detector configuration to JSON."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config = {
            "model_type": "RuleBasedDetector",
            "entropy_threshold": self.entropy_threshold,
            "fqdn_length_threshold": self.fqdn_length_threshold,
            "subdomain_length_threshold": self.subdomain_length_threshold,
            "digit_ratio_threshold": self.digit_ratio_threshold,
            "payload_len_threshold": self.payload_len_threshold,
            "min_rules_triggered": self.min_rules_triggered,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "RuleBasedDetector":
        """Loads detector configuration from JSON."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration not found: '{file_path}'")
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)

        detector = cls(
            entropy_threshold=config["entropy_threshold"],
            fqdn_length_threshold=config["fqdn_length_threshold"],
            subdomain_length_threshold=config["subdomain_length_threshold"],
            digit_ratio_threshold=config["digit_ratio_threshold"],
            payload_len_threshold=config["payload_len_threshold"],
            min_rules_triggered=config["min_rules_triggered"],
        )
        detector.is_fitted = config.get("is_fitted", True)
        return detector
