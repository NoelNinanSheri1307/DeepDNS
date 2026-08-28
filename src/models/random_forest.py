"""
Random Forest Baseline Classifier for DeepDNS.

Implements a scikit-learn RandomForestClassifier utilizing the 12 approved causal features,
with class balancing, feature importance extraction, and reproducible serialization.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.data.schema import CAUSAL_FEATURE_NAMES, FORBIDDEN_MODEL_COLUMNS


class RandomForestBaseline:
    """
    Random Forest baseline model for DNS tunneling / exfiltration detection.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 16,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        class_weight: str = "balanced",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.feature_names = CAUSAL_FEATURE_NAMES

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.is_fitted = False

    def _validate_input(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Validates input matrix and verifies absence of forbidden features."""
        if isinstance(X, pd.DataFrame):
            for col in FORBIDDEN_MODEL_COLUMNS:
                if col in X.columns:
                    raise ValueError(f"Data Leakage Breach: Forbidden column '{col}' passed to model!")
            data = X[self.feature_names].values.astype(np.float64)
        else:
            data = np.asarray(X, dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            if data.shape[1] != len(self.feature_names):
                raise ValueError(f"Expected input shape (*, {len(self.feature_names)}), got {data.shape}")
        return data

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> "RandomForestBaseline":
        """Fits Random Forest model on training features and labels."""
        X_arr = self._validate_input(X)
        y_arr = np.asarray(y, dtype=int)
        self.model.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts binary class labels (0 or 1)."""
        if not self.is_fitted:
            raise RuntimeError("RandomForestBaseline must be fitted before predicting.")
        X_arr = self._validate_input(X)
        return self.model.predict(X_arr)

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predicts class probabilities shape (N, 2)."""
        if not self.is_fitted:
            raise RuntimeError("RandomForestBaseline must be fitted before predicting.")
        X_arr = self._validate_input(X)
        return self.model.predict_proba(X_arr)

    def get_feature_importances(self) -> Dict[str, float]:
        """Returns feature importances mapped to feature names."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted to obtain feature importances.")
        importances = self.model.feature_importances_
        return {
            name: round(float(imp), 6)
            for name, imp in zip(self.feature_names, importances)
        }

    def save(self, file_path: Union[str, Path]) -> None:
        """Serializes model to disk via joblib."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model.")
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "is_fitted": self.is_fitted,
                "feature_names": self.feature_names,
                "config": {
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "min_samples_split": self.min_samples_split,
                    "min_samples_leaf": self.min_samples_leaf,
                    "class_weight": self.class_weight,
                    "random_state": self.random_state,
                },
            },
            path,
        )

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "RandomForestBaseline":
        """Loads serialized model from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: '{file_path}'")
        data = joblib.load(path)
        cfg = data["config"]
        baseline = cls(
            n_estimators=cfg["n_estimators"],
            max_depth=cfg["max_depth"],
            min_samples_split=cfg["min_samples_split"],
            min_samples_leaf=cfg["min_samples_leaf"],
            class_weight=cfg["class_weight"],
            random_state=cfg["random_state"],
        )
        baseline.model = data["model"]
        baseline.is_fitted = data["is_fitted"]
        baseline.feature_names = data["feature_names"]
        return baseline
