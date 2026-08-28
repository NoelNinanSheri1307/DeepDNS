"""
Causal feature extraction and robust normalization for DeepDNS.

Implements the verified 12-feature causal behavioral/lexical representation
and provides a serializable scaler that avoids data leakage.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.data.schema import CAUSAL_FEATURE_NAMES, FORBIDDEN_MODEL_COLUMNS


class CausalFeatureExtractor:
    """
    Extracts approved causal features from raw CIC-Bell stateless DataFrames.
    
    Guarantees strict causal ordering, excludes non-causal or leakage-prone columns,
    and supports controlled feature ablation.
    """

    def __init__(self, epsilon: float = 1e-6, exclude_features: Optional[List[str]] = None):
        self.epsilon = epsilon
        self.exclude_features = exclude_features or []
        self.feature_names = [f for f in CAUSAL_FEATURE_NAMES if f not in self.exclude_features]

    def extract_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms a raw stateless DataFrame into the 12 causal features.
        
        Args:
            df: Raw stateless DataFrame containing timestamp and base features.

        Returns:
            pd.DataFrame with exactly 12 causal numerical columns, sorted chronologically.
        """
        # Ensure chronological ordering per capture session
        if "timestamp" not in df.columns:
            raise ValueError("Required 'timestamp' column missing from stateless input.")

        df_sorted = df.copy()
        df_sorted["_parsed_ts"] = pd.to_datetime(df_sorted["timestamp"], format="mixed", errors="coerce")
        if df_sorted["_parsed_ts"].isnull().any():
            raise ValueError("Stateless DataFrame contains invalid or unparseable timestamps.")

        # Sort strictly by timestamp
        df_sorted = df_sorted.sort_values(by="_parsed_ts", ascending=True).reset_index(drop=True)

        # 1. Causal Inter-arrival time: delta_t = t_i - t_(i-1)
        # First observation deterministic delta_t = 0.0
        delta_t = df_sorted["_parsed_ts"].diff().dt.total_seconds().fillna(0.0).values
        # Enforce causality (delta_t >= 0.0)
        delta_t = np.maximum(delta_t, 0.0)
        inter_arrival_time = np.log1p(delta_t)

        # Base features with numeric coercion
        fqdn_count = pd.to_numeric(df_sorted["FQDN_count"], errors="coerce").fillna(0.0).values
        subdomain_len = pd.to_numeric(df_sorted["subdomain_length"], errors="coerce").fillna(0.0).values
        entropy = pd.to_numeric(df_sorted["entropy"], errors="coerce").fillna(0.0).values
        numeric = pd.to_numeric(df_sorted["numeric"], errors="coerce").fillna(0.0).values
        upper = pd.to_numeric(df_sorted["upper"], errors="coerce").fillna(0.0).values
        special = pd.to_numeric(df_sorted["special"], errors="coerce").fillna(0.0).values
        labels = pd.to_numeric(df_sorted["labels"], errors="coerce").fillna(0.0).values
        labels_max = pd.to_numeric(df_sorted["labels_max"], errors="coerce").fillna(0.0).values
        labels_avg = pd.to_numeric(df_sorted["labels_average"], errors="coerce").fillna(0.0).values
        subdomain_flag = pd.to_numeric(df_sorted["subdomain"], errors="coerce").fillna(0.0).values
        payload_len = pd.to_numeric(df_sorted["len"], errors="coerce").fillna(0.0).values

        # Derived ratios with epsilon safeguard
        denom = fqdn_count + self.epsilon
        digit_ratio = numeric / denom
        uppercase_ratio = upper / denom
        special_ratio = special / denom

        # Assemble causal feature DataFrame
        features_dict = {
            "inter_arrival_time": inter_arrival_time,
            "fqdn_length": fqdn_count,
            "subdomain_length": subdomain_len,
            "char_entropy": entropy,
            "digit_ratio": digit_ratio,
            "uppercase_ratio": uppercase_ratio,
            "special_ratio": special_ratio,
            "label_count": labels,
            "max_label_length": labels_max,
            "avg_label_length": labels_avg,
            "has_subdomain": subdomain_flag,
            "payload_len": payload_len,
        }

        feature_df = pd.DataFrame(features_dict, columns=self.feature_names)

        # Verification: ensure no forbidden columns are present
        for forbidden in FORBIDDEN_MODEL_COLUMNS:
            if forbidden in feature_df.columns:
                raise RuntimeError(f"Data Leakage Breach: Forbidden column '{forbidden}' found in feature matrix!")

        return feature_df


class FeatureScaler:
    """
    Serializable StandardScaler with optional robust quantile clipping.
    
    Must be fit ONLY on training data partitions and applied identically to val/test.
    """

    def __init__(
        self,
        clip_outliers: bool = True,
        clip_std: float = 8.0,
        feature_names: Optional[List[str]] = None,
    ):
        self.clip_outliers = clip_outliers
        self.clip_std = clip_std
        self.feature_names: List[str] = feature_names or CAUSAL_FEATURE_NAMES
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "FeatureScaler":
        """
        Fits mean and standard deviation from training data.
        """
        if isinstance(X, pd.DataFrame):
            data = X[self.feature_names].values.astype(np.float64)
        else:
            data = np.asarray(X, dtype=np.float64)

        if data.ndim != 2 or data.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected input shape (N, {len(self.feature_names)}), got {data.shape}")

        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        # Prevent division by zero for constant features
        std = np.where(std < 1e-8, 1.0, std)

        self.mean_ = mean
        self.scale_ = std
        self.is_fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Standardizes feature matrix using fitted statistics.
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureScaler must be fitted before transforming data.")

        if isinstance(X, pd.DataFrame):
            data = X[self.feature_names].values.astype(np.float64)
        else:
            data = np.asarray(X, dtype=np.float64)

        normalized = (data - self.mean_) / self.scale_

        if self.clip_outliers:
            normalized = np.clip(normalized, -self.clip_std, self.clip_std)

        return normalized.astype(np.float32)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Fits to data and transforms in a single call."""
        return self.fit(X).transform(X)

    def to_json(self, file_path: Union[str, Path]) -> None:
        """Serializes scaler statistics to a JSON file."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted FeatureScaler.")

        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "feature_names": self.feature_names,
            "mean": self.mean_.tolist(),
            "scale": self.scale_.tolist(),
            "clip_outliers": self.clip_outliers,
            "clip_std": self.clip_std,
            "is_fitted": self.is_fitted,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def from_json(cls, file_path: Union[str, Path]) -> "FeatureScaler":
        """Loads a serialized FeatureScaler from a JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Scaler file not found: '{file_path}'")

        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        scaler = cls(
            clip_outliers=payload.get("clip_outliers", True),
            clip_std=payload.get("clip_std", 8.0),
        )
        scaler.feature_names = payload["feature_names"]
        scaler.mean_ = np.array(payload["mean"], dtype=np.float64)
        scaler.scale_ = np.array(payload["scale"], dtype=np.float64)
        scaler.is_fitted = payload["is_fitted"]
        return scaler
