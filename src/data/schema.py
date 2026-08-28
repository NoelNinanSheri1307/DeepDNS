"""
Schema definitions and validation utilities for DeepDNS.

Defines the expected column specifications, data types, and structural constraints
for CIC-Bell Stateless CSV files and DNS Threats domain datasets.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np


# The exact 15 columns present in raw CIC-Bell stateless CSV files
CIC_BELL_STATELESS_COLUMNS = [
    "timestamp",
    "FQDN_count",
    "subdomain_length",
    "upper",
    "lower",
    "numeric",
    "entropy",
    "special",
    "labels",
    "labels_max",
    "labels_average",
    "longest_word",
    "sld",
    "len",
    "subdomain",
]

# Explicitly forbidden columns from model input tensors to prevent target leakage / artifacts
FORBIDDEN_MODEL_COLUMNS = [
    "timestamp",
    "sld",
    "longest_word",
    "src_file",
    "pair_id",
    "capture_id",
    "label",
    "label_name",
    "label_modality",
    "intensity",
]

# The approved 12 causal features for DeepDNS
CAUSAL_FEATURE_NAMES = [
    "inter_arrival_time",
    "fqdn_length",
    "subdomain_length",
    "char_entropy",
    "digit_ratio",
    "uppercase_ratio",
    "special_ratio",
    "label_count",
    "max_label_length",
    "avg_label_length",
    "has_subdomain",
    "payload_len",
]

# Expected columns in DNS Threats dataset
DNS_THREATS_COLUMNS = ["domain", "class"]


@dataclass
class ValidationResult:
    """Result of schema validation on a DataFrame or file."""
    is_valid: bool
    total_rows: int
    missing_columns: List[str]
    unexpected_columns: List[str]
    null_counts: Dict[str, int]
    nan_counts: Dict[str, int]
    inf_counts: Dict[str, int]
    non_numeric_columns: List[str]
    timestamp_parseable: bool
    is_chronological: bool
    anomalies: List[str]

    def to_dict(self) -> dict:
        return {
            "is_valid": self.is_valid,
            "total_rows": self.total_rows,
            "missing_columns": self.missing_columns,
            "unexpected_columns": self.unexpected_columns,
            "null_counts": self.null_counts,
            "nan_counts": self.nan_counts,
            "inf_counts": self.inf_counts,
            "non_numeric_columns": self.non_numeric_columns,
            "timestamp_parseable": self.timestamp_parseable,
            "is_chronological": self.is_chronological,
            "anomalies": self.anomalies,
        }


def validate_stateless_dataframe(df: pd.DataFrame, source_name: str = "") -> ValidationResult:
    """
    Validates a CIC-Bell stateless DataFrame against the required schema.

    Args:
        df: Input DataFrame to validate.
        source_name: Optional file or stream identifier for logging.

    Returns:
        ValidationResult object containing detailed structural metrics.
    """
    cols = list(df.columns)
    missing = [c for c in CIC_BELL_STATELESS_COLUMNS if c not in cols]
    unexpected = [c for c in cols if c not in CIC_BELL_STATELESS_COLUMNS]
    
    null_counts = {c: int(df[c].isnull().sum()) for c in cols}
    nan_counts = {c: int(df[c].isna().sum()) for c in cols}
    
    inf_counts = {}
    non_numeric = []
    anomalies = []

    # Check numeric columns
    numeric_check_cols = [
        "FQDN_count", "subdomain_length", "upper", "lower", "numeric",
        "entropy", "special", "labels", "labels_max", "labels_average",
        "len", "subdomain"
    ]
    
    for c in numeric_check_cols:
        if c in df.columns:
            ser_num = pd.to_numeric(df[c], errors="coerce")
            num_nans = int(ser_num.isnull().sum() - df[c].isnull().sum())
            if num_nans > 0:
                non_numeric.append(c)
                anomalies.append(f"Column '{c}' contains {num_nans} non-coercible non-numeric values.")
            
            # Check infinities
            inf_count = int(np.isinf(ser_num.dropna()).sum())
            inf_counts[c] = inf_count
            if inf_count > 0:
                anomalies.append(f"Column '{c}' contains {inf_count} infinite values.")

    # Validate timestamps
    ts_parseable = False
    is_chronological = False
    if "timestamp" in df.columns:
        try:
            ts_dt = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")
            ts_parseable = bool(ts_dt.notnull().all())
            if ts_parseable:
                is_chronological = bool(ts_dt.is_monotonic_increasing)
                if not is_chronological:
                    anomalies.append(f"Timestamps in '{source_name}' are not strictly sorted.")
            else:
                anomalies.append(f"Column 'timestamp' contains {ts_dt.isnull().sum()} invalid timestamp entries.")
        except Exception as e:
            anomalies.append(f"Timestamp parsing failed: {str(e)}")

    is_valid = (
        len(missing) == 0
        and len(non_numeric) == 0
        and ts_parseable
        and sum(inf_counts.values()) == 0
    )

    return ValidationResult(
        is_valid=is_valid,
        total_rows=len(df),
        missing_columns=missing,
        unexpected_columns=unexpected,
        null_counts=null_counts,
        nan_counts=nan_counts,
        inf_counts=inf_counts,
        non_numeric_columns=non_numeric,
        timestamp_parseable=ts_parseable,
        is_chronological=is_chronological,
        anomalies=anomalies,
    )
