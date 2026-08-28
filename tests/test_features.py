"""
Unit tests for causal feature extraction and normalization.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.data.features import CausalFeatureExtractor, FeatureScaler
from src.data.schema import CAUSAL_FEATURE_NAMES, FORBIDDEN_MODEL_COLUMNS


@pytest.fixture
def synthetic_stateless_df():
    """Generates a small synthetic stateless DataFrame."""
    return pd.DataFrame({
        "timestamp": [
            "2021-01-01 10:00:00.000000",
            "2021-01-01 10:00:00.500000",
            "2021-01-01 10:00:02.000000",
            "2021-01-01 10:00:05.000000",
            "2021-01-01 10:00:05.100000",
        ],
        "FQDN_count": [20, 25, 30, 15, 40],
        "subdomain_length": [5, 10, 15, 0, 25],
        "upper": [0, 2, 0, 0, 4],
        "lower": [15, 18, 20, 12, 26],
        "numeric": [3, 3, 8, 1, 8],
        "entropy": [2.5, 3.1, 3.8, 1.9, 4.2],
        "special": [2, 2, 2, 2, 2],
        "labels": [3, 4, 4, 2, 5],
        "labels_max": [10, 12, 15, 8, 20],
        "labels_average": [6.0, 5.5, 7.0, 7.0, 7.5],
        "longest_word": ["2", "4", "3", "5", "2"],
        "sld": ["192", "google", "attacker", "internal", "tunnel"],
        "len": [32, 45, 60, 28, 70],
        "subdomain": [1, 1, 1, 0, 1],
    })


def test_causal_feature_extraction(synthetic_stateless_df):
    extractor = CausalFeatureExtractor()
    feat_df = extractor.extract_from_dataframe(synthetic_stateless_df)

    # Verify column count and names
    assert list(feat_df.columns) == CAUSAL_FEATURE_NAMES
    assert len(feat_df) == len(synthetic_stateless_df)

    # First observation delta_t must be 0, log1p(0) == 0.0
    assert feat_df["inter_arrival_time"].iloc[0] == pytest.approx(0.0)

    # Second observation: delta_t = 0.5s -> log1p(0.5)
    assert feat_df["inter_arrival_time"].iloc[1] == pytest.approx(np.log1p(0.5))

    # Third observation: delta_t = 1.5s -> log1p(1.5)
    assert feat_df["inter_arrival_time"].iloc[2] == pytest.approx(np.log1p(1.5))


def test_forbidden_features_excluded(synthetic_stateless_df):
    extractor = CausalFeatureExtractor()
    feat_df = extractor.extract_from_dataframe(synthetic_stateless_df)

    for col in FORBIDDEN_MODEL_COLUMNS:
        assert col not in feat_df.columns


def test_feature_scaler_serialization(synthetic_stateless_df, tmp_path):
    extractor = CausalFeatureExtractor()
    feat_df = extractor.extract_from_dataframe(synthetic_stateless_df)

    scaler = FeatureScaler()
    scaler.fit(feat_df)
    norm_1 = scaler.transform(feat_df)

    scaler_file = tmp_path / "test_scaler.json"
    scaler.to_json(scaler_file)

    loaded_scaler = FeatureScaler.from_json(scaler_file)
    norm_2 = loaded_scaler.transform(feat_df)

    np.testing.assert_allclose(norm_1, norm_2, rtol=1e-5)
    assert norm_1.shape == (5, 12)
