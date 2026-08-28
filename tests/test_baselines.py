"""
Unit tests for DeepDNS baseline models and evaluation infrastructure.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.data.schema import CAUSAL_FEATURE_NAMES, FORBIDDEN_MODEL_COLUMNS
from src.models.rule_based import RuleBasedDetector
from src.models.random_forest import RandomForestBaseline
from src.models.mlp_baseline import MLPNetwork, MLPClassifierWrapper
from src.evaluation.metrics import compute_classification_metrics, EvaluationReport


@pytest.fixture
def toy_data():
    """Generates small synthetic training and test matrices."""
    np.random.seed(42)
    n_samples = 40
    n_features = 12

    # Benign samples (Class 0): lower entropy, shorter lengths
    X_benign = np.random.normal(loc=0.0, scale=1.0, size=(n_samples // 2, n_features))
    X_benign[:, 1] = np.random.uniform(10, 20, size=n_samples // 2)  # fqdn_length
    X_benign[:, 3] = np.random.uniform(1.5, 2.5, size=n_samples // 2) # entropy
    y_benign = np.zeros(n_samples // 2, dtype=int)

    # Attack samples (Class 1): higher entropy, longer lengths
    X_attack = np.random.normal(loc=1.0, scale=1.0, size=(n_samples // 2, n_features))
    X_attack[:, 1] = np.random.uniform(35, 60, size=n_samples // 2)  # fqdn_length
    X_attack[:, 3] = np.random.uniform(3.5, 4.8, size=n_samples // 2) # entropy
    y_attack = np.ones(n_samples // 2, dtype=int)

    X = np.vstack([X_benign, X_attack])
    y = np.concatenate([y_benign, y_attack])

    df = pd.DataFrame(X, columns=CAUSAL_FEATURE_NAMES)
    return df, y


# =========================================================
# 1. RULE-BASED DETECTOR TESTS
# =========================================================
def test_rule_based_predictions(toy_data):
    X_df, y = toy_data
    detector = RuleBasedDetector()
    detector.fit(X_df, y)

    preds = detector.predict(X_df)
    probas = detector.predict_proba(X_df)

    assert len(preds) == len(y)
    assert set(np.unique(preds)).issubset({0, 1})
    assert probas.shape == (len(y), 2)
    assert np.all((probas >= 0.0) & (probas <= 1.0))
    np.testing.assert_allclose(np.sum(probas, axis=1), 1.0, rtol=1e-5)


def test_rule_based_forbidden_columns_rejection(toy_data):
    X_df, _ = toy_data
    detector = RuleBasedDetector()
    # Inject forbidden column
    bad_df = X_df.copy()
    bad_df["sld"] = "tunnel.com"

    with pytest.raises(ValueError, match="Data Leakage Breach"):
        detector.predict(bad_df)


def test_rule_based_serialization(toy_data, tmp_path):
    X_df, y = toy_data
    detector = RuleBasedDetector(entropy_threshold=3.5)
    detector.fit(X_df, y)

    save_file = tmp_path / "rule_detector.json"
    detector.save(save_file)

    loaded = RuleBasedDetector.load(save_file)
    assert loaded.entropy_threshold == pytest.approx(detector.entropy_threshold)

    preds_1 = detector.predict(X_df)
    preds_2 = loaded.predict(X_df)
    np.testing.assert_array_equal(preds_1, preds_2)


# =========================================================
# 2. RANDOM FOREST BASELINE TESTS
# =========================================================
def test_random_forest_fit_predict(toy_data):
    X_df, y = toy_data
    rf = RandomForestBaseline(n_estimators=10, random_state=42)
    rf.fit(X_df, y)

    preds = rf.predict(X_df)
    probas = rf.predict_proba(X_df)
    importances = rf.get_feature_importances()

    assert len(preds) == len(y)
    assert probas.shape == (len(y), 2)
    assert np.all((probas >= 0.0) & (probas <= 1.0))
    np.testing.assert_allclose(np.sum(probas, axis=1), 1.0, rtol=1e-5)
    assert list(importances.keys()) == CAUSAL_FEATURE_NAMES


def test_random_forest_serialization(toy_data, tmp_path):
    X_df, y = toy_data
    rf = RandomForestBaseline(n_estimators=10, random_state=42)
    rf.fit(X_df, y)

    save_path = tmp_path / "rf_model.joblib"
    rf.save(save_path)

    loaded_rf = RandomForestBaseline.load(save_path)
    preds_1 = rf.predict(X_df)
    preds_2 = loaded_rf.predict(X_df)
    np.testing.assert_array_equal(preds_1, preds_2)


# =========================================================
# 3. MLP BASELINE TESTS
# =========================================================
def test_mlp_network_forward_shape():
    net = MLPNetwork(input_dim=12, hidden_dims=(32, 16), num_classes=2)
    x = torch.randn(8, 12)
    logits = net(x)
    assert logits.shape == (8, 2)
    assert not torch.isnan(logits).any()


def test_mlp_wrapper_fit_predict(toy_data):
    X_df, y = toy_data
    mlp = MLPClassifierWrapper(
        input_dim=12,
        hidden_dims=(32, 16),
        batch_size=8,
        random_seed=42,
        device="cpu",
    )
    mlp.fit(X_df, y, epochs=2)

    preds = mlp.predict(X_df)
    probas = mlp.predict_proba(X_df)

    assert len(preds) == len(y)
    assert probas.shape == (len(y), 2)
    assert np.all((probas >= 0.0) & (probas <= 1.0))
    np.testing.assert_allclose(np.sum(probas, axis=1), 1.0, rtol=1e-4)


def test_mlp_serialization(toy_data, tmp_path):
    X_df, y = toy_data
    mlp = MLPClassifierWrapper(input_dim=12, hidden_dims=(32, 16), device="cpu")
    mlp.fit(X_df, y, epochs=2)

    save_path = tmp_path / "mlp_test.pt"
    mlp.save(save_path)

    loaded_mlp = MLPClassifierWrapper.load(save_path, device="cpu")
    probas_1 = mlp.predict_proba(X_df)
    probas_2 = loaded_mlp.predict_proba(X_df)
    np.testing.assert_allclose(probas_1, probas_2, rtol=1e-4)


# =========================================================
# 4. EVALUATION METRICS TESTS
# =========================================================
def test_compute_classification_metrics():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.4, 0.7, 0.9, 0.95])

    rep = compute_classification_metrics(y_true, y_pred, y_prob, model_name="TestModel", dataset_split="test")

    assert rep.total_samples == 8
    # TP=3, FP=1, TN=3, FN=1
    assert rep.tp == 3
    assert rep.fp == 1
    assert rep.tn == 3
    assert rep.fn == 1

    assert rep.accuracy == pytest.approx(6 / 8)
    assert rep.precision == pytest.approx(3 / 4)
    assert rep.recall == pytest.approx(3 / 4)
    assert rep.fpr == pytest.approx(1 / 4)
    assert rep.fnr == pytest.approx(1 / 4)
    assert rep.roc_auc is not None
    assert 0.0 <= rep.roc_auc <= 1.0
    assert rep.pr_auc is not None

    table_str = rep.summary_table()
    assert "TESTMODEL" in table_str.upper()
    assert "FPR" in table_str
