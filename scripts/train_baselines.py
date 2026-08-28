"""
DeepDNS Baseline Training and Evaluation Script.

Trains and evaluates:
1. Rule-Based Detector
2. Random Forest Classifier
3. Lightweight MLP Classifier

All baselines are trained exclusively on the Train partition from split_manifest.json
and evaluated on Validation and Test partitions using security-oriented metrics.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.labels import parse_cic_bell_label
from src.data.features import CausalFeatureExtractor, FeatureScaler
from src.models.rule_based import RuleBasedDetector
from src.models.random_forest import RandomForestBaseline
from src.models.mlp_baseline import MLPClassifierWrapper
from src.evaluation.metrics import compute_classification_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TrainBaselines")


def load_partition_data(
    captures_info: list,
    extractor: CausalFeatureExtractor,
    scaler: Optional[FeatureScaler] = None,
    max_rows_per_capture: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads raw capture CSVs, extracts causal features, applies normalization,
    and returns feature arrays, binary labels, and raw feature arrays.
    """
    feature_list = []
    raw_feature_list = []
    label_list = []

    for cap in captures_info:
        p = PROJECT_ROOT / cap["rel_path"]
        df_raw = pd.read_csv(p, low_memory=False, nrows=max_rows_per_capture)
        meta = parse_cic_bell_label(p)

        feat_df = extractor.extract_from_dataframe(df_raw)
        raw_feature_list.append(feat_df.values)

        if scaler is not None:
            norm_feats = scaler.transform(feat_df)
        else:
            norm_feats = feat_df.values

        feature_list.append(norm_feats)
        labels = np.full(len(feat_df), meta.label, dtype=np.int64)
        label_list.append(labels)

    X_norm = np.concatenate(feature_list, axis=0)
    X_raw = np.concatenate(raw_feature_list, axis=0)
    y = np.concatenate(label_list, axis=0)

    return X_norm, y, X_raw


def run_baseline_training(
    max_train_samples: Optional[int] = None,
    rf_trees: int = 100,
    mlp_epochs: int = 10,
    mlp_batch_size: int = 256,
    mlp_lr: float = 1e-3,
    save_models: bool = True,
):
    logger.info("=========================================================")
    logger.info("STARTING DEEPDNS BASELINE TRAINING PIPELINE")
    logger.info("=========================================================")

    models_dir = PROJECT_ROOT / "data" / "processed" / "models"
    reports_dir = PROJECT_ROOT / "reports" / "model_baselines"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Split Manifest & Scaler
    manifest_path = PROJECT_ROOT / "reports" / "data_pipeline" / "split_manifest.json"
    scaler_path = PROJECT_ROOT / "data" / "processed" / "feature_scaler.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found at: {manifest_path}. Run scripts/dataset_loader.py first.")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Feature scaler not found at: {scaler_path}. Run scripts/dataset_loader.py first.")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    scaler = FeatureScaler.from_json(scaler_path)
    extractor = CausalFeatureExtractor()

    # 2. Ingest Partitions
    logger.info("Ingesting Training partition (12 captures)...")
    X_train_norm, y_train, X_train_raw = load_partition_data(
        manifest_data["train_captures"],
        extractor=extractor,
        scaler=scaler,
        max_rows_per_capture=max_train_samples,
    )
    logger.info(f"Train samples: {len(X_train_norm):,} (Positive rate: {np.mean(y_train)*100:.2f}%)")

    logger.info("Ingesting Validation partition (3 captures)...")
    X_val_norm, y_val, X_val_raw = load_partition_data(
        manifest_data["val_captures"],
        extractor=extractor,
        scaler=scaler,
    )
    logger.info(f"Val samples: {len(X_val_norm):,} (Positive rate: {np.mean(y_val)*100:.2f}%)")

    logger.info("Ingesting Test partition (3 captures)...")
    X_test_norm, y_test, X_test_raw = load_partition_data(
        manifest_data["test_captures"],
        extractor=extractor,
        scaler=scaler,
    )
    logger.info(f"Test samples: {len(X_test_norm):,} (Positive rate: {np.mean(y_test)*100:.2f}%)")

    results = {}

    # ---------------------------------------------------------
    # BASELINE 1: Rule-Based Detector
    # ---------------------------------------------------------
    logger.info("\n--- Training & Evaluating Baseline 1: Rule-Based Detector ---")
    rule_model = RuleBasedDetector()
    rule_model.fit(X_train_raw, y_train)

    val_preds_rule = rule_model.predict(X_val_raw)
    val_probs_rule = rule_model.predict_proba(X_val_raw)[:, 1]
    rule_val_rep = compute_classification_metrics(y_val, val_preds_rule, val_probs_rule, model_name="Rule_Based", dataset_split="val")
    print(rule_val_rep.summary_table())

    test_preds_rule = rule_model.predict(X_test_raw)
    test_probs_rule = rule_model.predict_proba(X_test_raw)[:, 1]
    rule_test_rep = compute_classification_metrics(y_test, test_preds_rule, test_probs_rule, model_name="Rule_Based", dataset_split="test")
    print(rule_test_rep.summary_table())

    if save_models:
        rule_model.save(models_dir / "rule_based_detector.json")

    results["rule_based"] = {
        "val": rule_val_rep.to_dict(),
        "test": rule_test_rep.to_dict(),
    }

    # ---------------------------------------------------------
    # BASELINE 2: Random Forest Classifier
    # ---------------------------------------------------------
    logger.info(f"\n--- Training & Evaluating Baseline 2: Random Forest ({rf_trees} trees) ---")
    rf_model = RandomForestBaseline(n_estimators=rf_trees, max_depth=16, random_state=42)
    rf_model.fit(X_train_norm, y_train)

    val_preds_rf = rf_model.predict(X_val_norm)
    val_probs_rf = rf_model.predict_proba(X_val_norm)[:, 1]
    rf_val_rep = compute_classification_metrics(y_val, val_preds_rf, val_probs_rf, model_name="Random_Forest", dataset_split="val")
    print(rf_val_rep.summary_table())

    test_preds_rf = rf_model.predict(X_test_norm)
    test_probs_rf = rf_model.predict_proba(X_test_norm)[:, 1]
    rf_test_rep = compute_classification_metrics(y_test, test_preds_rf, test_probs_rf, model_name="Random_Forest", dataset_split="test")
    print(rf_test_rep.summary_table())

    logger.info(f"RF Feature Importances: {rf_model.get_feature_importances()}")

    if save_models:
        rf_model.save(models_dir / "random_forest_baseline.joblib")

    results["random_forest"] = {
        "val": rf_val_rep.to_dict(),
        "test": rf_test_rep.to_dict(),
        "feature_importances": rf_model.get_feature_importances(),
    }

    # ---------------------------------------------------------
    # BASELINE 3: Lightweight MLP Classifier
    # ---------------------------------------------------------
    logger.info(f"\n--- Training & Evaluating Baseline 3: MLP Classifier ({mlp_epochs} epochs) ---")
    mlp_model = MLPClassifierWrapper(
        input_dim=12,
        hidden_dims=(64, 32),
        dropout_rate=0.1,
        learning_rate=mlp_lr,
        batch_size=mlp_batch_size,
        random_seed=42,
    )
    logger.info(f"MLP Target Compute Device: {mlp_model.device}")
    mlp_model.fit(X_train_norm, y_train, epochs=mlp_epochs)

    val_preds_mlp = mlp_model.predict(X_val_norm)
    val_probs_mlp = mlp_model.predict_proba(X_val_norm)[:, 1]
    mlp_val_rep = compute_classification_metrics(y_val, val_preds_mlp, val_probs_mlp, model_name="MLP_Baseline", dataset_split="val")
    print(mlp_val_rep.summary_table())

    test_preds_mlp = mlp_model.predict(X_test_norm)
    test_probs_mlp = mlp_model.predict_proba(X_test_norm)[:, 1]
    mlp_test_rep = compute_classification_metrics(y_test, test_preds_mlp, test_probs_mlp, model_name="MLP_Baseline", dataset_split="test")
    print(mlp_test_rep.summary_table())

    if save_models:
        mlp_model.save(models_dir / "mlp_baseline.pt")

    results["mlp_baseline"] = {
        "val": mlp_val_rep.to_dict(),
        "test": mlp_test_rep.to_dict(),
    }

    # 3. Save combined evaluation report
    results_path = reports_dir / "baseline_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved all baseline evaluation metrics to: {results_path}")
    logger.info("=========================================================")
    logger.info("BASELINE TRAINING & EVALUATION COMPLETED SUCCESSFULLY")
    logger.info("=========================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate DeepDNS baseline models.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional limit for fast training.")
    parser.add_argument("--rf_trees", type=int, default=100, help="Number of trees in Random Forest.")
    parser.add_argument("--mlp_epochs", type=int, default=10, help="Number of training epochs for MLP.")
    parser.add_argument("--mlp_batch_size", type=int, default=512, help="Batch size for MLP.")
    parser.add_argument("--mlp_lr", type=float, default=1e-3, help="Learning rate for MLP.")
    args = parser.parse_args()

    run_baseline_training(
        max_train_samples=args.max_train_samples,
        rf_trees=args.rf_trees,
        mlp_epochs=args.mlp_epochs,
        mlp_batch_size=args.mlp_batch_size,
        mlp_lr=args.mlp_lr,
    )
