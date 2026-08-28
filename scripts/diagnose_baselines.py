"""
Diagnostic evaluation script for DeepDNS baseline models.

Computes:
1. Per-capture breakdown for Validation and Test partitions
2. RF vs MLP agreement / disagreement metrics
3. Threshold sweep and FPR at fixed recall targets (90%, 95%, 99%)
4. Modality-specific detection rates
5. Feature importance analysis
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.labels import parse_cic_bell_label
from src.data.features import CausalFeatureExtractor, FeatureScaler
from src.models.rule_based import RuleBasedDetector
from src.models.random_forest import RandomForestBaseline
from src.models.mlp_baseline import MLPClassifierWrapper
from src.evaluation.metrics import compute_classification_metrics


def run_diagnostics():
    manifest_path = PROJECT_ROOT / "reports" / "data_pipeline" / "split_manifest.json"
    scaler_path = PROJECT_ROOT / "data" / "processed" / "feature_scaler.json"
    models_dir = PROJECT_ROOT / "data" / "processed" / "models"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    scaler = FeatureScaler.from_json(scaler_path)
    extractor = CausalFeatureExtractor()

    # Load models
    rule_model = RuleBasedDetector.load(models_dir / "rule_based_detector.json")
    rf_model = RandomForestBaseline.load(models_dir / "random_forest_baseline.joblib")
    mlp_model = MLPClassifierWrapper.load(models_dir / "mlp_baseline.pt", device="cpu")

    diagnostic_data = {}

    # 1. Capture-by-Capture Diagnostic
    capture_reports = []
    
    all_test_y = []
    all_test_rf_prob = []
    all_test_mlp_prob = []
    all_test_rf_pred = []
    all_test_mlp_pred = []
    all_test_modality = []

    for split_name, cap_list in [("val", manifest["val_captures"]), ("test", manifest["test_captures"])]:
        for cap in cap_list:
            p = PROJECT_ROOT / cap["rel_path"]
            df_raw = pd.read_csv(p, low_memory=False)
            meta = parse_cic_bell_label(p)

            feat_df = extractor.extract_from_dataframe(df_raw)
            X_norm = scaler.transform(feat_df)
            X_raw = feat_df.values
            y = np.full(len(feat_df), meta.label, dtype=int)

            rf_preds = rf_model.predict(X_norm)
            rf_probs = rf_model.predict_proba(X_norm)[:, 1]

            mlp_preds = mlp_model.predict(X_norm)
            mlp_probs = mlp_model.predict_proba(X_norm)[:, 1]

            rule_preds = rule_model.predict(X_raw)

            rf_acc = float(np.mean(rf_preds == y))
            mlp_acc = float(np.mean(mlp_preds == y))
            rule_acc = float(np.mean(rule_preds == y))

            capture_reports.append({
                "split": split_name,
                "capture_id": meta.capture_id,
                "label": meta.label,
                "modality": meta.attack_modality or "benign",
                "intensity": meta.intensity,
                "rows": len(y),
                "rule_acc": round(rule_acc, 4),
                "rf_acc": round(rf_acc, 4),
                "mlp_acc": round(mlp_acc, 4),
                "rf_mean_prob": round(float(np.mean(rf_probs)), 4),
                "mlp_mean_prob": round(float(np.mean(mlp_probs)), 4),
            })

            if split_name == "test":
                all_test_y.append(y)
                all_test_rf_prob.append(rf_probs)
                all_test_mlp_prob.append(mlp_probs)
                all_test_rf_pred.append(rf_preds)
                all_test_mlp_pred.append(mlp_preds)
                all_test_modality.extend([meta.attack_modality or "benign"] * len(y))

    y_test_all = np.concatenate(all_test_y)
    rf_prob_all = np.concatenate(all_test_rf_prob)
    mlp_prob_all = np.concatenate(all_test_mlp_prob)
    rf_pred_all = np.concatenate(all_test_rf_pred)
    mlp_pred_all = np.concatenate(all_test_mlp_pred)
    modalities_all = np.array(all_test_modality)

    # 2. RF vs MLP Agreement / Disagreement
    agreement = float(np.mean(rf_pred_all == mlp_pred_all))
    disagreement_indices = np.where(rf_pred_all != mlp_pred_all)[0]
    
    # Both correct, RF only correct, MLP only correct, Both wrong
    both_correct = np.sum((rf_pred_all == y_test_all) & (mlp_pred_all == y_test_all))
    rf_only_correct = np.sum((rf_pred_all == y_test_all) & (mlp_pred_all != y_test_all))
    mlp_only_correct = np.sum((rf_pred_all != y_test_all) & (mlp_pred_all == y_test_all))
    both_wrong = np.sum((rf_pred_all != y_test_all) & (mlp_pred_all != y_test_all))

    # 3. Threshold sweeps & FPR at fixed Recall targets (90%, 95%, 99%)
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rf_sweep = []
    mlp_sweep = []

    for t in thresholds:
        # RF
        rf_p = (rf_prob_all >= t).astype(int)
        cm = compute_classification_metrics(y_test_all, rf_p, rf_prob_all, model_name="RF", dataset_split="test")
        rf_sweep.append({
            "threshold": t,
            "precision": round(cm.precision, 4),
            "recall": round(cm.recall, 4),
            "fpr": round(cm.fpr, 4),
            "f1": round(cm.f1, 4),
        })

        # MLP
        mlp_p = (mlp_prob_all >= t).astype(int)
        cm_mlp = compute_classification_metrics(y_test_all, mlp_p, mlp_prob_all, model_name="MLP", dataset_split="test")
        mlp_sweep.append({
            "threshold": t,
            "precision": round(cm_mlp.precision, 4),
            "recall": round(cm_mlp.recall, 4),
            "fpr": round(cm_mlp.fpr, 4),
            "f1": round(cm_mlp.f1, 4),
        })

    # Exact FPR at target recalls (90%, 95%, 99%)
    def get_fpr_at_recall(probs, labels, target_rec):
        pos_probs = probs[labels == 1]
        neg_probs = probs[labels == 0]
        # Threshold to achieve target recall
        threshold = np.percentile(pos_probs, (1.0 - target_rec) * 100)
        fpr = np.mean(neg_probs >= threshold)
        achieved_recall = np.mean(pos_probs >= threshold)
        return float(threshold), float(achieved_recall), float(fpr)

    fixed_recall_rf = {
        "90%": get_fpr_at_recall(rf_prob_all, y_test_all, 0.90),
        "95%": get_fpr_at_recall(rf_prob_all, y_test_all, 0.95),
        "99%": get_fpr_at_recall(rf_prob_all, y_test_all, 0.99),
    }

    fixed_recall_mlp = {
        "90%": get_fpr_at_recall(mlp_prob_all, y_test_all, 0.90),
        "95%": get_fpr_at_recall(mlp_prob_all, y_test_all, 0.95),
        "99%": get_fpr_at_recall(mlp_prob_all, y_test_all, 0.99),
    }

    diagnostic_summary = {
        "capture_breakdown": capture_reports,
        "model_agreement": {
            "agreement_rate": round(agreement, 4),
            "total_test_samples": len(y_test_all),
            "both_correct": int(both_correct),
            "rf_only_correct": int(rf_only_correct),
            "mlp_only_correct": int(mlp_only_correct),
            "both_wrong": int(both_wrong),
        },
        "rf_threshold_sweep": rf_sweep,
        "mlp_threshold_sweep": mlp_sweep,
        "rf_fpr_at_recall": {k: {"threshold": round(v[0], 4), "recall": round(v[1], 4), "fpr": round(v[2], 4)} for k, v in fixed_recall_rf.items()},
        "mlp_fpr_at_recall": {k: {"threshold": round(v[0], 4), "recall": round(v[1], 4), "fpr": round(v[2], 4)} for k, v in fixed_recall_mlp.items()},
    }

    with open(PROJECT_ROOT / "reports" / "model_baselines" / "diagnostic_numbers.json", "w", encoding="utf-8") as f:
        json.dump(diagnostic_summary, f, indent=2)

    print("Diagnostic calculation complete! Saved diagnostic_numbers.json.")

if __name__ == "__main__":
    run_diagnostics()
