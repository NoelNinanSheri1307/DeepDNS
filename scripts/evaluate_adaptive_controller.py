"""
Adaptive Evidence Controller & CUSUM Evaluation Orchestrator for DeepDNS.

Compares:
1. Fixed Horizons K in [5, 10, 15, 20, 25, 30]
2. Adaptive Confidence Thresholding (Dynamic Horizon Early-Stopping)
3. Sequential CUSUM Change-Point Detection

Evaluates on:
A. In-Distribution Standard Test Partition (multiview_both.pt)
B. Leave-One-Modality-Out OOD Test Partition (multiview_ood_both.pt)

Outputs:
  - reports/adaptive/adaptive_results_id.json
  - reports/adaptive/adaptive_results_ood.json
  - reports/adaptive/adaptive_controller_audit.json
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.labels import parse_cic_bell_label
from src.data.features import CausalFeatureExtractor, FeatureScaler
from src.data.sequences import SequenceBuilder
from src.models.char_cnn import CharacterTokenizer
from src.data.multiview_dataset import build_multiview_partition_dataset, multiview_collate_fn
from src.models.multiview_fusion import DeepDNSMultiViewClassifier
from src.inference.adaptive_controller import AdaptiveEvidenceController
from src.inference.cusum import SequentialCUSUMDetector
from src.evaluation.metrics import compute_classification_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("EvalAdaptiveController")


def extract_batch_step_probabilities(
    classifier: DeepDNSMultiViewClassifier,
    dataset,
    batch_size: int = 256,
    device: str = "cuda",
    max_k: int = 30,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Extracts step-by-step P(Attack) probabilities across time 1..max_k for all sequences."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=multiview_collate_fn)
    all_step_probs = []
    all_labels = []
    all_metas = []

    classifier.model.eval()
    with torch.no_grad():
        for beh_b, lex_b, lens_b, labels_b, metas_b in loader:
            beh_b = beh_b.to(device) if beh_b is not None else None
            lex_b = lex_b.to(device) if lex_b is not None else None
            step_logits, _, _, _, _ = classifier.model(beh_b, lex_b, mode="both")
            
            # Extract softmax probabilities for Attack class (index 1) across all steps
            step_probs = torch.softmax(step_logits[:, :max_k, :], dim=-1)[:, :, 1].cpu().numpy()

            all_step_probs.append(step_probs)
            all_labels.append(labels_b.numpy())
            all_metas.extend(metas_b)

    step_probs_matrix = np.vstack(all_step_probs)
    labels_vector = np.concatenate(all_labels)
    return step_probs_matrix, labels_vector, all_metas


def evaluate_partition_strategies(
    split_name: str,
    classifier: DeepDNSMultiViewClassifier,
    val_dataset,
    test_dataset,
    device: str = "cuda",
) -> Dict[str, Any]:
    logger.info(f"\n=========================================================")
    logger.info(f"EVALUATING ADAPTIVE CONTROLLER ON: {split_name.upper()}")
    logger.info(f"=========================================================")

    horizons = [5, 10, 15, 20, 25, 30]

    # 1. Extract Validation Probabilities for Zero-Test Calibration
    logger.info("Extracting validation step probabilities for calibration...")
    val_probs, val_labels, _ = extract_batch_step_probabilities(classifier, val_dataset, device=device)

    # 2. Extract Test Probabilities
    logger.info("Extracting test step probabilities...")
    test_probs, test_labels, test_metas = extract_batch_step_probabilities(classifier, test_dataset, device=device)
    total_test = len(test_labels)

    strategy_results = {}

    # Strategy A: Fixed Horizons K in [5, 10, 15, 20, 25, 30]
    logger.info("Evaluating Fixed Horizon Baselines...")
    for k in horizons:
        p_k = test_probs[:, k - 1]
        y_pred = (p_k >= 0.5).astype(int)
        rep = compute_classification_metrics(
            y_true=test_labels,
            y_pred=y_pred,
            y_prob=p_k,
            model_name=f"Fixed_K{k}",
            dataset_split=split_name,
        )
        strategy_results[f"Fixed_K{k}"] = {
            "strategy": f"Fixed Horizon K={k}",
            "accuracy": round(rep.accuracy * 100, 2),
            "precision": round(rep.precision * 100, 2),
            "recall": round(rep.recall * 100, 2),
            "f1_score": round(rep.f1, 4),
            "fpr": round(rep.fpr * 100, 4),
            "roc_auc": round(rep.roc_auc, 4) if rep.roc_auc is not None else None,
            "pr_auc": round(rep.pr_auc, 4) if rep.pr_auc is not None else None,
            "mean_stopping_horizon": float(k),
            "median_stopping_horizon": float(k),
            "early_decision_pct": 0.0 if k == 30 else 100.0,
            "confusion_matrix": {"TP": rep.tp, "FP": rep.fp, "TN": rep.tn, "FN": rep.fn},
        }

    # Strategy B: Adaptive Confidence Thresholding (Calibrated on Validation)
    logger.info("Calibrating Adaptive Evidence Controller on validation data...")
    calibrated_ctrl = AdaptiveEvidenceController.calibrate_from_validation(
        val_step_probabilities=val_probs,
        val_labels=val_labels,
        target_max_fpr=0.005,
        target_min_recall=0.98,
        horizons=horizons,
    )
    logger.info(f"Calibrated Thresholds: tau_attack = {calibrated_ctrl.tau_attack}, tau_benign = {calibrated_ctrl.tau_benign}")

    _, adapt_res = calibrated_ctrl.evaluate_batch(test_probs, test_labels)
    strategy_results["Adaptive_Confidence"] = {
        "strategy": "Adaptive Horizon Confidence Thresholding",
        "tau_attack": calibrated_ctrl.tau_attack,
        "tau_benign": calibrated_ctrl.tau_benign,
        "accuracy": adapt_res.accuracy,
        "precision": adapt_res.precision,
        "recall": adapt_res.recall,
        "f1_score": adapt_res.f1_score,
        "fpr": adapt_res.fpr,
        "mean_stopping_horizon": adapt_res.mean_stopping_horizon,
        "median_stopping_horizon": adapt_res.median_stopping_horizon,
        "early_decision_pct": adapt_res.early_decision_percentage,
        "horizon_decision_percentages": adapt_res.horizon_decision_percentages,
        "confusion_matrix": adapt_res.confusion_matrix,
    }

    # Strategy C: Sequential CUSUM Detector
    logger.info("Evaluating Sequential CUSUM Detector...")
    cusum_detector = SequentialCUSUMDetector(
        h_attack=4.0,
        h_benign=4.0,
        drift=0.0,
        horizons=horizons,
        min_horizon=5,
        max_horizon=30,
    )
    _, cusum_res = cusum_detector.evaluate_batch(test_probs, test_labels)
    strategy_results["Adaptive_CUSUM"] = {
        "strategy": "Sequential CUSUM Statistical Change Detector",
        "h_attack": cusum_detector.h_attack,
        "h_benign": cusum_detector.h_benign,
        "drift": cusum_detector.drift,
        "accuracy": cusum_res.accuracy,
        "precision": cusum_res.precision,
        "recall": cusum_res.recall,
        "f1_score": cusum_res.f1_score,
        "fpr": cusum_res.fpr,
        "mean_stopping_horizon": cusum_res.mean_stopping_horizon,
        "median_stopping_horizon": cusum_res.median_stopping_horizon,
        "early_decision_pct": cusum_res.early_decision_percentage,
        "horizon_decision_percentages": cusum_res.horizon_decision_percentages,
        "confusion_matrix": cusum_res.confusion_matrix,
    }

    # Print Summary Table
    print("\n" + "=" * 90)
    print(f"ADAPTIVE CONTROLLER COMPARISON TABLE ({split_name.upper()})")
    print("=" * 90)
    print(f"{'Strategy':<30} | {'Recall':<8} | {'FPR':<8} | {'F1-Score':<8} | {'Mean K':<8} | {'Early %':<8}")
    print("-" * 90)
    for k_name, res in strategy_results.items():
        print(f"{res['strategy']:<30} | {res['recall']:<7.2f}% | {res['fpr']:<7.4f}% | {res['f1_score']:<8.4f} | {res['mean_stopping_horizon']:<8.2f} | {res['early_decision_pct']:<7.1f}%")
    print("=" * 90)

    return strategy_results


def run_full_adaptive_evaluation():
    reports_dir = PROJECT_ROOT / "reports" / "adaptive"
    reports_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = CausalFeatureExtractor()
    tokenizer = CharacterTokenizer()
    seq_builder = SequenceBuilder(min_seq_len=5, max_seq_len=30, step_size=10)

    # -------------------------------------------------------------
    # 1. Evaluate on In-Distribution (Standard Split)
    # -------------------------------------------------------------
    std_manifest_path = PROJECT_ROOT / "reports" / "data_pipeline" / "split_manifest.json"
    with open(std_manifest_path, "r", encoding="utf-8") as f:
        std_manifest = json.load(f)

    std_scaler = FeatureScaler.from_json(PROJECT_ROOT / "data" / "processed" / "feature_scaler.json")
    std_model_path = PROJECT_ROOT / "data" / "processed" / "models" / "multiview_both.pt"

    if std_model_path.exists():
        logger.info("Loading In-Distribution Multi-View Model...")
        std_classifier = DeepDNSMultiViewClassifier.load(std_model_path, device=device)
        std_val_ds = build_multiview_partition_dataset(std_manifest["val_captures"], PROJECT_ROOT, extractor, std_scaler, tokenizer, seq_builder)
        std_test_ds = build_multiview_partition_dataset(std_manifest["test_captures"], PROJECT_ROOT, extractor, std_scaler, tokenizer, seq_builder)

        id_results = evaluate_partition_strategies(
            split_name="in_distribution",
            classifier=std_classifier,
            val_dataset=std_val_ds,
            test_dataset=std_test_ds,
            device=device,
        )
        with open(reports_dir / "adaptive_results_id.json", "w", encoding="utf-8") as f:
            json.dump(id_results, f, indent=2)

    # -------------------------------------------------------------
    # 2. Evaluate on Leave-One-Modality-Out OOD Split
    # -------------------------------------------------------------
    ood_manifest_path = PROJECT_ROOT / "reports" / "data_pipeline" / "ood_split_manifest.json"
    with open(ood_manifest_path, "r", encoding="utf-8") as f:
        ood_manifest = json.load(f)

    ood_scaler = FeatureScaler.from_json(PROJECT_ROOT / "data" / "processed" / "feature_scaler_lomo.json")
    ood_model_path = PROJECT_ROOT / "data" / "processed" / "models" / "multiview_ood_both.pt"

    if ood_model_path.exists():
        logger.info("\nLoading OOD Multi-View Model...")
        ood_classifier = DeepDNSMultiViewClassifier.load(ood_model_path, device=device)
        ood_val_ds = build_multiview_partition_dataset(ood_manifest["val_captures"], PROJECT_ROOT, extractor, ood_scaler, tokenizer, seq_builder)
        ood_test_ds = build_multiview_partition_dataset(ood_manifest["test_captures"], PROJECT_ROOT, extractor, ood_scaler, tokenizer, seq_builder)

        ood_results = evaluate_partition_strategies(
            split_name="ood_lomo",
            classifier=ood_classifier,
            val_dataset=ood_val_ds,
            test_dataset=ood_test_ds,
            device=device,
        )
        with open(reports_dir / "adaptive_results_ood.json", "w", encoding="utf-8") as f:
            json.dump(ood_results, f, indent=2)

    logger.info(f"\nSaved all adaptive evaluation artifacts to {reports_dir}")


if __name__ == "__main__":
    run_full_adaptive_evaluation()
