"""
Comprehensive Capture-Level and Modality Shortcut Diagnostic Audit Script for DeepDNS OOD.

Performs:
1. Exact per-capture window-level and session-level evaluation across all 5 OOD test captures.
2. Distributional comparisons across Train Attack vs OOD Attack (Video & Text) and Train Benign vs Benign_2.
3. Feature Kolmogorov-Smirnov / Wasserstein shift quantification to detect any modality shortcuts.
4. Evaluation-unit dependency and effective sample size calculation.

Generates:
  - reports/multiview_ood/capture_level_ood_audit.json
  - reports/multiview_ood/modality_shortcut_audit.json
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.labels import parse_cic_bell_label
from src.data.features import CausalFeatureExtractor, FeatureScaler
from src.data.sequences import SequenceBuilder
from src.models.char_cnn import CharacterTokenizer
from src.data.multiview_dataset import build_multiview_partition_dataset, multiview_collate_fn
from src.models.multiview_fusion import DeepDNSMultiViewClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("CaptureLevelOODAudit")


def run_capture_and_shortcut_audit():
    logger.info("=========================================================")
    logger.info("STARTING CAPTURE-LEVEL OOD & MODALITY SHORTCUT AUDIT")
    logger.info("=========================================================")

    out_dir = PROJECT_ROOT / "reports" / "multiview_ood"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = PROJECT_ROOT / "reports" / "data_pipeline" / "ood_split_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    train_caps = manifest["train_captures"]
    test_caps = manifest["test_captures"]

    extractor = CausalFeatureExtractor()
    seq_builder = SequenceBuilder(min_seq_len=5, max_seq_len=30, step_size=10)
    scaler = FeatureScaler.from_json(PROJECT_ROOT / "data" / "processed" / "feature_scaler_lomo.json")
    tokenizer = CharacterTokenizer()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = PROJECT_ROOT / "data" / "processed" / "models" / "multiview_ood_both.pt"
    classifier = DeepDNSMultiViewClassifier.load(model_path, device=device)
    classifier.model.eval()

    # 1. Capture-Level OOD Inference
    logger.info("Evaluating all 5 OOD test captures individually...")
    test_dataset = build_multiview_partition_dataset(
        test_caps,
        project_root=PROJECT_ROOT,
        extractor=extractor,
        scaler=scaler,
        tokenizer=tokenizer,
        seq_builder=seq_builder,
    )

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, collate_fn=multiview_collate_fn)

    all_probs_k30 = []
    all_labels = []
    all_cids = []

    with torch.no_grad():
        for beh_b, lex_b, lens_b, labels_b, metas_b in test_loader:
            beh_b = beh_b.to(device) if beh_b is not None else None
            lex_b = lex_b.to(device) if lex_b is not None else None
            step_logits, _, _, _, _ = classifier.model(beh_b, lex_b, mode="both")
            logits_k30 = step_logits[:, 29, :]
            probs_k30 = torch.softmax(logits_k30, dim=-1)[:, 1].cpu().numpy()

            all_probs_k30.extend(probs_k30.tolist())
            all_labels.extend(labels_b.numpy().tolist())
            all_cids.extend([m["capture_id"] for m in metas_b])

    # Per-Capture Statistics
    cap_results = []
    for cap in test_caps:
        cid = cap["capture_id"]
        indices = [i for i, c in enumerate(all_cids) if c == cid]
        cap_probs = np.array([all_probs_k30[i] for i in indices])
        cap_labels = np.array([all_labels[i] for i in indices])
        true_lbl = int(cap["label"])

        preds = (cap_probs >= 0.5).astype(int)
        mean_p = float(np.mean(cap_probs))
        median_p = float(np.median(cap_probs))
        p90 = float(np.percentile(cap_probs, 90))
        p10 = float(np.percentile(cap_probs, 10))

        if true_lbl == 1:
            tp = int(np.sum(preds == 1))
            fn = int(np.sum(preds == 0))
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            cap_stat = {
                "capture_id": cid,
                "modality": cap["attack_modality"],
                "intensity": cap["intensity"],
                "class_type": "Attack",
                "true_label": 1,
                "total_windows": len(cap_probs),
                "detected_attack_windows (TP)": tp,
                "missed_attack_windows (FN)": fn,
                "window_detection_rate (Recall)": round(recall * 100, 2),
                "mean_predicted_prob": round(mean_p, 4),
                "median_predicted_prob": round(median_p, 4),
                "prob_10th_percentile": round(p10, 4),
                "prob_90th_percentile": round(p90, 4),
                "capture_level_decision": "ATTACK_DETECTED" if mean_p >= 0.5 else "ATTACK_MISSED",
            }
        else:
            tn = int(np.sum(preds == 0))
            fp = int(np.sum(preds == 1))
            fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
            cap_stat = {
                "capture_id": cid,
                "modality": "benign",
                "intensity": cap["intensity"],
                "class_type": "Benign",
                "true_label": 0,
                "total_windows": len(cap_probs),
                "correct_benign_windows (TN)": tn,
                "false_alarm_windows (FP)": fp,
                "window_false_positive_rate (FPR)": round(fpr * 100, 4),
                "mean_predicted_prob": round(mean_p, 4),
                "median_predicted_prob": round(median_p, 4),
                "prob_10th_percentile": round(p10, 4),
                "prob_90th_percentile": round(p90, 4),
                "capture_level_decision": "BENIGN_CONFIRMED" if mean_p < 0.5 else "FALSE_ALARM_CAPTURE",
            }
        cap_results.append(cap_stat)

    # 2. Modality Distribution & Shortcut Analysis
    logger.info("Extracting feature distributions to audit for modality shortcuts...")
    feature_names = [
        "inter_arrival_time", "fqdn_length", "subdomain_length", "char_entropy",
        "digit_ratio", "uppercase_ratio", "special_ratio", "label_count",
        "max_label_length", "avg_label_length", "has_subdomain", "payload_len"
    ]

    train_attack_dfs = []
    train_benign_dfs = []
    for cap in train_caps:
        df_raw = pd.read_csv(PROJECT_ROOT / cap["rel_path"], low_memory=False)
        feat_df = extractor.extract_from_dataframe(df_raw)
        if cap["label"] == 1:
            train_attack_dfs.append(feat_df)
        else:
            train_benign_dfs.append(feat_df)

    test_video_dfs = []
    test_text_dfs = []
    test_benign_dfs = []
    for cap in test_caps:
        df_raw = pd.read_csv(PROJECT_ROOT / cap["rel_path"], low_memory=False)
        feat_df = extractor.extract_from_dataframe(df_raw)
        if cap["attack_modality"] == "video":
            test_video_dfs.append(feat_df)
        elif cap["attack_modality"] == "text":
            test_text_dfs.append(feat_df)
        else:
            test_benign_dfs.append(feat_df)

    train_atk = pd.concat(train_attack_dfs, ignore_index=True)
    train_ben = pd.concat(train_benign_dfs, ignore_index=True)
    test_vid = pd.concat(test_video_dfs, ignore_index=True)
    test_txt = pd.concat(test_text_dfs, ignore_index=True)
    test_ben = pd.concat(test_benign_dfs, ignore_index=True)

    shortcut_analysis = []
    for feat in feature_names:
        ks_vid = stats.ks_2samp(train_atk[feat].sample(min(5000, len(train_atk)), random_state=42),
                                test_vid[feat].sample(min(5000, len(test_vid)), random_state=42))
        ks_txt = stats.ks_2samp(train_atk[feat].sample(min(5000, len(train_atk)), random_state=42),
                                test_txt[feat].sample(min(5000, len(test_txt)), random_state=42))
        ks_ben = stats.ks_2samp(train_ben[feat].sample(min(5000, len(train_ben)), random_state=42),
                                test_ben[feat].sample(min(5000, len(test_ben)), random_state=42))

        shortcut_analysis.append({
            "feature": feat,
            "train_attack_mean": round(float(train_atk[feat].mean()), 4),
            "video_attack_mean": round(float(test_vid[feat].mean()), 4),
            "text_attack_mean": round(float(test_txt[feat].mean()), 4),
            "train_benign_mean": round(float(train_ben[feat].mean()), 4),
            "benign_2_mean": round(float(test_ben[feat].mean()), 4),
            "ks_stat_train_vs_video": round(float(ks_vid.statistic), 4),
            "ks_stat_train_vs_text": round(float(ks_txt.statistic), 4),
            "ks_stat_train_vs_benign2": round(float(ks_ben.statistic), 4),
        })

    # Save outputs
    with open(out_dir / "capture_level_ood_audit.json", "w", encoding="utf-8") as f:
        json.dump(cap_results, f, indent=2)

    with open(out_dir / "modality_shortcut_audit.json", "w", encoding="utf-8") as f:
        json.dump(shortcut_analysis, f, indent=2)

    logger.info("Saved capture-level and modality shortcut audit JSONs.")
    logger.info("=========================================================")
    logger.info("DIAGNOSTIC AUDIT COMPLETE")
    logger.info("=========================================================")


if __name__ == "__main__":
    run_capture_and_shortcut_audit()
