"""
DeepDNS Out-of-Distribution (LOMO) Multi-View Training and Evaluation Script.

Trains on: Audio, Compressed, Exe, Benign captures
Validates on: Image, Benign_Heavy_2 captures
Evaluates on: 100% UNSEEN Modalities (Video, Text, Benign_2)

Outputs:
  - data/processed/models/multiview_ood_{mode}.pt
  - reports/multiview_ood/multiview_ood_results_{mode}.json
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import CausalFeatureExtractor, FeatureScaler
from src.data.sequences import SequenceBuilder
from src.models.char_cnn import CharacterTokenizer
from src.data.multiview_dataset import build_multiview_partition_dataset, multiview_collate_fn
from src.models.multiview_fusion import DeepDNSMultiViewClassifier
from src.data.ood_splits import build_lomo_split_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TrainMultiViewOOD")


def run_multiview_ood_training(
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    mode: str = "both",
    step_size: int = 10,
    max_train_samples: Optional[int] = None,
):
    logger.info("=========================================================")
    logger.info(f"STARTING DEEPDNS OOD (LOMO) MULTI-VIEW TRAINING (MODE: {mode.upper()})")
    logger.info("=========================================================")

    manifest_path = PROJECT_ROOT / "reports" / "data_pipeline" / "ood_split_manifest.json"
    if not manifest_path.exists():
        manifest = build_lomo_split_manifest()
    else:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

    models_dir = PROJECT_ROOT / "data" / "processed" / "models"
    reports_dir = PROJECT_ROOT / "reports" / "multiview_ood"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Scaler must be fit ONLY on LOMO training partition
    scaler_path = PROJECT_ROOT / "data" / "processed" / "feature_scaler_lomo.json"
    extractor = CausalFeatureExtractor()
    tokenizer = CharacterTokenizer()
    seq_builder = SequenceBuilder(min_seq_len=5, max_seq_len=30, step_size=step_size)

    if scaler_path.exists():
        scaler = FeatureScaler.from_json(scaler_path)
    else:
        logger.info("Fitting dedicated LOMO training scaler...")
        train_dfs = []
        import pandas as pd
        for cap in manifest["train_captures"]:
            df_raw = pd.read_csv(PROJECT_ROOT / cap["rel_path"], low_memory=False)
            train_dfs.append(extractor.extract_from_dataframe(df_raw))
        train_all_df = pd.concat(train_dfs, ignore_index=True)
        scaler = FeatureScaler()
        scaler.fit(train_all_df)
        scaler.to_json(scaler_path)

    # 1. Ingest LOMO Datasets
    logger.info(f"Building LOMO Training dataset (Modalities: {manifest['training_modalities']})...")
    train_dataset = build_multiview_partition_dataset(
        manifest["train_captures"],
        project_root=PROJECT_ROOT,
        extractor=extractor,
        scaler=scaler,
        tokenizer=tokenizer,
        seq_builder=seq_builder,
        max_rows_per_capture=max_train_samples,
    )
    logger.info(f"LOMO Train Windows: {len(train_dataset):,}")

    logger.info(f"Building LOMO Validation dataset (Modalities: {manifest['held_out_val_modalities']})...")
    val_dataset = build_multiview_partition_dataset(
        manifest["val_captures"],
        project_root=PROJECT_ROOT,
        extractor=extractor,
        scaler=scaler,
        tokenizer=tokenizer,
        seq_builder=seq_builder,
    )
    logger.info(f"LOMO Val Windows: {len(val_dataset):,}")

    logger.info(f"Building LOMO Test dataset (100% UNSEEN Modalities: {manifest['held_out_test_modalities']})...")
    test_dataset = build_multiview_partition_dataset(
        manifest["test_captures"],
        project_root=PROJECT_ROOT,
        extractor=extractor,
        scaler=scaler,
        tokenizer=tokenizer,
        seq_builder=seq_builder,
    )
    logger.info(f"LOMO Test (OOD) Windows: {len(test_dataset):,}")

    # 2. Initialize Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Target Compute Device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")

    classifier = DeepDNSMultiViewClassifier(
        beh_input_dim=12,
        beh_hidden_dim=64,
        lex_vocab_size=tokenizer.vocab_size,
        lex_embedding_dim=32,
        lex_output_dim=128,
        fuse_dim=64,
        num_classes=2,
        dropout_rate=0.1,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device,
    )

    logger.info(f"\n--- Training DeepDNS Multi-View OOD ({epochs} Epochs, Mode: {mode}) ---")
    classifier.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        mode=mode,
        verbose=True,
        collate_fn=multiview_collate_fn,
    )

    # 3. Multi-Horizon OOD Evaluation on Test Partition
    logger.info("\n--- Multi-Horizon Evaluation on Held-Out OOD Test Partition (K = 5, 10, 15, 20, 25, 30) ---")
    horizons = [5, 10, 15, 20, 25, 30]
    horizon_reports = classifier.evaluate_at_horizons(
        test_dataset,
        horizons=horizons,
        mode=mode,
        split_name="ood_test",
        collate_fn=multiview_collate_fn,
    )

    results = {}
    for k, rep in horizon_reports.items():
        print(rep.summary_table())
        results[f"K_{k}"] = rep.to_dict()

    # 4. Modality Breakdown at K=30
    logger.info("\n--- Sub-Modality Evaluation Breakdown at Horizon K=30 ---")
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=multiview_collate_fn,
    )

    all_probs = []
    all_labels = []
    all_modalities = []
    all_cids = []

    classifier.model.eval()
    with torch.no_grad():
        for beh_b, lex_b, lens_b, labels_b, metas_b in test_loader:
            beh_b = beh_b.to(device) if beh_b is not None else None
            lex_b = lex_b.to(device) if lex_b is not None else None
            step_logits, _, _, _, _ = classifier.model(beh_b, lex_b, mode=mode)
            logits_k30 = step_logits[:, 29, :]
            probs_k30 = torch.softmax(logits_k30, dim=-1)[:, 1].cpu().numpy()

            all_probs.extend(probs_k30.tolist())
            all_labels.extend(labels_b.numpy().tolist())
            all_modalities.extend([str(m.get("attack_modality", "benign")) for m in metas_b])
            all_cids.extend([m.get("capture_id", "unknown") for m in metas_b])

    y_prob_arr = np.array(all_probs)
    y_true_arr = np.array(all_labels)
    y_pred_arr = (y_prob_arr >= 0.5).astype(int)
    mods_arr = np.array(all_modalities)

    mod_breakdown = {}
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    for target_mod in ["video", "text", "None"]:
        mask = (mods_arr == target_mod)
        if np.sum(mask) > 0:
            mod_y_true = y_true_arr[mask]
            mod_y_pred = y_pred_arr[mask]
            mod_name = "benign_2" if target_mod == "None" else f"{target_mod}_only"
            acc = float(accuracy_score(mod_y_true, mod_y_pred))
            rec = float(recall_score(mod_y_true, mod_y_pred, zero_division=0))
            prec = float(precision_score(mod_y_true, mod_y_pred, zero_division=0))
            f1 = float(f1_score(mod_y_true, mod_y_pred, zero_division=0))
            cm = confusion_matrix(mod_y_true, mod_y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            mod_report = {
                "modality": mod_name,
                "total_windows": int(np.sum(mask)),
                "accuracy": round(acc * 100, 2),
                "precision": round(prec * 100, 2),
                "recall": round(rec * 100, 2),
                "f1_score": round(f1, 4),
                "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
            }
            mod_breakdown[mod_name] = mod_report
            logger.info(f"  [{mod_name.upper()}] Windows: {mod_report['total_windows']:,} | Acc: {mod_report['accuracy']}% | Rec: {mod_report['recall']}% | Prec: {mod_report['precision']}% | F1: {mod_report['f1_score']} | CM: {mod_report['confusion_matrix']}")

    results["modality_breakdown_k30"] = mod_breakdown

    # 5. Save Model Checkpoint & JSON Report
    model_save_path = models_dir / f"multiview_ood_{mode}.pt"
    classifier.save(model_save_path)
    logger.info(f"\nSaved trained OOD Multi-View model to: {model_save_path}")

    results_save_path = reports_dir / f"multiview_ood_results_{mode}.json"
    with open(results_save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved OOD multi-horizon evaluation results to: {results_save_path}")

    logger.info("=========================================================")
    logger.info("OOD MULTI-VIEW TRAINING & EVALUATION COMPLETE")
    logger.info("=========================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate DeepDNS Multi-View OOD (LOMO) model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--mode", type=str, default="both", choices=["both", "behavioral_only", "lexical_only"], help="Ablation mode.")
    parser.add_argument("--step_size", type=int, default=10, help="Stride between sequence window starts.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional sample cap per capture.")
    args = parser.parse_args()

    run_multiview_ood_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        mode=args.mode,
        step_size=args.step_size,
        max_train_samples=args.max_train_samples,
    )
