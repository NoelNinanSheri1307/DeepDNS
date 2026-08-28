"""
DeepDNS Multi-View Fusion Training and Multi-Horizon Evaluation Script.

Trains the unified Dual-View Network (Temporal GRU + Character-CNN + Late Fusion Head)
and evaluates performance across expanding evidence horizons K in [5, 10, 15, 20, 25, 30].
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
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import CausalFeatureExtractor, FeatureScaler
from src.data.sequences import SequenceBuilder
from src.models.char_cnn import CharacterTokenizer
from src.data.multiview_dataset import build_multiview_partition_dataset, multiview_collate_fn
from src.models.multiview_fusion import DeepDNSMultiViewClassifier
from src.evaluation.metrics import compute_classification_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TrainMultiView")


def run_multiview_training(
    epochs: int = 10,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    mode: str = "both",
    step_size: int = 10,
    max_train_samples: Optional[int] = None,
):
    logger.info("=========================================================")
    logger.info(f"STARTING DEEPDNS MULTI-VIEW FUSION TRAINING (MODE: {mode.upper()})")
    logger.info("=========================================================")

    manifest_path = PROJECT_ROOT / "reports" / "data_pipeline" / "split_manifest.json"
    scaler_path = PROJECT_ROOT / "data" / "processed" / "feature_scaler.json"
    models_dir = PROJECT_ROOT / "data" / "processed" / "models"
    reports_dir = PROJECT_ROOT / "reports" / "multiview"

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    scaler = FeatureScaler.from_json(scaler_path)
    extractor = CausalFeatureExtractor()
    tokenizer = CharacterTokenizer()
    seq_builder = SequenceBuilder(min_seq_len=5, max_seq_len=30, step_size=step_size)

    # 1. Build Multi-View Datasets
    logger.info("Building Training Multi-View dataset...")
    train_dataset = build_multiview_partition_dataset(
        manifest["train_captures"],
        project_root=PROJECT_ROOT,
        extractor=extractor,
        scaler=scaler,
        tokenizer=tokenizer,
        seq_builder=seq_builder,
        max_rows_per_capture=max_train_samples,
    )
    logger.info(f"Train Sequence Windows: {len(train_dataset):,}")

    logger.info("Building Validation Multi-View dataset...")
    val_dataset = build_multiview_partition_dataset(
        manifest["val_captures"],
        project_root=PROJECT_ROOT,
        extractor=extractor,
        scaler=scaler,
        tokenizer=tokenizer,
        seq_builder=seq_builder,
    )
    logger.info(f"Val Sequence Windows: {len(val_dataset):,}")

    logger.info("Building Test Multi-View dataset...")
    test_dataset = build_multiview_partition_dataset(
        manifest["test_captures"],
        project_root=PROJECT_ROOT,
        extractor=extractor,
        scaler=scaler,
        tokenizer=tokenizer,
        seq_builder=seq_builder,
    )
    logger.info(f"Test Sequence Windows: {len(test_dataset):,}")

    # 2. Train Multi-View Network
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

    logger.info(f"\n--- Training DeepDNS Multi-View ({epochs} Epochs, Mode: {mode}) ---")
    classifier.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=epochs,
        mode=mode,
        verbose=True,
        collate_fn=multiview_collate_fn,
    )

    # 3. Multi-Horizon Evaluation on Test Partition
    logger.info(f"\n--- Multi-Horizon Evaluation on Test Partition (K = 5, 10, 15, 20, 25, 30) ---")
    horizons = [5, 10, 15, 20, 25, 30]
    horizon_reports = classifier.evaluate_at_horizons(
        test_dataset,
        horizons=horizons,
        mode=mode,
        split_name="test",
        collate_fn=multiview_collate_fn,
    )

    results = {}
    for k, rep in horizon_reports.items():
        print(rep.summary_table())
        results[f"K_{k}"] = rep.to_dict()

    # 4. Save Model & Reports
    model_save_path = models_dir / f"multiview_{mode}.pt"
    classifier.save(model_save_path)
    logger.info(f"Saved trained Multi-View model to: {model_save_path}")

    results_save_path = reports_dir / f"multiview_results_{mode}.json"
    with open(results_save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved multi-horizon evaluation results to: {results_save_path}")

    logger.info("=========================================================")
    logger.info("MULTI-VIEW TRAINING & EVALUATION COMPLETE")
    logger.info("=========================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate DeepDNS Multi-View Fusion model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--mode", type=str, default="both", choices=["both", "behavioral_only", "lexical_only"], help="Ablation mode.")
    parser.add_argument("--step_size", type=int, default=10, help="Stride between sequence window starts.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional sample cap per capture.")
    args = parser.parse_args()

    run_multiview_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        mode=args.mode,
        step_size=args.step_size,
        max_train_samples=args.max_train_samples,
    )
