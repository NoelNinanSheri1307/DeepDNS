"""
DeepDNS Temporal GRU Training and Multi-Horizon Evaluation Script.

Trains the causal Temporal GRU model on the training partition sequences
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

from src.data.labels import parse_cic_bell_label
from src.data.features import CausalFeatureExtractor, FeatureScaler
from src.data.sequences import SequenceBuilder, StreamingSequenceDataset
from src.models.temporal_gru import TemporalGRUClassifier
from src.evaluation.metrics import compute_classification_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TrainTemporalGRU")


def build_partition_dataset(
    captures_info: list,
    extractor: CausalFeatureExtractor,
    scaler: FeatureScaler,
    seq_builder: SequenceBuilder,
    max_rows_per_capture: Optional[int] = None,
) -> StreamingSequenceDataset:
    """
    Ingests captures, extracts causal features, standardizes via scaler,
    and constructs StreamingSequenceDataset with causal prefix windows.
    """
    all_features = []
    all_windows = []
    current_offset = 0

    for cap in captures_info:
        p = PROJECT_ROOT / cap["rel_path"]
        df_raw = pd.read_csv(p, low_memory=False, nrows=max_rows_per_capture)
        meta = parse_cic_bell_label(p)

        feat_df = extractor.extract_from_dataframe(df_raw)
        norm_feats = scaler.transform(feat_df)

        windows = seq_builder.build_prefix_windows(
            total_rows=len(norm_feats),
            meta=meta,
            offset=current_offset,
        )

        all_features.append(norm_feats)
        all_windows.extend(windows)
        current_offset += len(norm_feats)

    features_matrix = np.concatenate(all_features, axis=0)
    dataset = StreamingSequenceDataset(
        features=features_matrix,
        windows=all_windows,
        max_seq_len=seq_builder.max_seq_len,
    )
    return dataset


def run_temporal_gru_training(
    epochs: int = 10,
    batch_size: int = 128,
    hidden_dim: int = 64,
    learning_rate: float = 1e-3,
    step_size: int = 10,
    max_train_samples: Optional[int] = None,
):
    logger.info("=========================================================")
    logger.info("STARTING DEEPDNS TEMPORAL GRU TRAINING PIPELINE")
    logger.info("=========================================================")

    manifest_path = PROJECT_ROOT / "reports" / "data_pipeline" / "split_manifest.json"
    scaler_path = PROJECT_ROOT / "data" / "processed" / "feature_scaler.json"
    models_dir = PROJECT_ROOT / "data" / "processed" / "models"
    reports_dir = PROJECT_ROOT / "reports" / "temporal_gru"

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    scaler = FeatureScaler.from_json(scaler_path)
    extractor = CausalFeatureExtractor()
    seq_builder = SequenceBuilder(min_seq_len=5, max_seq_len=30, step_size=step_size)

    # 1. Ingest datasets
    logger.info("Building Training sequence dataset...")
    train_dataset = build_partition_dataset(
        manifest["train_captures"],
        extractor=extractor,
        scaler=scaler,
        seq_builder=seq_builder,
        max_rows_per_capture=max_train_samples,
    )
    logger.info(f"Train Sequence Windows: {len(train_dataset):,}")

    logger.info("Building Validation sequence dataset...")
    val_dataset = build_partition_dataset(
        manifest["val_captures"],
        extractor=extractor,
        scaler=scaler,
        seq_builder=seq_builder,
    )
    logger.info(f"Val Sequence Windows: {len(val_dataset):,}")

    logger.info("Building Test sequence dataset...")
    test_dataset = build_partition_dataset(
        manifest["test_captures"],
        extractor=extractor,
        scaler=scaler,
        seq_builder=seq_builder,
    )
    logger.info(f"Test Sequence Windows: {len(test_dataset):,}")

    # 2. Train Temporal GRU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Target Compute Device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")

    gru_classifier = TemporalGRUClassifier(
        input_dim=12,
        hidden_dim=hidden_dim,
        num_layers=1,
        num_classes=2,
        dropout_rate=0.1,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device,
    )

    logger.info(f"\n--- Training Temporal GRU ({epochs} Epochs) ---")
    gru_classifier.fit(train_dataset=train_dataset, val_dataset=val_dataset, epochs=epochs, verbose=True)

    # 3. Multi-Horizon Evaluation on Test Partition
    logger.info("\n--- Multi-Horizon Evaluation on Test Partition (K = 5, 10, 15, 20, 25, 30) ---")
    horizons = [5, 10, 15, 20, 25, 30]
    horizon_reports = gru_classifier.evaluate_at_horizons(test_dataset, horizons=horizons, split_name="test")

    results = {}
    for k, rep in horizon_reports.items():
        print(rep.summary_table())
        results[f"K_{k}"] = rep.to_dict()

    # 4. Save model and reports
    model_save_path = models_dir / "temporal_gru.pt"
    gru_classifier.save(model_save_path)
    logger.info(f"Saved trained Temporal GRU model to: {model_save_path}")

    results_save_path = reports_dir / "temporal_gru_results.json"
    with open(results_save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved multi-horizon evaluation results to: {results_save_path}")

    logger.info("=========================================================")
    logger.info("TEMPORAL GRU TRAINING & EVALUATION COMPLETE")
    logger.info("=========================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate DeepDNS Temporal GRU model.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--hidden_dim", type=int, default=64, help="GRU hidden dimension.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--step_size", type=int, default=10, help="Stride between sequence starts.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Optional sample cap per capture.")
    args = parser.parse_args()

    run_temporal_gru_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        step_size=args.step_size,
        max_train_samples=args.max_train_samples,
    )
