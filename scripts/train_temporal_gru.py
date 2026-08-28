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
    shuffle_order: bool = False,
    seed: int = 42,
    shuffle_labels: bool = False,
    label_seed: int = 42,
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
        shuffle_order=shuffle_order,
        seed=seed,
        shuffle_labels=shuffle_labels,
        label_seed=label_seed,
    )
    return dataset


def run_temporal_gru_training(
    epochs: int = 10,
    batch_size: int = 128,
    hidden_dim: int = 64,
    learning_rate: float = 1e-3,
    step_size: int = 10,
    max_train_samples: Optional[int] = None,
    exclude_iat: bool = False,
    shuffle_order: bool = False,
    shuffle_labels: bool = False,
    label_seed: int = 42,
):
    if shuffle_labels:
        ablation_suffix = "_label_shuffle"
        exclude_iat = True  # The controlled negative test uses 11 features without IAT
    elif shuffle_order:
        ablation_suffix = "_shuffled_order"
        exclude_iat = True
    elif exclude_iat:
        ablation_suffix = "_no_iat"
    else:
        ablation_suffix = ""

    logger.info("=========================================================")
    logger.info(
        f"STARTING DEEPDNS TEMPORAL GRU TRAINING PIPELINE "
        f"(EXCLUDE_IAT={exclude_iat}, SHUFFLE_ORDER={shuffle_order}, SHUFFLE_LABELS={shuffle_labels})"
    )
    logger.info("=========================================================")

    manifest_path = PROJECT_ROOT / "reports" / "data_pipeline" / "split_manifest.json"
    models_dir = PROJECT_ROOT / "data" / "processed" / "models"
    reports_dir = PROJECT_ROOT / "reports" / "temporal_gru"

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if exclude_iat:
        logger.info("Feature Setup: 11 causal features (inter_arrival_time excluded).")
        extractor = CausalFeatureExtractor(exclude_features=["inter_arrival_time"])
        feature_names = extractor.feature_names
        input_dim = len(feature_names)  # 11

        scaler_no_iat_path = PROJECT_ROOT / "data" / "processed" / "feature_scaler_no_iat.json"
        if scaler_no_iat_path.exists():
            scaler = FeatureScaler.from_json(scaler_no_iat_path)
            logger.info(f"Loaded existing training scaler from: {scaler_no_iat_path}")
        else:
            logger.info("Fitting dedicated 11-feature scaler exclusively on Training partition...")
            train_dfs = []
            for cap in manifest["train_captures"]:
                p = PROJECT_ROOT / cap["rel_path"]
                df_raw = pd.read_csv(p, low_memory=False, nrows=max_train_samples)
                train_dfs.append(extractor.extract_from_dataframe(df_raw))
            train_all_df = pd.concat(train_dfs, ignore_index=True)

            scaler = FeatureScaler(feature_names=feature_names)
            scaler.fit(train_all_df)
            scaler.to_json(scaler_no_iat_path)
            logger.info(f"Saved ablated training scaler to: {scaler_no_iat_path}")
    else:
        scaler_path = PROJECT_ROOT / "data" / "processed" / "feature_scaler.json"
        scaler = FeatureScaler.from_json(scaler_path)
        extractor = CausalFeatureExtractor()
        feature_names = extractor.feature_names
        input_dim = len(feature_names)  # 12

    seq_builder = SequenceBuilder(min_seq_len=5, max_seq_len=30, step_size=step_size)

    # 1. Ingest datasets
    logger.info(f"Building Training sequence dataset (shuffle_order={shuffle_order}, shuffle_labels={shuffle_labels})...")
    train_dataset = build_partition_dataset(
        manifest["train_captures"],
        extractor=extractor,
        scaler=scaler,
        seq_builder=seq_builder,
        max_rows_per_capture=max_train_samples,
        shuffle_order=shuffle_order,
        seed=42,
        shuffle_labels=shuffle_labels,
        label_seed=label_seed,
    )
    logger.info(f"Train Sequence Windows: {len(train_dataset):,}")

    logger.info(f"Building Validation sequence dataset (labels untouched)...")
    val_dataset = build_partition_dataset(
        manifest["val_captures"],
        extractor=extractor,
        scaler=scaler,
        seq_builder=seq_builder,
        shuffle_order=shuffle_order,
        seed=1042,
        shuffle_labels=False,
    )
    logger.info(f"Val Sequence Windows: {len(val_dataset):,}")

    logger.info(f"Building Test sequence dataset (labels untouched)...")
    test_dataset = build_partition_dataset(
        manifest["test_captures"],
        extractor=extractor,
        scaler=scaler,
        seq_builder=seq_builder,
        shuffle_order=shuffle_order,
        seed=2042,
        shuffle_labels=False,
    )
    logger.info(f"Test Sequence Windows: {len(test_dataset):,}")

    # 2. Train Temporal GRU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Target Compute Device: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")

    gru_classifier = TemporalGRUClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=1,
        num_classes=2,
        dropout_rate=0.1,
        learning_rate=learning_rate,
        batch_size=batch_size,
        device=device,
    )
    gru_classifier.feature_names = feature_names

    logger.info(f"\n--- Training Temporal GRU ({epochs} Epochs, input_dim={input_dim}, shuffle_labels={shuffle_labels}) ---")
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
    model_save_path = models_dir / f"temporal_gru{ablation_suffix}.pt"
    gru_classifier.save(model_save_path)
    logger.info(f"Saved trained Temporal GRU model to: {model_save_path}")

    results_save_path = reports_dir / f"temporal_gru_results{ablation_suffix}.json"
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
    parser.add_argument("--exclude_iat", action="store_true", help="Exclude inter_arrival_time for controlled ablation study.")
    parser.add_argument("--shuffle_order", action="store_true", help="Permute temporal order within each sequence window for order-ablation study.")
    parser.add_argument("--shuffle_labels", action="store_true", help="Permute training labels for negative-control Y-permutation test.")
    parser.add_argument("--label_seed", type=int, default=42, help="Seed for training label permutation.")
    args = parser.parse_args()

    run_temporal_gru_training(
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        step_size=args.step_size,
        max_train_samples=args.max_train_samples,
        exclude_iat=args.exclude_iat,
        shuffle_order=args.shuffle_order,
        shuffle_labels=args.shuffle_labels,
        label_seed=args.label_seed,
    )
