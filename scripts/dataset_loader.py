"""
DeepDNS Production Dataset Loader and Validation Pipeline.

Orchestrates data discovery, schema validation, leakage-safe grouped splitting,
causal feature extraction, scaler fitting, and sequential window construction.
"""

import os
import sys
import json
import gzip
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.schema import validate_stateless_dataframe, CAUSAL_FEATURE_NAMES, FORBIDDEN_MODEL_COLUMNS
from src.data.labels import parse_cic_bell_label, CaptureMetadata
from src.data.features import CausalFeatureExtractor, FeatureScaler
from src.data.sequences import SequenceBuilder, StreamingSequenceDataset
from src.data.splits import GroupedSplitter, SplitManifest
from src.data.lexical import DNSThreatsLexicalTokenizer, DNSThreatsDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("DeepDNS_Pipeline")


def run_pipeline():
    logger.info("==================================================")
    logger.info("STARTING DEEPDNS PRODUCTION DATA PIPELINE")
    logger.info("==================================================")

    reports_dir = PROJECT_ROOT / "reports" / "data_pipeline"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    reports_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------
    # 1. DATA DISCOVERY
    # ----------------------------------------------------
    logger.info("Phase 1: Discovering raw datasets...")
    raw_dir = PROJECT_ROOT / "data" / "raw"
    inventory = []

    for root, _, files in os.walk(raw_dir):
        for f in files:
            full_path = Path(root) / f
            rel_path = full_path.relative_to(PROJECT_ROOT).as_posix()
            ext = full_path.suffix.lower()
            is_gz = f.endswith(".gz")

            dataset_source = "cic_bell_dns_exf_2021" if "cic_bell_dns_exf_2021" in rel_path else (
                "dns_threats" if "dns_threats" in rel_path else "other"
            )

            state_type = "n/a"
            if "stateful" in f.lower():
                state_type = "stateful"
            elif "stateless" in f.lower():
                state_type = "stateless"

            # Row count
            row_count = 0
            if is_gz:
                with gzip.open(full_path, "rt", encoding="utf-8", errors="ignore") as gf:
                    for _ in gf:
                        row_count += 1
                row_count = max(0, row_count - 1)
            elif ext == ".csv":
                with open(full_path, "r", encoding="utf-8", errors="ignore") as cf:
                    for _ in cf:
                        row_count += 1
                row_count = max(0, row_count - 1)

            # Metadata parsing for CIC-Bell
            meta_dict = {}
            if dataset_source == "cic_bell_dns_exf_2021":
                try:
                    meta = parse_cic_bell_label(full_path)
                    meta_dict = meta.to_dict()
                except Exception as e:
                    meta_dict = {"parse_error": str(e)}

            item = {
                "filename": f,
                "rel_path": rel_path,
                "full_path": str(full_path.resolve()),
                "dataset_source": dataset_source,
                "state_type": state_type,
                "row_count": row_count,
                "file_size_mb": round(full_path.stat().st_size / (1024 * 1024), 4),
                **meta_dict,
            }
            inventory.append(item)

    inv_df = pd.DataFrame(inventory)
    inv_csv_path = reports_dir / "dataset_inventory.csv"
    inv_df.to_csv(inv_csv_path, index=False)
    logger.info(f"Saved inventory: {inv_csv_path} ({len(inventory)} files)")

    # ----------------------------------------------------
    # 2. STATELESS SCHEMA VALIDATION
    # ----------------------------------------------------
    logger.info("Phase 2: Validating CIC-Bell stateless schemas...")
    stateless_files = [i for i in inventory if i.get("dataset_source") == "cic_bell_dns_exf_2021" and i.get("state_type") == "stateless"]
    
    validation_records = []
    total_stateless_rows = 0

    for sfile in stateless_files:
        p = PROJECT_ROOT / sfile["rel_path"]
        df_raw = pd.read_csv(p, low_memory=False)
        v_res = validate_stateless_dataframe(df_raw, source_name=sfile["filename"])
        total_stateless_rows += v_res.total_rows

        validation_records.append({
            "filename": sfile["filename"],
            "capture_id": sfile.get("capture_id"),
            "is_valid": v_res.is_valid,
            "total_rows": v_res.total_rows,
            "missing_columns_count": len(v_res.missing_columns),
            "unexpected_columns_count": len(v_res.unexpected_columns),
            "timestamp_parseable": v_res.timestamp_parseable,
            "is_chronological": v_res.is_chronological,
            "non_numeric_count": len(v_res.non_numeric_columns),
            "total_nulls": sum(v_res.null_counts.values()),
            "anomalies_count": len(v_res.anomalies),
            "anomalies": "; ".join(v_res.anomalies) if v_res.anomalies else "None",
        })

    val_df = pd.DataFrame(validation_records)
    val_csv_path = reports_dir / "stateless_schema_validation.csv"
    val_df.to_csv(val_csv_path, index=False)
    logger.info(f"Validated {len(stateless_files)} stateless CSVs. Total rows: {total_stateless_rows:,}")

    # ----------------------------------------------------
    # 3. GROUPED DATASET SPLITTING (ZERO LEAKAGE)
    # ----------------------------------------------------
    logger.info("Phase 3: Creating session-grouped split manifest...")
    splitter = GroupedSplitter(target_train_pct=0.70, target_val_pct=0.15)
    manifest = splitter.create_canonical_split(inventory)
    manifest_path = reports_dir / "split_manifest.json"
    splitter.save_manifest(manifest, manifest_path)
    logger.info(f"Split Manifest saved: {manifest_path}")
    logger.info(f"  Train captures: {len(manifest.train_captures)} ({manifest.train_total_rows:,} rows, {manifest.train_row_pct}%)")
    logger.info(f"  Val captures:   {len(manifest.val_captures)} ({manifest.val_total_rows:,} rows, {manifest.val_row_pct}%)")
    logger.info(f"  Test captures:  {len(manifest.test_captures)} ({manifest.test_total_rows:,} rows, {manifest.test_row_pct}%)")

    # ----------------------------------------------------
    # 4. CAUSAL FEATURE EXTRACTION & SCALER FITTING
    # ----------------------------------------------------
    logger.info("Phase 4: Extracting causal features and fitting scaler on Train partition...")
    extractor = CausalFeatureExtractor()

    # Ingest and extract all train captures
    train_feature_dfs = []
    for cap_info in manifest.train_captures:
        p = PROJECT_ROOT / cap_info["rel_path"]
        df_raw = pd.read_csv(p, low_memory=False)
        feat_df = extractor.extract_from_dataframe(df_raw)
        train_feature_dfs.append(feat_df)

    all_train_feats = pd.concat(train_feature_dfs, ignore_index=True)
    logger.info(f"Fitting FeatureScaler on {len(all_train_feats):,} training feature vectors...")

    scaler = FeatureScaler(clip_outliers=True, clip_std=8.0)
    scaler.fit(all_train_feats)
    scaler_path = processed_dir / "feature_scaler.json"
    scaler.to_json(scaler_path)
    logger.info(f"Saved fitted scaler: {scaler_path}")

    # ----------------------------------------------------
    # 5. CAUSAL SEQUENCE CONSTRUCTION VERIFICATION
    # ----------------------------------------------------
    logger.info("Phase 5: Verifying causal sequence prefix generation (K=5..30)...")
    seq_builder = SequenceBuilder(min_seq_len=5, max_seq_len=30, step_size=10)

    total_train_windows = 0
    total_val_windows = 0
    total_test_windows = 0

    for split_name, cap_list in [("Train", manifest.train_captures), ("Val", manifest.val_captures), ("Test", manifest.test_captures)]:
        split_windows_count = 0
        for cap_info in cap_list:
            meta = parse_cic_bell_label(PROJECT_ROOT / cap_info["rel_path"])
            df_raw = pd.read_csv(PROJECT_ROOT / cap_info["rel_path"], low_memory=False)
            feat_df = extractor.extract_from_dataframe(df_raw)
            norm_feats = scaler.transform(feat_df)
            windows = seq_builder.build_prefix_windows(total_rows=len(norm_feats), meta=meta, offset=0)
            split_windows_count += len(windows)

        if split_name == "Train":
            total_train_windows = split_windows_count
        elif split_name == "Val":
            total_val_windows = split_windows_count
        else:
            total_test_windows = split_windows_count

        logger.info(f"  {split_name} Prefix Windows generated: {split_windows_count:,}")

    # ----------------------------------------------------
    # 6. LEXICAL TOKENIZER VERIFICATION
    # ----------------------------------------------------
    logger.info("Phase 6: Verifying DNS Threats Lexical Tokenizer...")
    tokenizer = DNSThreatsLexicalTokenizer(max_length=128)
    sample_domain = "exfil-01a9f3b.tunnel.attacker.domain.com"
    encoded = tokenizer.encode_domain(sample_domain)
    decoded = tokenizer.decode_tokens(encoded)
    logger.info(f"  Sample Domain: {sample_domain}")
    logger.info(f"  Encoded Shape: {encoded.shape}, Non-zero tokens: {np.count_nonzero(encoded)}")
    logger.info(f"  Decoded Match: '{decoded}' == '{sample_domain.lower()}' -> {decoded == sample_domain.lower()}")

    # Save summary json
    pipeline_summary = {
        "status": "SUCCESS",
        "total_files_discovered": len(inventory),
        "cic_bell_stateless_captures": len(stateless_files),
        "total_stateless_rows": total_stateless_rows,
        "causal_features_extracted": CAUSAL_FEATURE_NAMES,
        "forbidden_columns_excluded": FORBIDDEN_MODEL_COLUMNS,
        "scaler_fitted_on_rows": len(all_train_feats),
        "train_windows_count": total_train_windows,
        "val_windows_count": total_val_windows,
        "test_windows_count": total_test_windows,
        "dns_threats_vocab_size": tokenizer.vocab_size,
    }
    with open(reports_dir / "dataset_validation.json", "w", encoding="utf-8") as f:
        json.dump(pipeline_summary, f, indent=2)

    logger.info("==================================================")
    logger.info("DATA PIPELINE INITIALIZATION COMPLETE")
    logger.info("==================================================")


if __name__ == "__main__":
    run_pipeline()
