"""
Lightweight diagnostic script to audit temporal GRU sequence construction and leakage.
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.labels import parse_cic_bell_label
from src.data.features import CausalFeatureExtractor, FeatureScaler
from src.data.sequences import SequenceBuilder, StreamingSequenceDataset

def run_leakage_audit():
    manifest_path = PROJECT_ROOT / "reports" / "data_pipeline" / "split_manifest.json"
    scaler_path = PROJECT_ROOT / "data" / "processed" / "feature_scaler.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    with open(scaler_path, "r", encoding="utf-8") as f:
        scaler_data = json.load(f)

    # 1. Capture Disjointness Audit
    train_caps = set(c["capture_id"] for c in manifest["train_captures"])
    val_caps = set(c["capture_id"] for c in manifest["val_captures"])
    test_caps = set(c["capture_id"] for c in manifest["test_captures"])

    train_val_overlap = train_caps.intersection(val_caps)
    train_test_overlap = train_caps.intersection(test_caps)
    val_test_overlap = val_caps.intersection(test_caps)

    # 2. Sequence Window Overlap within Test
    seq_builder = SequenceBuilder(min_seq_len=5, max_seq_len=30, step_size=10)
    
    test_windows = []
    current_offset = 0
    total_test_rows = 0
    cap_offsets = {}

    for cap in manifest["test_captures"]:
        p = PROJECT_ROOT / cap["rel_path"]
        df = pd.read_csv(p, low_memory=False)
        meta = parse_cic_bell_label(p)
        total_rows = len(df)
        total_test_rows += total_rows

        windows = seq_builder.build_prefix_windows(total_rows=total_rows, meta=meta, offset=current_offset)
        cap_offsets[cap["capture_id"]] = (current_offset, current_offset + total_rows)
        test_windows.extend(windows)
        current_offset += total_rows

    # Verify that no window in test spans across capture boundaries
    boundary_violations = []
    for w in test_windows:
        cap_start, cap_end = cap_offsets[w.capture_id]
        if w.start_idx < cap_start or w.end_idx > cap_end:
            boundary_violations.append(w)

    # Check inter-window query overlap for consecutive sliding windows
    overlap_ratios = []
    for i in range(len(test_windows) - 1):
        w1 = test_windows[i]
        w2 = test_windows[i+1]
        if w1.capture_id == w2.capture_id and w1.seq_len == 30 and w2.seq_len == 30:
            # Range overlap
            overlap_queries = max(0, min(w1.end_idx, w2.end_idx) - max(w1.start_idx, w2.start_idx))
            overlap_ratios.append(overlap_queries / 30.0)

    mean_overlap = float(np.mean(overlap_ratios)) if overlap_ratios else 0.0

    # 3. Label Consistency Audit
    # Verify that all windows in a capture have identical label equal to meta.label
    label_inconsistencies = 0
    for w in test_windows:
        expected_label = 0 if "benign" in w.capture_id else 1
        if w.label != expected_label:
            label_inconsistencies += 1

    audit_results = {
        "train_captures_count": len(train_caps),
        "val_captures_count": len(val_caps),
        "test_captures_count": len(test_caps),
        "cross_split_capture_overlap": {
            "train_val": list(train_val_overlap),
            "train_test": list(train_test_overlap),
            "val_test": list(val_test_overlap),
        },
        "test_sequence_windows_count": len(test_windows),
        "test_boundary_violations": len(boundary_violations),
        "consecutive_window_overlap_ratio": round(mean_overlap, 4),
        "label_inconsistencies": label_inconsistencies,
        "scaler_fitted_on_split": scaler_data.get("fitted_on_split", "train"),
        "scaler_n_samples_seen": scaler_data.get("n_samples_seen", 0),
    }

    with open(PROJECT_ROOT / "reports" / "temporal_gru" / "audit_numbers.json", "w", encoding="utf-8") as f:
        json.dump(audit_results, f, indent=2)

    print("Audit numbers generated successfully!")
    print(json.dumps(audit_results, indent=2))

if __name__ == "__main__":
    run_leakage_audit()
