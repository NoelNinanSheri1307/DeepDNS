"""
Leave-One-Modality-Out (LOMO) Out-of-Distribution Split Generator for DeepDNS.

Guarantees:
1. Strict Capture-Grouped boundaries (zero capture overlap).
2. Complete modality isolation:
   - Train: Audio, Compressed, Exe, Benign (10 captures)
   - Val: Image, Benign_Heavy_2 (3 captures)
   - Test (OOD): Video, Text, Benign_2 (5 captures)
3. Zero training exposure to Video or Text exfiltration modalities.
"""

import json
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def build_lomo_split_manifest() -> Dict[str, Any]:
    """Constructs the Leave-One-Modality-Out (LOMO) OOD split manifest."""
    
    # Load base capture information from standard manifest
    base_manifest_path = PROJECT_ROOT / "reports" / "data_pipeline" / "split_manifest.json"
    with open(base_manifest_path, "r", encoding="utf-8") as f:
        base_manifest = json.load(f)
    
    all_captures = (
        base_manifest["train_captures"] +
        base_manifest["val_captures"] +
        base_manifest["test_captures"]
    )
    
    # Define strict modality partitioning
    # Train: audio, compressed, exe + benign (1, 3, light_benign, heavy_1)
    train_caps = [
        c for c in all_captures
        if c["attack_modality"] in ["audio", "compressed", "exe"] or
           c["capture_id"] in ["benign_heavy_1", "benign_heavy_3", "light_benign", "benign_1"]
    ]
    
    # Val: image + benign_heavy_2
    val_caps = [
        c for c in all_captures
        if c["attack_modality"] == "image" or c["capture_id"] == "benign_heavy_2"
    ]
    
    # Test / OOD: video, text + benign_2 (100% held-out modalities)
    test_caps = [
        c for c in all_captures
        if c["attack_modality"] in ["video", "text"] or c["capture_id"] == "benign_2"
    ]
    
    train_rows = sum(c["row_count"] for c in train_caps)
    val_rows = sum(c["row_count"] for c in val_caps)
    test_rows = sum(c["row_count"] for c in test_caps)
    total_rows = train_rows + val_rows + test_rows
    
    manifest = {
        "split_strategy": "Leave_One_Modality_Out_Capture_Grouped",
        "description": "Strict OOD split: Video and Text modalities are 100% held out for Test evaluation.",
        "held_out_test_modalities": ["video", "text"],
        "held_out_val_modalities": ["image"],
        "training_modalities": ["audio", "compressed", "exe"],
        "train_row_pct": round((train_rows / total_rows) * 100, 2),
        "val_row_pct": round((val_rows / total_rows) * 100, 2),
        "test_row_pct": round((test_rows / total_rows) * 100, 2),
        "train_total_rows": train_rows,
        "val_total_rows": val_rows,
        "test_total_rows": test_rows,
        "train_captures": train_caps,
        "val_captures": val_caps,
        "test_captures": test_caps,
    }
    
    out_path = PROJECT_ROOT / "reports" / "data_pipeline" / "ood_split_manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    return manifest


if __name__ == "__main__":
    manifest = build_lomo_split_manifest()
    print(f"Generated OOD Split Manifest: {manifest['split_strategy']}")
    print(f"Train: {manifest['train_total_rows']:,} rows ({len(manifest['train_captures'])} captures, {manifest['training_modalities']})")
    print(f"Val:   {manifest['val_total_rows']:,} rows ({len(manifest['val_captures'])} captures, {manifest['held_out_val_modalities']})")
    print(f"Test:  {manifest['test_total_rows']:,} rows ({len(manifest['test_captures'])} captures, {manifest['held_out_test_modalities']})")
