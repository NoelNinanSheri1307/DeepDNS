"""
Leakage-safe session-grouped and temporal dataset splitting for DeepDNS.

Ensures that entire PCAP capture sessions remain isolated within single partitions
with zero cross-split capture leakage.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass

from src.data.labels import CaptureMetadata


@dataclass
class SplitManifest:
    """Detailed partition manifest for reproducible experiments."""
    train_captures: List[dict]
    val_captures: List[dict]
    test_captures: List[dict]
    train_total_rows: int
    val_total_rows: int
    test_total_rows: int
    train_row_pct: float
    val_row_pct: float
    test_row_pct: float
    split_strategy: str

    def to_dict(self) -> dict:
        return {
            "split_strategy": self.split_strategy,
            "train_row_pct": self.train_row_pct,
            "val_row_pct": self.val_row_pct,
            "test_row_pct": self.test_row_pct,
            "train_total_rows": self.train_total_rows,
            "val_total_rows": self.val_total_rows,
            "test_total_rows": self.test_total_rows,
            "train_captures": self.train_captures,
            "val_captures": self.val_captures,
            "test_captures": self.test_captures,
        }

    def verify_zero_leakage(self) -> bool:
        """Verifies that no capture_id appears in multiple splits."""
        train_ids = {c["capture_id"] for c in self.train_captures}
        val_ids = {c["capture_id"] for c in self.val_captures}
        test_ids = {c["capture_id"] for c in self.test_captures}

        train_val = train_ids.intersection(val_ids)
        train_test = train_ids.intersection(test_ids)
        val_test = val_ids.intersection(test_ids)

        if train_val or train_test or val_test:
            raise RuntimeError(
                f"Data Leakage Detected in Split Manifest!\n"
                f"Train/Val overlap: {train_val}\n"
                f"Train/Test overlap: {train_test}\n"
                f"Val/Test overlap: {val_test}"
            )
        return True


class GroupedSplitter:
    """
    Partitions PCAP capture sessions into Train (~70%), Validation (~15%), and Test (~15%)
    while balancing attack modalities and traffic intensities across splits.
    """

    def __init__(self, target_train_pct: float = 0.70, target_val_pct: float = 0.15):
        self.target_train_pct = target_train_pct
        self.target_val_pct = target_val_pct

    def create_canonical_split(self, inventory_records: List[dict]) -> SplitManifest:
        """
        Creates a deterministic, balanced partition across all 18 CIC-Bell stateless captures.

        Stratification criteria:
        - Validation receives 1 Heavy Attack, 1 Light Attack, and 1 Benign session.
        - Test receives 1 Heavy Attack, 1 Light Attack, and 1 Benign session (including low-volume light exfil).
        - Train receives remaining 12 captures (~70% of volume).
        """
        # Filter only stateless CIC-Bell records
        stateless_items = [
            r for r in inventory_records
            if r.get("dataset_source") == "cic_bell_dns_exf_2021"
            and r.get("state_type") == "stateless"
        ]

        if not stateless_items:
            raise ValueError("No stateless CIC-Bell captures found in inventory.")

        # Fixed deterministic assignment strategy for standard 18 captures
        # Ensuring modality diversity in val/test:
        # Test: heavy_video (attack heavy), light_text (attack light), benign_2 (benign standard)
        # Val: heavy_image (attack heavy), light_audio (attack light), benign_heavy_2 (benign heavy)
        # Train: all remaining 12 captures

        test_capture_ids = {"heavy_video", "light_text", "benign_2"}
        val_capture_ids = {"heavy_image", "light_audio", "benign_heavy_2"}

        train_list = []
        val_list = []
        test_list = []

        for item in stateless_items:
            cid = item.get("capture_id")
            if cid in test_capture_ids:
                test_list.append(item)
            elif cid in val_capture_ids:
                val_list.append(item)
            else:
                train_list.append(item)

        total_rows = sum(r.get("row_count", 0) for r in stateless_items)
        train_rows = sum(r.get("row_count", 0) for r in train_list)
        val_rows = sum(r.get("row_count", 0) for r in val_list)
        test_rows = sum(r.get("row_count", 0) for r in test_list)

        manifest = SplitManifest(
            split_strategy="Session_Grouped_Stratified_PCAP",
            train_captures=train_list,
            val_captures=val_list,
            test_captures=test_list,
            train_total_rows=train_rows,
            val_total_rows=val_rows,
            test_total_rows=test_rows,
            train_row_pct=round(train_rows / total_rows * 100, 2) if total_rows > 0 else 0.0,
            val_row_pct=round(val_rows / total_rows * 100, 2) if total_rows > 0 else 0.0,
            test_row_pct=round(test_rows / total_rows * 100, 2) if total_rows > 0 else 0.0,
        )

        manifest.verify_zero_leakage()
        return manifest

    def save_manifest(self, manifest: SplitManifest, output_path: Union[str, Path]) -> None:
        """Saves split manifest to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)
