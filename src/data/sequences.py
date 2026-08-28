"""
Sequential stream construction and causal prefix windowing for DeepDNS.

Builds strictly causal prefix sequences [x_1 ... x_k] for k in [K_min, K_max],
ensuring zero cross-capture leakage and supporting PyTorch DataLoader batching.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.labels import CaptureMetadata


DEFAULT_MIN_SEQ_LEN = 5
DEFAULT_MAX_SEQ_LEN = 30


@dataclass
class SequenceWindow:
    """Metadata and indices for a single causal observation sequence window."""
    capture_id: str
    start_idx: int
    end_idx: int
    seq_len: int
    label: int
    label_name: str
    attack_modality: Optional[str]
    intensity: str


class StreamingSequenceDataset(Dataset):
    """
    PyTorch Dataset representing causal prefix sequence windows for streaming DNS detection.
    
    Each item returns:
        - features: Tensor of shape (MAX_SEQ_LEN, feature_dim) [padded with 0s]
        - length: Integer actual sequence length k in [MIN_SEQ_LEN, MAX_SEQ_LEN]
        - label: Binary target label (0 or 1) as torch.float32 or torch.long
        - meta: Dictionary containing capture_id, modality, intensity, and indices.
    """

    def __init__(
        self,
        features: np.ndarray,
        windows: List[SequenceWindow],
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    ):
        """
        Args:
            features: 2D numpy array of shape (Total_Capture_Queries, Feature_Dim).
            windows: List of SequenceWindow objects referencing slices in features.
            max_seq_len: Maximum sequence horizon length for zero-padding.
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.windows = windows
        self.max_seq_len = max_seq_len
        self.feature_dim = features.shape[1] if features.ndim > 1 else 1

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, torch.Tensor, dict]:
        win = self.windows[idx]
        seq_slice = self.features[win.start_idx : win.end_idx]
        actual_len = win.seq_len

        # Pad sequence to max_seq_len with zeros if actual_len < max_seq_len
        if actual_len < self.max_seq_len:
            pad_tensor = torch.zeros(
                (self.max_seq_len - actual_len, self.feature_dim),
                dtype=torch.float32,
            )
            padded_seq = torch.cat([seq_slice, pad_tensor], dim=0)
        else:
            padded_seq = seq_slice[: self.max_seq_len]

        label_tensor = torch.tensor(win.label, dtype=torch.float32)

        meta = {
            "capture_id": win.capture_id,
            "start_idx": win.start_idx,
            "end_idx": win.end_idx,
            "seq_len": actual_len,
            "attack_modality": win.attack_modality,
            "intensity": win.intensity,
        }

        return padded_seq, actual_len, label_tensor, meta


class SequenceBuilder:
    """
    Constructs causal sequence windows from chronologically sorted capture features.
    """

    def __init__(
        self,
        min_seq_len: int = DEFAULT_MIN_SEQ_LEN,
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        step_size: int = 1,
    ):
        """
        Args:
            min_seq_len: Minimum evidence window length (default 5).
            max_seq_len: Maximum evidence window length (default 30).
            step_size: Stride between successive sliding sequence starts (default 1).
        """
        if min_seq_len < 1:
            raise ValueError("min_seq_len must be at least 1.")
        if max_seq_len < min_seq_len:
            raise ValueError("max_seq_len must be >= min_seq_len.")

        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.step_size = step_size

    def build_prefix_windows(
        self,
        total_rows: int,
        meta: CaptureMetadata,
        offset: int = 0,
    ) -> List[SequenceWindow]:
        """
        Generates expanding causal prefix windows [0...k] for k from min_seq_len to max_seq_len,
        followed by sliding windows across the capture stream.

        Args:
            total_rows: Number of query records in the single capture.
            meta: CaptureMetadata for the capture session.
            offset: Global row index offset if features are concatenated.

        Returns:
            List of SequenceWindow definitions.
        """
        windows: List[SequenceWindow] = []

        if total_rows < self.min_seq_len:
            # Not enough observations for minimal evidence window
            return windows

        # 1. Expanding Prefix Windows starting from t=0: [0...5], [0...6], ..., [0...K_max]
        max_k = min(self.max_seq_len, total_rows)
        for k in range(self.min_seq_len, max_k + 1):
            windows.append(
                SequenceWindow(
                    capture_id=meta.capture_id,
                    start_idx=offset,
                    end_idx=offset + k,
                    seq_len=k,
                    label=meta.label,
                    label_name=meta.label_name,
                    attack_modality=meta.attack_modality,
                    intensity=meta.intensity,
                )
            )

        # 2. Sliding Horizons for subsequent observations: [i ... i+K_max]
        for start_i in range(1, total_rows - self.min_seq_len + 1, self.step_size):
            end_i = min(start_i + self.max_seq_len, total_rows)
            k = end_i - start_i
            if k >= self.min_seq_len:
                windows.append(
                    SequenceWindow(
                        capture_id=meta.capture_id,
                        start_idx=offset + start_i,
                        end_idx=offset + end_i,
                        seq_len=k,
                        label=meta.label,
                        label_name=meta.label_name,
                        attack_modality=meta.attack_modality,
                        intensity=meta.intensity,
                    )
                )

        return windows
