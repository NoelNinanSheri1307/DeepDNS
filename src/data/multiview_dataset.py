"""
Multi-View Sequential Dataset loader for DeepDNS.

Simultaneously yields:
1. Padded behavioral feature sequence tensors: (MAX_SEQ_LEN, 12)
2. Padded lexical character token sequence tensors: (MAX_SEQ_LEN, 128)
3. Horizon length k in [5, 30]
4. Target class label (0 or 1)
5. Session metadata
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.labels import parse_cic_bell_label, CaptureMetadata
from src.data.features import CausalFeatureExtractor, FeatureScaler
from src.data.sequences import SequenceWindow, SequenceBuilder, DEFAULT_MAX_SEQ_LEN
from src.models.char_cnn import CharacterTokenizer, DEFAULT_MAX_CHAR_LEN


class MultiViewSequenceDataset(Dataset):
    """
    PyTorch Dataset providing aligned Dual-View sequences (Behavioral + Lexical).
    """

    def __init__(
        self,
        beh_features: np.ndarray,
        lex_tokens: np.ndarray,
        windows: List[SequenceWindow],
        max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
        max_char_len: int = DEFAULT_MAX_CHAR_LEN,
    ):
        self.beh_features = torch.tensor(beh_features, dtype=torch.float32)
        self.lex_tokens = torch.tensor(lex_tokens, dtype=torch.long)
        self.windows = windows
        self.max_seq_len = max_seq_len
        self.max_char_len = max_char_len
        self.beh_dim = beh_features.shape[1] if beh_features.ndim > 1 else 12

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor, dict]:
        win = self.windows[idx]
        actual_len = win.seq_len

        beh_slice = self.beh_features[win.start_idx : win.end_idx]
        lex_slice = self.lex_tokens[win.start_idx : win.end_idx]

        # Pad behavioral sequence to max_seq_len (zeros)
        if actual_len < self.max_seq_len:
            beh_pad = torch.zeros((self.max_seq_len - actual_len, self.beh_dim), dtype=torch.float32)
            padded_beh = torch.cat([beh_slice, beh_pad], dim=0)

            # Pad lexical sequence to max_seq_len (pad_idx = 0)
            lex_pad = torch.zeros((self.max_seq_len - actual_len, self.max_char_len), dtype=torch.long)
            padded_lex = torch.cat([lex_slice, lex_pad], dim=0)
        else:
            padded_beh = beh_slice[: self.max_seq_len]
            padded_lex = lex_slice[: self.max_seq_len]

        label_tensor = torch.tensor(win.label, dtype=torch.long)

        meta = {
            "capture_id": win.capture_id,
            "start_idx": win.start_idx,
            "end_idx": win.end_idx,
            "seq_len": actual_len,
            "attack_modality": win.attack_modality or "none",
            "intensity": win.intensity,
        }

        return padded_beh, padded_lex, actual_len, label_tensor, meta


def multiview_collate_fn(batch: list) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list]:
    """
    Custom collate function for MultiViewSequenceDataset batches.
    """
    beh_seqs = torch.stack([item[0] for item in batch], dim=0)
    lex_seqs = torch.stack([item[1] for item in batch], dim=0)
    lens = torch.tensor([item[2] for item in batch], dtype=torch.long)
    labels = torch.stack([item[3] for item in batch], dim=0)
    metas = [item[4] for item in batch]
    return beh_seqs, lex_seqs, lens, labels, metas


def build_multiview_partition_dataset(
    captures_info: list,
    project_root: Path,
    extractor: CausalFeatureExtractor,
    scaler: FeatureScaler,
    tokenizer: CharacterTokenizer,
    seq_builder: SequenceBuilder,
    max_rows_per_capture: Optional[int] = None,
) -> MultiViewSequenceDataset:
    """
    Builds aligned MultiViewSequenceDataset across a list of capture metadata records.
    """
    all_beh = []
    all_lex = []
    all_windows = []
    current_offset = 0

    for cap in captures_info:
        p = project_root / cap["rel_path"]
        df_raw = pd.read_csv(p, low_memory=False, nrows=max_rows_per_capture)
        meta = parse_cic_bell_label(p)

        # 1. Behavioral features
        feat_df = extractor.extract_from_dataframe(df_raw)
        norm_beh = scaler.transform(feat_df)

        # 2. Lexical domain text encoding
        domain_series = df_raw["sld"].fillna("").astype(str).tolist() if "sld" in df_raw.columns else [""] * len(df_raw)
        lex_encoded = tokenizer.batch_encode(domain_series)

        # 3. Build sequence windows
        windows = seq_builder.build_prefix_windows(
            total_rows=len(norm_beh),
            meta=meta,
            offset=current_offset,
        )

        all_beh.append(norm_beh)
        all_lex.append(lex_encoded)
        all_windows.extend(windows)
        current_offset += len(norm_beh)

    beh_matrix = np.concatenate(all_beh, axis=0)
    lex_matrix = np.concatenate(all_lex, axis=0)

    dataset = MultiViewSequenceDataset(
        beh_features=beh_matrix,
        lex_tokens=lex_matrix,
        windows=all_windows,
        max_seq_len=seq_builder.max_seq_len,
        max_char_len=tokenizer.max_length,
    )
    return dataset
