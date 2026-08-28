"""
Unit tests for causal sequence generation and Dataset windowing.
"""

import pytest
import numpy as np
import torch
from src.data.labels import CaptureMetadata
from src.data.sequences import SequenceBuilder, StreamingSequenceDataset


@pytest.fixture
def mock_capture_meta():
    return CaptureMetadata(
        label=1,
        label_name="attack",
        attack_modality="audio",
        intensity="heavy",
        capture_id="heavy_audio",
        source_file="stateless_features-heavy_audio.pcap.csv",
    )


def test_sequence_prefix_bounds(mock_capture_meta):
    builder = SequenceBuilder(min_seq_len=5, max_seq_len=10)
    windows = builder.build_prefix_windows(total_rows=15, meta=mock_capture_meta)

    # Initial prefixes should be lengths 5, 6, 7, 8, 9, 10
    prefix_lens = [w.seq_len for w in windows[:6]]
    assert prefix_lens == [5, 6, 7, 8, 9, 10]

    # Every window must satisfy min_seq_len <= len <= max_seq_len
    for w in windows:
        assert 5 <= w.seq_len <= 10
        assert w.capture_id == "heavy_audio"
        assert w.label == 1


def test_causal_no_future_leakage(mock_capture_meta):
    """Verifies that prefix at step k contains strictly features up to index k."""
    total_rows = 20
    feature_dim = 12
    # Create time-identifying features (row i has value i)
    mock_features = np.repeat(np.arange(total_rows)[:, None], feature_dim, axis=1).astype(np.float32)

    builder = SequenceBuilder(min_seq_len=5, max_seq_len=10)
    windows = builder.build_prefix_windows(total_rows=total_rows, meta=mock_capture_meta)

    dataset = StreamingSequenceDataset(features=mock_features, windows=windows, max_seq_len=10)

    for i, (seq, seq_len, label, meta) in enumerate(dataset):
        start_idx = meta["start_idx"]
        end_idx = meta["end_idx"]
        assert end_idx - start_idx == seq_len

        # Sliced sequence values must strictly match the range [start_idx, end_idx-1]
        expected_values = np.arange(start_idx, end_idx)
        actual_values = seq[:seq_len, 0].numpy()
        np.testing.assert_array_equal(actual_values, expected_values)


def test_dataset_padding(mock_capture_meta):
    mock_features = np.ones((10, 12), dtype=np.float32)
    builder = SequenceBuilder(min_seq_len=5, max_seq_len=30)
    windows = builder.build_prefix_windows(total_rows=10, meta=mock_capture_meta)

    dataset = StreamingSequenceDataset(features=mock_features, windows=windows, max_seq_len=30)
    padded_seq, seq_len, label, meta = dataset[0]

    # Length of first prefix is 5, but tensor must be padded to (30, 12)
    assert padded_seq.shape == (30, 12)
    assert seq_len == 5
    # Slices beyond seq_len must be zeros
    assert torch.all(padded_seq[5:] == 0.0)
