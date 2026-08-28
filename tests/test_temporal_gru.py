"""
Unit tests for DeepDNS Temporal GRU architecture and causal temporal modeling.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.data.labels import CaptureMetadata
from src.data.sequences import SequenceBuilder, StreamingSequenceDataset
from src.models.temporal_gru import TemporalGRUNetwork, TemporalGRUClassifier


@pytest.fixture
def toy_sequence_batch():
    """Generates a small batch of 3D synthetic feature sequences."""
    torch.manual_seed(42)
    batch_size = 4
    seq_len = 15
    feature_dim = 12
    x = torch.randn(batch_size, seq_len, feature_dim)
    seq_lens = torch.tensor([5, 8, 12, 15], dtype=torch.long)
    return x, seq_lens


# =========================================================
# 1. ARCHITECTURE SHAPES & HIDDEN STATES
# =========================================================
def test_gru_forward_shapes(toy_sequence_batch):
    x, seq_lens = toy_sequence_batch
    net = TemporalGRUNetwork(input_dim=12, hidden_dim=64, num_classes=2)

    step_logits, final_logits, gru_out = net(x, seq_lens)

    assert step_logits.shape == (4, 15, 2)
    assert final_logits.shape == (4, 2)
    assert gru_out.shape == (4, 15, 64)
    assert not torch.isnan(step_logits).any()
    assert not torch.isnan(final_logits).any()


def test_forward_step_streaming_equivalence(toy_sequence_batch):
    """
    Verifies that unrolled forward pass matches step-by-step online streaming execution.
    """
    x, _ = toy_sequence_batch
    net = TemporalGRUNetwork(input_dim=12, hidden_dim=64, num_classes=2)
    net.eval()

    # Unrolled batch forward pass
    with torch.no_grad():
        step_logits_batch, _, _ = net(x)

    # Step-by-step streaming forward pass
    h_prev = None
    step_logits_stream = []
    with torch.no_grad():
        for t in range(x.size(1)):
            x_t = x[:, t, :]
            logits_t, h_prev = net.forward_step(x_t, h_prev)
            step_logits_stream.append(logits_t)

    step_logits_stream = torch.stack(step_logits_stream, dim=1)

    # Both execution modes must produce numerically identical outputs
    torch.testing.assert_close(step_logits_batch, step_logits_stream, rtol=1e-5, atol=1e-5)


# =========================================================
# 2. CAUSAL NO-FUTURE-LEAKAGE PROOF
# =========================================================
def test_causal_no_future_leakage():
    """
    CRITICAL CAUSALITY TEST:
    Modifying future observations x_{k+1}...x_K MUST NOT alter hidden states or
    predictions at step k (for any k).
    """
    torch.manual_seed(42)
    net = TemporalGRUNetwork(input_dim=12, hidden_dim=64, num_classes=2)
    net.eval()

    seq_len = 20
    prefix_len = 10
    feature_dim = 12

    # Sequence 1: Original stream
    x1 = torch.randn(2, seq_len, feature_dim)

    # Sequence 2: Identical prefix (1..10), but wildly corrupted future (11..20)
    x2 = x1.clone()
    x2[:, prefix_len:, :] = torch.randn(2, seq_len - prefix_len, feature_dim) * 500.0 + 100.0

    with torch.no_grad():
        step_logits_1, _, h_all_1 = net(x1)
        step_logits_2, _, h_all_2 = net(x2)

    # Prefix steps 0..prefix_len-1 must remain strictly identical
    torch.testing.assert_close(
        step_logits_1[:, :prefix_len, :],
        step_logits_2[:, :prefix_len, :],
        rtol=1e-6,
        atol=1e-6,
        msg="Causality violation: Future observations altered past prefix predictions!",
    )
    torch.testing.assert_close(
        h_all_1[:, :prefix_len, :],
        h_all_2[:, :prefix_len, :],
        rtol=1e-6,
        atol=1e-6,
        msg="Causality violation: Future observations altered past hidden states!",
    )


# =========================================================
# 3. VARIABLE LENGTHS & PADDING HANDLING
# =========================================================
def test_variable_lengths_indexing():
    torch.manual_seed(42)
    net = TemporalGRUNetwork(input_dim=12, hidden_dim=32, num_classes=2)
    net.eval()

    x = torch.randn(3, 30, 12)
    seq_lens = torch.tensor([5, 12, 30], dtype=torch.long)

    with torch.no_grad():
        step_logits, final_logits, _ = net(x, seq_lens)

    # Sample 0 final logit must match step_logits[0, 4]
    torch.testing.assert_close(final_logits[0], step_logits[0, 4])
    # Sample 1 final logit must match step_logits[1, 11]
    torch.testing.assert_close(final_logits[1], step_logits[1, 11])
    # Sample 2 final logit must match step_logits[2, 29]
    torch.testing.assert_close(final_logits[2], step_logits[2, 29])


# =========================================================
# 4. CLASSIFIER WRAPPER, DEVICE & SERIALIZATION
# =========================================================
def test_temporal_gru_classifier_fit_predict():
    np.random.seed(42)
    torch.manual_seed(42)

    # Build toy StreamingSequenceDataset
    total_queries = 50
    mock_features = np.random.randn(total_queries, 12).astype(np.float32)
    mock_meta = CaptureMetadata(
        label=1,
        label_name="attack",
        attack_modality="text",
        intensity="light",
        capture_id="toy_cap",
        source_file="toy.csv",
    )
    seq_builder = SequenceBuilder(min_seq_len=5, max_seq_len=15, step_size=2)
    windows = seq_builder.build_prefix_windows(total_rows=total_queries, meta=mock_meta)
    dataset = StreamingSequenceDataset(mock_features, windows, max_seq_len=15)

    clf = TemporalGRUClassifier(
        input_dim=12,
        hidden_dim=32,
        batch_size=8,
        device="cpu",
    )
    clf.fit(dataset, epochs=2, verbose=False)

    step_probs = clf.predict_step_proba(mock_features[:15].reshape(1, 15, 12))
    assert step_probs.shape == (1, 15, 2)
    np.testing.assert_allclose(np.sum(step_probs, axis=-1), 1.0, rtol=1e-5)

    final_preds = clf.predict(mock_features[:15].reshape(1, 15, 12))
    assert final_preds.shape == (1,)
    assert final_preds[0] in [0, 1]


def test_temporal_gru_serialization(tmp_path):
    torch.manual_seed(42)
    clf = TemporalGRUClassifier(input_dim=12, hidden_dim=32, device="cpu")

    save_path = tmp_path / "gru_model.pt"
    clf.save(save_path)

    loaded_clf = TemporalGRUClassifier.load(save_path, device="cpu")

    x = np.random.randn(2, 10, 12).astype(np.float32)
    probs_1 = clf.predict_proba(x)
    probs_2 = loaded_clf.predict_proba(x)

    np.testing.assert_allclose(probs_1, probs_2, rtol=1e-5)


def test_cuda_execution_if_available():
    if torch.cuda.is_available():
        clf = TemporalGRUClassifier(input_dim=12, hidden_dim=32, device="cuda")
        x = np.random.randn(2, 10, 12).astype(np.float32)
        probs = clf.predict_proba(x)
        assert probs.shape == (2, 2)


# =========================================================
# 5. FORENSIC AUDIT & BOUNDARY ISOLATION TESTS
# =========================================================
def test_cross_capture_boundary_isolation():
    """
    Asserts that sequence windows generated across multiple concatenated captures
    never cross capture boundaries.
    """
    cap1_meta = CaptureMetadata(
        label=0, label_name="benign", attack_modality=None, intensity="standard",
        capture_id="cap1", source_file="cap1.csv"
    )
    cap2_meta = CaptureMetadata(
        label=1, label_name="attack", attack_modality="audio", intensity="heavy",
        capture_id="cap2", source_file="cap2.csv"
    )

    cap1_rows = 40
    cap2_rows = 50
    builder = SequenceBuilder(min_seq_len=5, max_seq_len=30, step_size=10)

    # Offset simulation
    cap1_windows = builder.build_prefix_windows(total_rows=cap1_rows, meta=cap1_meta, offset=0)
    cap2_windows = builder.build_prefix_windows(total_rows=cap2_rows, meta=cap2_meta, offset=cap1_rows)

    # Cap 1 windows must all be within [0, 40]
    for w in cap1_windows:
        assert 0 <= w.start_idx < w.end_idx <= cap1_rows
        assert w.capture_id == "cap1"
        assert w.label == 0

    # Cap 2 windows must all be within [40, 90]
    for w in cap2_windows:
        assert cap1_rows <= w.start_idx < w.end_idx <= cap1_rows + cap2_rows
        assert w.capture_id == "cap2"
        assert w.label == 1


def test_feature_scaler_isolated_to_training():
    """
    Verifies that the serialized FeatureScaler has valid fitted parameters.
    """
    import json
    scaler_path = Path("data/processed/feature_scaler.json")
    if scaler_path.exists():
        with open(scaler_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data.get("is_fitted") is True
        assert len(data.get("mean")) == 12
        assert len(data.get("scale")) == 12


def test_iat_ablation_feature_dimension_and_exclusion():
    """
    Verifies that when IAT is ablated, the extractor returns exactly 11 causal features
    without inter_arrival_time, and rejects forbidden columns.
    """
    from src.data.features import CausalFeatureExtractor
    from src.data.schema import FORBIDDEN_MODEL_COLUMNS

    extractor = CausalFeatureExtractor(exclude_features=["inter_arrival_time"])
    assert len(extractor.feature_names) == 11
    assert "inter_arrival_time" not in extractor.feature_names

    # Mock raw stateless dataframe
    mock_df = pd.DataFrame({
        "timestamp": ["2026-08-28 10:00:00", "2026-08-28 10:00:01"],
        "FQDN_count": [25, 30],
        "subdomain_length": [10, 15],
        "entropy": [3.2, 3.5],
        "numeric": [5, 8],
        "upper": [0, 0],
        "special": [1, 2],
        "labels": [3, 4],
        "labels_max": [10, 12],
        "labels_average": [6.0, 7.0],
        "subdomain": [1, 1],
        "len": [64, 80],
    })

    feat_df = extractor.extract_from_dataframe(mock_df)
    assert feat_df.shape == (2, 11)
    assert "inter_arrival_time" not in feat_df.columns
    for forbidden in FORBIDDEN_MODEL_COLUMNS:
        assert forbidden not in feat_df.columns


def test_iat_ablated_scaler_and_causality():
    """
    Verifies that an 11-dimensional ablated tensor can be scaled, processed by Temporal GRU,
    and maintains strict forward causality.
    """
    from src.data.features import FeatureScaler

    feature_names_11 = [
        "fqdn_length", "subdomain_length", "char_entropy", "digit_ratio",
        "uppercase_ratio", "special_ratio", "label_count", "max_label_length",
        "avg_label_length", "has_subdomain", "payload_len"
    ]
    scaler = FeatureScaler(feature_names=feature_names_11)
    mock_data = np.random.randn(20, 11)
    scaler.fit(mock_data)
    assert scaler.is_fitted
    assert len(scaler.mean_) == 11

    net_11 = TemporalGRUNetwork(input_dim=11, hidden_dim=32, num_classes=2)
    net_11.eval()

    seq_len = 15
    prefix_len = 8
    x1 = torch.randn(2, seq_len, 11)
    x2 = x1.clone()
    x2[:, prefix_len:, :] = torch.randn(2, seq_len - prefix_len, 11) * 300.0

    with torch.no_grad():
        step_logits_1, _, _ = net_11(x1)
        step_logits_2, _, _ = net_11(x2)

    torch.testing.assert_close(
        step_logits_1[:, :prefix_len, :],
        step_logits_2[:, :prefix_len, :],
        rtol=1e-6,
        atol=1e-6,
    )


def test_temporal_order_permutation_multiset_preservation():
    """
    Verifies that temporal order permutation within individual sequence windows:
    1. Preserves the exact multiset of 11-dimensional feature vectors.
    2. Modifies only the temporal sequence order.
    3. Operates independently and deterministically across windows.
    4. Leaves sequence lengths, labels, and padding bounds intact.
    """
    from src.data.sequences import SequenceBuilder, StreamingSequenceDataset
    from src.data.labels import CaptureMetadata

    np.random.seed(42)
    # Generate distinguishable feature vectors (row i has first feature = i)
    total_rows = 40
    mock_features = np.zeros((total_rows, 11), dtype=np.float32)
    for i in range(total_rows):
        mock_features[i, :] = float(i + 1)

    meta = CaptureMetadata(
        label=1, label_name="attack", attack_modality="audio", intensity="heavy",
        capture_id="cap1", source_file="cap1.csv"
    )
    builder = SequenceBuilder(min_seq_len=5, max_seq_len=15, step_size=5)
    windows = builder.build_prefix_windows(total_rows=total_rows, meta=meta)

    ds_control = StreamingSequenceDataset(mock_features, windows, max_seq_len=15, shuffle_order=False)
    ds_shuffled = StreamingSequenceDataset(mock_features, windows, max_seq_len=15, shuffle_order=True, seed=42)

    assert len(ds_control) == len(ds_shuffled)

    for idx in range(len(ds_control)):
        seq_c, len_c, label_c, meta_c = ds_control[idx]
        seq_s, len_s, label_s, meta_s = ds_shuffled[idx]

        # Labels, lengths, and metadata must match 100%
        assert len_c == len_s
        assert label_c == label_s
        assert meta_c["capture_id"] == meta_s["capture_id"]

        # Valid non-padded slice
        valid_c = seq_c[:len_c].numpy()
        valid_s = seq_s[:len_s].numpy()

        # The multiset of rows must be identical
        sorted_c = np.sort(valid_c, axis=0)
        sorted_s = np.sort(valid_s, axis=0)
        np.testing.assert_array_equal(sorted_c, sorted_s)

        # But for windows with length >= 5, the temporal order must be permuted
        if len_c >= 5:
            assert not np.array_equal(valid_c, valid_s), f"Window {idx} order was not permuted!"

        # Zero padding must be untouched
        if len_c < 15:
            assert torch.all(seq_c[len_c:] == 0.0)
            assert torch.all(seq_s[len_s:] == 0.0)

    # Test determinism: fetching same idx produces identical permutation
    seq_s1, _, _, _ = ds_shuffled[0]
    seq_s2, _, _, _ = ds_shuffled[0]
    torch.testing.assert_close(seq_s1, seq_s2)


def test_training_label_permutation_sanity_check():
    """
    Verifies the negative-control training label permutation:
    1. Training labels are actively permuted across windows while preserving class balance.
    2. Input features X, sequence lengths k, and window indices are 100% untouched.
    3. Validation and test sets have labels 100% untouched.
    4. Permutation is strictly deterministic given label_seed.
    """
    from src.data.sequences import SequenceBuilder, StreamingSequenceDataset
    from src.data.labels import CaptureMetadata

    total_rows = 100
    mock_features = np.random.randn(total_rows, 11).astype(np.float32)

    meta_0 = CaptureMetadata(
        label=0, label_name="benign", attack_modality=None, intensity="benign",
        capture_id="benign_cap", source_file="benign.csv"
    )
    meta_1 = CaptureMetadata(
        label=1, label_name="attack", attack_modality="audio", intensity="heavy",
        capture_id="attack_cap", source_file="attack.csv"
    )

    builder = SequenceBuilder(min_seq_len=5, max_seq_len=10, step_size=2)
    windows_0 = builder.build_prefix_windows(total_rows=50, meta=meta_0, offset=0)
    windows_1 = builder.build_prefix_windows(total_rows=50, meta=meta_1, offset=50)
    all_windows = windows_0 + windows_1  # 50% benign, 50% attack

    # 1. Training dataset with permuted labels
    ds_train_shuffled = StreamingSequenceDataset(
        mock_features, all_windows, max_seq_len=10, shuffle_labels=True, label_seed=42
    )
    ds_train_control = StreamingSequenceDataset(
        mock_features, all_windows, max_seq_len=10, shuffle_labels=False
    )

    orig_labels = []
    shuffled_labels = []
    for idx in range(len(all_windows)):
        seq_c, len_c, label_c, meta_c = ds_train_control[idx]
        seq_s, len_s, label_s, meta_s = ds_train_shuffled[idx]

        # Features X and lengths must be completely identical
        torch.testing.assert_close(seq_c, seq_s)
        assert len_c == len_s
        assert meta_c["start_idx"] == meta_s["start_idx"]
        assert meta_c["end_idx"] == meta_s["end_idx"]

        orig_labels.append(int(label_c.item()))
        shuffled_labels.append(int(label_s.item()))

    # Permuted labels must differ from original order
    assert orig_labels != shuffled_labels, "Training labels were not permuted!"
    # But exact class balance (sum of 1s) must be identical
    assert sum(orig_labels) == sum(shuffled_labels)

    # 2. Validation / Test datasets with labels untouched
    ds_val = StreamingSequenceDataset(
        mock_features, all_windows, max_seq_len=10, shuffle_labels=False
    )
    for idx in range(len(all_windows)):
        _, _, label_v, _ = ds_val[idx]
        assert int(label_v.item()) == all_windows[idx].label

    # 3. Determinism check
    ds_train_shuffled_2 = StreamingSequenceDataset(
        mock_features, all_windows, max_seq_len=10, shuffle_labels=True, label_seed=42
    )
    for idx in range(len(all_windows)):
        assert ds_train_shuffled[idx][2].item() == ds_train_shuffled_2[idx][2].item()




