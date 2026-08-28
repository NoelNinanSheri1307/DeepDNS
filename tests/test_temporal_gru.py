"""
Unit tests for DeepDNS Temporal GRU architecture and causal temporal modeling.
"""

import pytest
import numpy as np
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

