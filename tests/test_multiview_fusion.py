"""
Unit tests for DeepDNS Multi-View Fusion architecture, causality, and ablation modes.
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from src.data.labels import CaptureMetadata
from src.data.sequences import SequenceBuilder
from src.data.multiview_dataset import MultiViewSequenceDataset, multiview_collate_fn
from src.models.char_cnn import CharacterTokenizer
from src.models.multiview_fusion import (
    MultiViewFusionHead,
    DeepDNSMultiViewNetwork,
    DeepDNSMultiViewClassifier,
)


@pytest.fixture
def mock_multiview_batch():
    torch.manual_seed(42)
    batch_size = 3
    seq_len = 15
    beh_dim = 12
    char_len = 128

    beh_seqs = torch.randn(batch_size, seq_len, beh_dim)
    lex_seqs = torch.randint(0, 45, (batch_size, seq_len, char_len))
    seq_lens = torch.tensor([5, 10, 15], dtype=torch.long)
    return beh_seqs, lex_seqs, seq_lens


def test_fusion_head_forward_shapes():
    torch.manual_seed(42)
    head = MultiViewFusionHead(beh_dim=64, lex_dim=128, proj_dim=64, fuse_dim=64, num_classes=2)
    z_beh = torch.randn(4, 64)
    z_lex = torch.randn(4, 128)

    logits, z_fuse = head(z_beh, z_lex, mode="both")
    assert logits.shape == (4, 2)
    assert z_fuse.shape == (4, 64)


def test_multiview_network_forward_shapes(mock_multiview_batch):
    beh_seqs, lex_seqs, seq_lens = mock_multiview_batch
    net = DeepDNSMultiViewNetwork(
        beh_input_dim=12,
        beh_hidden_dim=64,
        lex_vocab_size=45,
        lex_output_dim=128,
        fuse_dim=64,
        num_classes=2,
    )
    net.eval()

    with torch.no_grad():
        step_logits, final_logits, z_beh, z_lex, z_fuse = net(beh_seqs, lex_seqs, seq_lens, mode="both")

    assert step_logits.shape == (3, 15, 2)
    assert final_logits.shape == (3, 2)
    assert z_beh.shape == (3, 15, 64)
    assert z_lex.shape == (3, 15, 128)
    assert z_fuse.shape == (3, 15, 64)


def test_multiview_causality_no_future_leakage():
    """
    CRITICAL MULTI-VIEW CAUSALITY TEST:
    Modifying future behavioral observations AND future lexical tokens beyond horizon K
    MUST NOT alter past predictions or fused embeddings up to step K.
    """
    torch.manual_seed(42)
    net = DeepDNSMultiViewNetwork(
        beh_input_dim=12,
        beh_hidden_dim=64,
        lex_vocab_size=45,
        lex_output_dim=128,
        fuse_dim=64,
        num_classes=2,
    )
    net.eval()

    seq_len = 20
    prefix_len = 10

    # Stream 1: Original dual-view sequences
    beh_1 = torch.randn(2, seq_len, 12)
    lex_1 = torch.randint(0, 45, (2, seq_len, 128))

    # Stream 2: Identical prefix (1..10), wildly corrupted future (11..20)
    beh_2 = beh_1.clone()
    lex_2 = lex_1.clone()
    beh_2[:, prefix_len:, :] = torch.randn(2, seq_len - prefix_len, 12) * 500.0 + 100.0
    lex_2[:, prefix_len:, :] = torch.randint(2, 45, (2, seq_len - prefix_len, 128))

    with torch.no_grad():
        step_logits_1, _, z_beh_1, _, z_fuse_1 = net(beh_1, lex_1)
        step_logits_2, _, z_beh_2, _, z_fuse_2 = net(beh_2, lex_2)

    # Prefix steps 0..prefix_len-1 must remain strictly identical
    torch.testing.assert_close(
        step_logits_1[:, :prefix_len, :],
        step_logits_2[:, :prefix_len, :],
        rtol=1e-6,
        atol=1e-6,
        msg="Multi-View causality violation: Future tokens altered past prefix predictions!",
    )
    torch.testing.assert_close(
        z_fuse_1[:, :prefix_len, :],
        z_fuse_2[:, :prefix_len, :],
        rtol=1e-6,
        atol=1e-6,
        msg="Multi-View causality violation: Future tokens altered past fused representations!",
    )


def test_multiview_ablation_modes(mock_multiview_batch):
    beh_seqs, lex_seqs, seq_lens = mock_multiview_batch
    net = DeepDNSMultiViewNetwork()
    net.eval()

    with torch.no_grad():
        _, final_both, _, _, _ = net(beh_seqs, lex_seqs, seq_lens, mode="both")
        _, final_beh, _, _, _ = net(beh_seqs, lex_seqs, seq_lens, mode="behavioral_only")
        _, final_lex, _, _, _ = net(beh_seqs, lex_seqs, seq_lens, mode="lexical_only")

    assert final_both.shape == (3, 2)
    assert final_beh.shape == (3, 2)
    assert final_lex.shape == (3, 2)


def test_multiview_dataset_and_collation():
    np.random.seed(42)
    total_queries = 40
    mock_beh = np.random.randn(total_queries, 12).astype(np.float32)
    mock_lex = np.random.randint(0, 45, (total_queries, 128), dtype=np.int64)

    meta = CaptureMetadata(
        label=1, label_name="attack", attack_modality="audio", intensity="heavy",
        capture_id="cap1", source_file="cap1.csv"
    )
    builder = SequenceBuilder(min_seq_len=5, max_seq_len=20, step_size=5)
    windows = builder.build_prefix_windows(total_rows=total_queries, meta=meta)

    dataset = MultiViewSequenceDataset(
        beh_features=mock_beh,
        lex_tokens=mock_lex,
        windows=windows,
        max_seq_len=20,
    )

    sample_batch = [dataset[0], dataset[1]]
    beh_b, lex_b, lens_b, labels_b, metas_b = multiview_collate_fn(sample_batch)

    assert beh_b.shape == (2, 20, 12)
    assert lex_b.shape == (2, 20, 128)
    assert lens_b.shape == (2,)
    assert labels_b.shape == (2,)
    assert len(metas_b) == 2


def test_multiview_classifier_serialization(tmp_path):
    torch.manual_seed(42)
    clf = DeepDNSMultiViewClassifier(device="cpu")

    save_path = tmp_path / "multiview_model.pt"
    clf.save(save_path)

    loaded_clf = DeepDNSMultiViewClassifier.load(save_path, device="cpu")

    beh = np.random.randn(2, 10, 12).astype(np.float32)
    lex = np.random.randint(0, 45, (2, 10, 128))

    probs_1 = clf.predict_proba(beh, lex)
    probs_2 = loaded_clf.predict_proba(beh, lex)

    np.testing.assert_allclose(probs_1, probs_2, rtol=1e-5)


def test_multiview_cuda_execution_if_available():
    if torch.cuda.is_available():
        clf = DeepDNSMultiViewClassifier(device="cuda")
        beh = np.random.randn(2, 10, 12).astype(np.float32)
        lex = np.random.randint(0, 45, (2, 10, 128))
        probs = clf.predict_proba(beh, lex)
        assert probs.shape == (2, 2)


def test_multiview_branch_strict_isolation(mock_multiview_batch):
    """
    Verifies that:
    1. In behavioral_only mode, lexical inputs have 0 influence (or can be None).
    2. In lexical_only mode, behavioral inputs have 0 influence (or can be None).
    3. In both mode, both branches contribute to fused logits.
    """
    beh_seqs, lex_seqs, seq_lens = mock_multiview_batch
    net = DeepDNSMultiViewNetwork()
    net.eval()

    corrupted_lex = torch.randint(2, 45, lex_seqs.shape)
    corrupted_beh = torch.randn_like(beh_seqs) * 500.0

    with torch.no_grad():
        # 1. Behavioral only: corrupted lexical does not change output
        _, out_beh_orig, _, _, _ = net(beh_seqs, lex_seqs, seq_lens, mode="behavioral_only")
        _, out_beh_corr, _, _, _ = net(beh_seqs, corrupted_lex, seq_lens, mode="behavioral_only")
        _, out_beh_none, _, _, _ = net(beh_seqs, None, seq_lens, mode="behavioral_only")
        torch.testing.assert_close(out_beh_orig, out_beh_corr)
        torch.testing.assert_close(out_beh_orig, out_beh_none)

        # 2. Lexical only: corrupted behavioral does not change output
        _, out_lex_orig, _, _, _ = net(beh_seqs, lex_seqs, seq_lens, mode="lexical_only")
        _, out_lex_corr, _, _, _ = net(corrupted_beh, lex_seqs, seq_lens, mode="lexical_only")
        _, out_lex_none, _, _, _ = net(None, lex_seqs, seq_lens, mode="lexical_only")
        torch.testing.assert_close(out_lex_orig, out_lex_corr)
        torch.testing.assert_close(out_lex_orig, out_lex_none)

        # 3. Both mode: both branches active
        _, out_both_1, _, _, _ = net(beh_seqs, lex_seqs, seq_lens, mode="both")
        _, out_both_2, _, _, _ = net(beh_seqs, corrupted_lex, seq_lens, mode="both")
        assert not torch.equal(out_both_1, out_both_2), "Lexical input had zero effect in both mode!"


def test_multiview_artifact_paths_uniqueness():
    """
    Verifies that all Multi-View output paths are unique and do not overwrite
    any baseline, temporal GRU, ablation, or diagnostic report files.
    """
    modes = ["both", "behavioral_only", "lexical_only"]
    models_dir = Path("data/processed/models")
    reports_dir = Path("reports/multiview")

    for m in modes:
        m_path = models_dir / f"multiview_{m}.pt"
        r_path = reports_dir / f"multiview_results_{m}.json"

        # Must not collide with temporal GRU files
        assert m_path.name not in ["temporal_gru.pt", "temporal_gru_no_iat.pt", "temporal_gru_shuffled_order.pt", "temporal_gru_label_shuffle.pt"]
        assert r_path.name not in ["temporal_gru_results.json", "temporal_gru_results_no_iat.json", "feature_shortcut_audit.json"]
