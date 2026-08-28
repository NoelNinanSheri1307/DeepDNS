"""
Unit tests for grouped splitting, zero-leakage enforcement, and lexical tokenization.
"""

import pytest
import numpy as np
from src.data.splits import GroupedSplitter, SplitManifest
from src.data.lexical import DNSThreatsLexicalTokenizer


@pytest.fixture
def mock_inventory():
    """Generates mock inventory records covering multiple captures."""
    captures = [
        ("heavy_audio", "Attack_Heavy_Benign/Attacks/stateless_features-heavy_audio.pcap.csv", 100, "audio", "heavy", 1),
        ("heavy_compressed", "Attack_Heavy_Benign/Attacks/stateless_features-heavy_compressed.pcap.csv", 100, "compressed", "heavy", 1),
        ("heavy_video", "Attack_Heavy_Benign/Attacks/stateless_features-heavy_video.pcap.csv", 100, "video", "heavy", 1),
        ("heavy_image", "Attack_Heavy_Benign/Attacks/stateless_features-heavy_image.pcap.csv", 100, "image", "heavy", 1),
        ("light_audio", "Attack_Light_Benign/Attacks/stateless_features-light_audio.pcap.csv", 50, "audio", "light", 1),
        ("light_text", "Attack_Light_Benign/Attacks/stateless_features-light_text.pcap.csv", 50, "text", "light", 1),
        ("benign_1", "Benign/stateless_features-benign_1.pcap.csv", 200, None, "standard", 0),
        ("benign_2", "Benign/stateless_features-benign_2.pcap.csv", 150, None, "standard", 0),
        ("benign_heavy_1", "Attack_Heavy_Benign/Benign/stateless_features-benign_heavy_1.pcap.csv", 150, None, "heavy", 0),
        ("benign_heavy_2", "Attack_Heavy_Benign/Benign/stateless_features-benign_heavy_2.pcap.csv", 150, None, "heavy", 0),
    ]

    records = []
    for cid, path, rows, mod, intens, lbl in captures:
        records.append({
            "capture_id": cid,
            "filename": path.split("/")[-1],
            "rel_path": path,
            "dataset_source": "cic_bell_dns_exf_2021",
            "state_type": "stateless",
            "row_count": rows,
            "attack_modality": mod,
            "intensity": intens,
            "label": lbl,
        })
    return records


def test_grouped_split_zero_leakage(mock_inventory):
    splitter = GroupedSplitter()
    manifest = splitter.create_canonical_split(mock_inventory)

    assert manifest.verify_zero_leakage() is True
    assert len(manifest.train_captures) > 0
    assert len(manifest.val_captures) > 0
    assert len(manifest.test_captures) > 0

    train_cids = {c["capture_id"] for c in manifest.train_captures}
    val_cids = {c["capture_id"] for c in manifest.val_captures}
    test_cids = {c["capture_id"] for c in manifest.test_captures}

    assert len(train_cids.intersection(val_cids)) == 0
    assert len(train_cids.intersection(test_cids)) == 0
    assert len(val_cids.intersection(test_cids)) == 0


def test_leakage_detector_catches_overlap():
    bad_manifest = SplitManifest(
        split_strategy="Broken_Test",
        train_captures=[{"capture_id": "heavy_audio"}],
        val_captures=[{"capture_id": "heavy_audio"}],
        test_captures=[{"capture_id": "benign_1"}],
        train_total_rows=100,
        val_total_rows=100,
        test_total_rows=200,
        train_row_pct=25.0,
        val_row_pct=25.0,
        test_row_pct=50.0,
    )
    with pytest.raises(RuntimeError):
        bad_manifest.verify_zero_leakage()


def test_lexical_character_tokenizer():
    tokenizer = DNSThreatsLexicalTokenizer(max_length=32)
    domain = "test-01.tunnel.com"
    tokens = tokenizer.encode_domain(domain)

    assert tokens.shape == (32,)
    # Verify non-zero tokens count equals length of domain
    assert np.count_nonzero(tokens) == len(domain)

    decoded = tokenizer.decode_tokens(tokens)
    assert decoded == domain

    # Test unknown character fallback
    domain_with_special = "test§weird.com"
    tokens_unk = tokenizer.encode_domain(domain_with_special)
    # The unknown character should map to index 1 (<unk>)
    assert tokenizer.unk_idx in tokens_unk
