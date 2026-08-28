"""
Unit tests for Leave-One-Modality-Out (LOMO) OOD partitioning, evaluation unit audit, and zero-shot compatibility.
"""

import pytest
import json
from pathlib import Path
import numpy as np
import torch

from src.data.ood_splits import build_lomo_split_manifest
from src.data.labels import parse_cic_bell_label
from src.data.sequences import SequenceBuilder
from src.models.char_cnn import CharacterTokenizer
from src.models.multiview_fusion import DeepDNSMultiViewNetwork, DeepDNSMultiViewClassifier


def test_lomo_split_manifest_generation():
    manifest = build_lomo_split_manifest()
    assert manifest["split_strategy"] == "Leave_One_Modality_Out_Capture_Grouped"
    assert manifest["held_out_test_modalities"] == ["video", "text"]
    assert manifest["held_out_val_modalities"] == ["image"]
    assert set(manifest["training_modalities"]) == {"audio", "compressed", "exe"}

    train_ids = set(c["capture_id"] for c in manifest["train_captures"])
    val_ids = set(c["capture_id"] for c in manifest["val_captures"])
    test_ids = set(c["capture_id"] for c in manifest["test_captures"])

    # Zero capture overlap
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0


def test_lomo_modality_zero_leakage():
    manifest = build_lomo_split_manifest()
    train_mods = set(c["attack_modality"] for c in manifest["train_captures"] if c["attack_modality"])
    test_mods = set(c["attack_modality"] for c in manifest["test_captures"] if c["attack_modality"])

    assert len(train_mods.intersection(test_mods)) == 0, "Modality leakage detected between train and test!"


def test_zero_shot_domain_tokenization_compatibility():
    tokenizer = CharacterTokenizer()
    sample_dga_domains = [
        "xujozlat.ru",
        "myomgeocygocsouc.org",
        "skqgowoggookqigy.org",
        "normal-benign-domain.com",
    ]
    tokens = tokenizer.batch_encode(sample_dga_domains)
    assert tokens.shape == (4, 128)
    assert np.all(tokens >= 0) and np.all(tokens < 45)


def test_ood_model_initialization_and_forward():
    net = DeepDNSMultiViewNetwork()
    net.eval()
    beh = torch.randn(2, 30, 12)
    lex = torch.randint(0, 45, (2, 30, 128))

    with torch.no_grad():
        step_logits, final_logits, _, _, _ = net(beh, lex, mode="both")
        assert step_logits.shape == (2, 30, 2)
        assert final_logits.shape == (2, 2)
