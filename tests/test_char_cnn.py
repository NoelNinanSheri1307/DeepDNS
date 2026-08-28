"""
Unit tests for Character-level Lexical Encoder (Character-CNN).
"""

import pytest
import numpy as np
import torch
from pathlib import Path

from src.models.char_cnn import (
    CharacterTokenizer,
    CharacterCNNEncoder,
    CharacterCNNClassifier,
    DEFAULT_MAX_CHAR_LEN,
)


def test_character_tokenizer_determinism():
    tokenizer = CharacterTokenizer(max_length=32)
    domain = "sub1.malicious-tunnel.example.com"
    tokens1 = tokenizer.encode(domain)
    tokens2 = tokenizer.encode(domain)

    np.testing.assert_array_equal(tokens1, tokens2)
    assert len(tokens1) == 32
    assert tokens1.dtype == np.int64


def test_tokenizer_unknown_and_padding_tokens():
    tokenizer = CharacterTokenizer(max_length=20)
    domain = "abc#$%"  # '#' and '$' and '%' are outside standard DNS charset -> map to <unk> (1)
    tokens = tokenizer.encode(domain)

    assert tokens[0] == tokenizer.char_to_idx["a"]
    assert tokens[1] == tokenizer.char_to_idx["b"]
    assert tokens[2] == tokenizer.char_to_idx["c"]
    assert tokens[3] == tokenizer.unk_idx
    assert tokens[4] == tokenizer.unk_idx
    assert tokens[5] == tokenizer.unk_idx
    # Slices beyond length 6 must be padding (0)
    assert np.all(tokens[6:] == tokenizer.pad_idx)


def test_char_cnn_encoder_forward_shapes():
    torch.manual_seed(42)
    encoder = CharacterCNNEncoder(
        vocab_size=45,
        embedding_dim=32,
        num_filters=32,
        kernel_sizes=(3, 5, 7),
        output_dim=128,
        num_classes=2,
    )
    encoder.eval()

    # 1. Test 2D Single-Query Tensor: (Batch_Size=4, Length=128)
    x_2d = torch.randint(0, 45, (4, 128))
    with torch.no_grad():
        logits_2d, z_lex_2d = encoder(x_2d)

    assert logits_2d.shape == (4, 2)
    assert z_lex_2d.shape == (4, 128)

    # 2. Test 3D Sequential Tensor: (Batch_Size=4, Horizon=10, Length=128)
    x_3d = torch.randint(0, 45, (4, 10, 128))
    with torch.no_grad():
        logits_3d, z_lex_3d = encoder(x_3d)

    assert logits_3d.shape == (4, 10, 2)
    assert z_lex_3d.shape == (4, 10, 128)


def test_char_cnn_classifier_fit_predict():
    torch.manual_seed(42)
    domains = [
        "ns1.google.com",
        "tunnel.exfil-01.attacker.net",
        "api.github.com",
        "chunk123.payload.c2server.org",
    ]
    labels = np.array([0, 1, 0, 1], dtype=int)

    clf = CharacterCNNClassifier(
        vocab_size=45,
        embedding_dim=16,
        num_filters=16,
        output_dim=64,
        batch_size=2,
        device="cpu",
    )
    clf.fit(domains, labels, epochs=2, verbose=False)

    probs = clf.predict_proba(domains)
    assert probs.shape == (4, 2)
    np.testing.assert_allclose(np.sum(probs, axis=-1), 1.0, rtol=1e-5)

    preds = clf.predict(domains)
    assert preds.shape == (4,)
    assert set(preds).issubset({0, 1})


def test_char_cnn_serialization(tmp_path):
    torch.manual_seed(42)
    clf = CharacterCNNClassifier(
        vocab_size=45,
        embedding_dim=16,
        num_filters=16,
        output_dim=64,
        device="cpu",
    )

    save_path = tmp_path / "char_cnn.pt"
    clf.save(save_path)

    loaded_clf = CharacterCNNClassifier.load(save_path, device="cpu")

    domains = ["test.domain.com", "random.query.org"]
    p1 = clf.predict_proba(domains)
    p2 = loaded_clf.predict_proba(domains)

    np.testing.assert_allclose(p1, p2, rtol=1e-5)
