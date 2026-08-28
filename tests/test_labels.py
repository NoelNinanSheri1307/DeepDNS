"""
Unit tests for label parsing and metadata extraction.
"""

import pytest
from pathlib import Path
from src.data.labels import parse_cic_bell_label, CaptureMetadata


def test_heavy_attack_labels():
    p = "data/raw/cic_bell_dns_exf_2021/Attack_Heavy_Benign/Attacks/stateless_features-heavy_audio.pcap.csv"
    meta = parse_cic_bell_label(p)
    assert meta.label == 1
    assert meta.label_name == "attack"
    assert meta.attack_modality == "audio"
    assert meta.intensity == "heavy"
    assert meta.capture_id == "heavy_audio"


def test_light_attack_labels():
    p = "data/raw/cic_bell_dns_exf_2021/Attack_Light_Benign/Attacks/stateless_features-light_text.pcap.csv"
    meta = parse_cic_bell_label(p)
    assert meta.label == 1
    assert meta.label_name == "attack"
    assert meta.attack_modality == "text"
    assert meta.intensity == "light"
    assert meta.capture_id == "light_text"


def test_benign_labels():
    p = "data/raw/cic_bell_dns_exf_2021/Benign/stateless_features-benign_1.pcap.csv"
    meta = parse_cic_bell_label(p)
    assert meta.label == 0
    assert meta.label_name == "benign"
    assert meta.attack_modality is None
    assert meta.intensity == "standard"
    assert meta.capture_id == "benign_1"


def test_heavy_benign_labels():
    p = "data/raw/cic_bell_dns_exf_2021/Attack_Heavy_Benign/Benign/stateless_features-benign_heavy_2.pcap.csv"
    meta = parse_cic_bell_label(p)
    assert meta.label == 0
    assert meta.label_name == "benign"
    assert meta.attack_modality is None
    assert meta.intensity == "heavy"
    assert meta.capture_id == "benign_heavy_2"


def test_invalid_path_raises_error():
    with pytest.raises(ValueError):
        parse_cic_bell_label("data/raw/unknown_folder/sample.csv")
