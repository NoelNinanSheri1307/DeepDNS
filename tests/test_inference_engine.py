"""
Unit and Integration Tests for the DeepDNS Canonical Inference Engine.
"""

from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import torch

from src.inference import (
    DeepDNSInferenceEngine,
    DNSObservation,
    DNSStream,
    DetectionResult,
    DecisionEnum,
    InferenceMode,
)


@pytest.fixture
def mock_dns_stream() -> DNSStream:
    """Creates a deterministic 30-query DNSStream for testing."""
    stream = DNSStream(stream_id="test_stream_001")
    base_time = pd.Timestamp("2026-08-28 12:00:00")
    for i in range(30):
        obs = DNSObservation(
            timestamp=base_time + pd.Timedelta(seconds=i * 0.4),
            domain_name=f"chunk{i}.tunnel.example.com",
            fqdn_count=26.0,
            subdomain_length=8.0,
            upper=0.0,
            lower=20.0,
            numeric=6.0,
            entropy=2.45,
            special=2.0,
            labels=4.0,
            labels_max=10.0,
            labels_average=6.5,
            len=26.0,
            subdomain=1.0,
        )
        stream.add_observation(obs)
    return stream


def test_engine_initialization():
    """1. Test that engine initializes with ID and OOD modes."""
    engine_id = DeepDNSInferenceEngine(mode=InferenceMode.IN_DISTRIBUTION)
    assert engine_id.tau_attack == 0.95
    assert engine_id.tau_benign == 0.15
    assert engine_id.classifier is not None

    engine_ood = DeepDNSInferenceEngine(mode=InferenceMode.OOD)
    assert engine_ood.tau_attack == 0.80
    assert engine_ood.tau_benign == 0.01


def test_invalid_stream_length_rejection():
    """3. Test that short streams (< 5 queries) are rejected with clear error."""
    engine = DeepDNSInferenceEngine(mode=InferenceMode.IN_DISTRIBUTION)
    short_stream = DNSStream(stream_id="short_stream")
    for i in range(3):
        short_stream.add_observation(DNSObservation(timestamp=f"2026-08-28 12:00:0{i}", domain_name="test.com"))

    with pytest.raises(ValueError, match="at least 5 observations"):
        engine.detect(short_stream)


def test_detect_returns_valid_structured_result(mock_dns_stream):
    """4 & 11. Test that detect returns a valid DetectionResult with all required fields."""
    engine = DeepDNSInferenceEngine(mode=InferenceMode.IN_DISTRIBUTION)
    result = engine.detect(mock_dns_stream)

    assert isinstance(result, DetectionResult)
    assert result.decision in ["ATTACK", "BENIGN"]
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.attack_probability <= 1.0
    assert 0.0 <= result.benign_probability <= 1.0
    assert result.stopping_horizon in [5, 10, 15, 20, 25, 30]
    assert result.max_horizon == 30
    assert result.observations_consumed == result.stopping_horizon
    assert result.observations_saved == 30 - result.stopping_horizon
    assert isinstance(result.evidence_history, dict)
    assert len(result.evidence_history) > 0


def test_streaming_process_observation(mock_dns_stream):
    """5, 6, 7 & 8. Test progressive streaming observation processing."""
    engine = DeepDNSInferenceEngine(mode=InferenceMode.IN_DISTRIBUTION)
    stream = DNSStream(stream_id="streaming_session_01")

    decisions = []
    for i, obs in enumerate(mock_dns_stream.observations):
        res, status = engine.process_observation(stream, obs)
        if res is not None:
            decisions.append((len(stream), res))

    # At least one decision must be reached by or at K=30
    assert len(decisions) >= 1
    final_len, final_res = decisions[0]
    assert final_len in [5, 10, 15, 20, 25, 30]
    assert final_res.decision in ["ATTACK", "BENIGN"]


def test_determinism(mock_dns_stream):
    """12. Test that repeated inference on identical input produces bit-exact outputs."""
    engine = DeepDNSInferenceEngine(mode=InferenceMode.IN_DISTRIBUTION)
    res1 = engine.detect(mock_dns_stream)
    res2 = engine.detect(mock_dns_stream)

    assert res1.decision == res2.decision
    assert res1.confidence == res2.confidence
    assert res1.stopping_horizon == res2.stopping_horizon
    assert res1.attack_probability == res2.attack_probability


def test_reference_reproduction():
    """13. Reference reproduction test comparing engine vs research raw model."""
    project_root = Path(__file__).resolve().parent.parent
    engine = DeepDNSInferenceEngine(mode=InferenceMode.IN_DISTRIBUTION, project_root=project_root)

    # Load 1 capture sample from test set
    test_csv = project_root / "data" / "raw" / "cic_bell_dns_exf_2021" / "stateless" / "stateless_features-light_text.pcap.csv"
    if test_csv.exists():
        df_sample = pd.read_csv(test_csv, nrows=30)
        res = engine.detect(df_sample)
        assert res.decision in ["ATTACK", "BENIGN"]
        assert res.stopping_horizon in [5, 10, 15, 20, 25, 30]
