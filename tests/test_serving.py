"""
Unit and Integration Tests for DeepDNS Serving Layer (FastAPI & WebSockets).
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
import numpy as np
import pandas as pd

from src.serving.app import app
from src.inference import DeepDNSInferenceEngine, InferenceMode, DNSStream, DNSObservation


@pytest.fixture
def client() -> TestClient:
    """Creates a test client for the FastAPI serving layer."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_attack_sequence() -> list:
    """Returns a list of 30 attack-like DNS observation dicts."""
    records = []
    base_time = pd.Timestamp("2026-08-28 12:00:00")
    for i in range(30):
        records.append({
            "timestamp": (base_time + pd.Timedelta(seconds=i * 0.4)).isoformat(),
            "domain_name": f"a9f3b2{i}.exfiltration.tunnel.example.com",
            "fqdn_count": 36.0,
            "subdomain_length": 18.0,
            "upper": 0.0,
            "lower": 30.0,
            "numeric": 6.0,
            "entropy": 2.45,
            "special": 3.0,
            "labels": 5.0,
            "labels_max": 18.0,
            "labels_average": 7.2,
            "len": 36.0,
            "subdomain": 1.0,
        })
    return records


@pytest.fixture
def sample_benign_sequence() -> list:
    """Returns a list of 30 benign-like DNS observation dicts."""
    records = []
    base_time = pd.Timestamp("2026-08-28 12:00:00")
    for i in range(30):
        records.append({
            "timestamp": (base_time + pd.Timedelta(seconds=i * 1.2)).isoformat(),
            "domain_name": f"static-cdn-{i}.google.com",
            "fqdn_count": 22.0,
            "subdomain_length": 12.0,
            "upper": 0.0,
            "lower": 20.0,
            "numeric": 1.0,
            "entropy": 2.10,
            "special": 2.0,
            "labels": 3.0,
            "labels_max": 12.0,
            "labels_average": 7.0,
            "len": 22.0,
            "subdomain": 1.0,
        })
    return records


def test_health_endpoint(client: TestClient):
    """Test GET /health returns ready status and available modes."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "deepdns"
    assert data["engine"] == "ready"
    assert "in_distribution" in data["available_modes"]
    assert "ood" in data["available_modes"]


def test_detect_endpoint_success(client: TestClient, sample_attack_sequence):
    """Test POST /detect with a valid 30-query sequence."""
    payload = {
        "mode": "in_distribution",
        "stream_id": "test-stream-001",
        "observations": sample_attack_sequence,
    }
    response = client.post("/detect", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["decision"] in ["ATTACK", "BENIGN"]
    assert 0.0 <= data["confidence"] <= 1.0
    assert 0.0 <= data["attack_probability"] <= 1.0
    assert 0.0 <= data["benign_probability"] <= 1.0
    assert data["stopping_horizon"] in [5, 10, 15, 20, 25, 30]
    assert data["max_horizon"] == 30
    assert data["observations_consumed"] == data["stopping_horizon"]
    assert data["observations_saved"] == 30 - data["stopping_horizon"]
    assert "evidence_history" in data
    assert data["mode"] == "in_distribution"


def test_detect_endpoint_min_observations_validation(client: TestClient, sample_attack_sequence):
    """Test POST /detect rejects sequences with fewer than 5 observations."""
    short_payload = {
        "mode": "in_distribution",
        "observations": sample_attack_sequence[:3],
    }
    response = client.post("/detect", json=short_payload)
    assert response.status_code == 400
    assert "requires at least 5 observations" in response.json()["detail"]


def test_detect_endpoint_invalid_mode(client: TestClient, sample_attack_sequence):
    """Test POST /detect rejects unsupported inference modes."""
    payload = {
        "mode": "invalid_mode_xyz",
        "observations": sample_attack_sequence[:10],
    }
    response = client.post("/detect", json=payload)
    assert response.status_code == 400
    assert "Unsupported inference mode" in response.json()["detail"]


def test_exact_reproduction_api_vs_engine(client: TestClient, sample_attack_sequence):
    """Test that the API result bit-for-bit reproduces the Canonical Inference Engine directly."""
    engine = DeepDNSInferenceEngine(mode=InferenceMode.IN_DISTRIBUTION)
    
    # Direct engine run
    direct_res = engine.detect(sample_attack_sequence)

    # API run
    payload = {
        "mode": "in_distribution",
        "observations": sample_attack_sequence,
    }
    response = client.post("/detect", json=payload)
    assert response.status_code == 200
    api_res = response.json()

    assert api_res["decision"] == direct_res.decision
    assert api_res["confidence"] == direct_res.confidence
    assert api_res["stopping_horizon"] == direct_res.stopping_horizon
    assert api_res["attack_probability"] == direct_res.attack_probability
    assert api_res["observations_saved"] == direct_res.observations_saved


def test_ood_mode_selection(client: TestClient, sample_attack_sequence):
    """Test POST /detect with OOD mode uses the OOD model and thresholds."""
    payload = {
        "mode": "ood",
        "observations": sample_attack_sequence,
    }
    response = client.post("/detect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "ood"
    assert data["model_name"] == "multiview_ood_both"


def test_websocket_streaming_flow(client: TestClient, sample_attack_sequence):
    """Test WebSocket /ws/stream/{stream_id} progressive streaming inference."""
    with client.websocket_connect("/ws/stream/ws-test-stream-01?mode=in_distribution") as ws:
        # Initial connection message
        init_msg = ws.receive_json()
        assert init_msg["event"] == "CONNECTED"
        assert init_msg["data"]["status"] == "READY"

        events_received = []
        for i, obs in enumerate(sample_attack_sequence):
            ws.send_text(json.dumps(obs))
            evt = ws.receive_json()
            events_received.append(evt)
            
            # If terminal decision received, stop sending
            if evt["event"] == "DECISION":
                break

        # Must receive at least one DECISION or EVIDENCE_UPDATE
        assert any(e["event"] in ["DECISION", "EVIDENCE_UPDATE", "OBSERVATION_BUFFERED"] for e in events_received)


def test_stream_isolation(client: TestClient, sample_attack_sequence, sample_benign_sequence):
    """Test that two separate streaming sessions maintain completely isolated state."""
    with client.websocket_connect("/ws/stream/stream-alpha?mode=in_distribution") as ws_a:
        ws_a.receive_json()  # CONNECTED

        with client.websocket_connect("/ws/stream/stream-beta?mode=in_distribution") as ws_b:
            ws_b.receive_json()  # CONNECTED

            # Send 5 attack observations to Stream A
            for obs in sample_attack_sequence[:5]:
                ws_a.send_text(json.dumps(obs))
                evt_a = ws_a.receive_json()

            # Send 5 benign observations to Stream B
            for obs in sample_benign_sequence[:5]:
                ws_b.send_text(json.dumps(obs))
                evt_b = ws_b.receive_json()

            assert evt_a["stream_id"] == "stream-alpha"
            assert evt_b["stream_id"] == "stream-beta"


def test_stream_termination_rejects_further_observations(client: TestClient, sample_attack_sequence):
    """Test that once a stream reaches DECISION, subsequent observations are rejected with an ERROR event."""
    with client.websocket_connect("/ws/stream/stream-term-test?mode=in_distribution") as ws:
        ws.receive_json()  # CONNECTED

        decision_received = False
        for obs in sample_attack_sequence:
            ws.send_text(json.dumps(obs))
            evt = ws.receive_json()
            if evt["event"] == "DECISION":
                decision_received = True
                break

        assert decision_received is True

        # Send one extra observation to the terminated stream
        ws.send_text(json.dumps(sample_attack_sequence[0]))
        err_evt = ws.receive_json()
        assert err_evt["event"] == "ERROR"
        assert "already terminated" in err_evt["data"]["message"]
