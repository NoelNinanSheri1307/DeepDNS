# DeepDNS Production Serving Layer Documentation

## 1. Architecture Overview

The **DeepDNS Production Serving Layer** exposes the canonical Multi-View neural network and Adaptive Evidence Controller (AEC) as a high-performance REST and WebSocket API built with **FastAPI**.

```
                ┌─────────────────────────────────────┐
                │          DNS Traffic Client         │
                │     REST Client / Web Dashboard     │
                └──────────────────┬──────────────────┘
                                   │
              HTTP POST /detect    │   WebSocket /ws/stream/{id}
                                   │
                                   ▼
                ┌─────────────────────────────────────┐
                │         FastAPI Serving Layer       │
                │         (src/serving/app.py)        │
                ├──────────────────┬──────────────────┤
                │  Health Routes   │ Detection Routes │
                │  (/health)       │ (/detect)        │
                ├──────────────────┴──────────────────┤
                │     WebSocket Stream Manager        │
                │  (/ws/stream/{stream_id})           │
                └──────────────────┬──────────────────┘
                                   │
                                   ▼
                ┌─────────────────────────────────────┐
                │      Canonical Inference Engine     │
                │      (src/inference/engine.py)      │
                ├──────────────────┬──────────────────┤
                │ Behavioral GRU   │ Lexical Char-CNN │
                │ (z_beh in R^64)  │ (z_lex in R^128) │
                ├──────────────────┴──────────────────┤
                │        Dual-View Fusion Head        │
                │   Adaptive Evidence Controller      │
                └──────────────────┬──────────────────┘
                                   │
                                   ▼
                        JSON / WebSocket Event
```

---

## 2. Starting the Server

### Local Development / Production Server
```powershell
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload
```

### Auto-Generated Interactive API Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc UI**: `http://localhost:8000/redoc`
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

---

## 3. REST API Endpoints

### A. `GET /health`
Verifies service uptime and pre-warmed inference engine readiness.

**Response (200 OK)**:
```json
{
  "status": "ok",
  "service": "deepdns",
  "engine": "ready",
  "available_modes": [
    "in_distribution",
    "ood",
    "behavioral_only",
    "lexical_only"
  ]
}
```

---

### B. `POST /detect`
Primary batch/sequence detection endpoint for evaluated host sessions ($5 \le K \le 30$).

**Request Headers**: `Content-Type: application/json`

**Request Body**:
```json
{
  "mode": "in_distribution",
  "stream_id": "host-192-168-1-105",
  "observations": [
    {
      "timestamp": "2026-08-28T12:00:00.000",
      "domain_name": "chunk0.tunnel.exfiltration.com",
      "fqdn_count": 32.0,
      "subdomain_length": 14.0,
      "entropy": 2.45,
      "numeric": 4.0,
      "upper": 0.0,
      "special": 3.0
    },
    {
      "timestamp": "2026-08-28T12:00:00.400",
      "domain_name": "chunk1.tunnel.exfiltration.com",
      "fqdn_count": 32.0,
      "subdomain_length": 14.0,
      "entropy": 2.48,
      "numeric": 4.0,
      "upper": 0.0,
      "special": 3.0
    }
  ]
}
```

**Response (200 OK)**:
```json
{
  "decision": "ATTACK",
  "confidence": 0.9850,
  "attack_probability": 0.9850,
  "benign_probability": 0.0150,
  "stopping_horizon": 15,
  "max_horizon": 30,
  "observations_consumed": 15,
  "observations_saved": 15,
  "query_savings_pct": 50.0,
  "is_early_decision": true,
  "stop_reason": "ATTACK_THRESHOLD_MET (p=0.9850 >= 0.95)",
  "evidence_history": {
    "5": 0.2100,
    "10": 0.6700,
    "15": 0.9850
  },
  "model_name": "multiview_both",
  "mode": "in_distribution"
}
```

---

## 4. WebSocket Streaming Protocol (`/ws/stream/{stream_id}`)

Enables real-time streaming detection where DNS queries are fed as they occur on the wire.

### Connection
```
ws://localhost:8000/ws/stream/session-991?mode=in_distribution
```

### 1. Connection Established Event (Server $\to$ Client)
```json
{
  "event": "CONNECTED",
  "stream_id": "session-991",
  "data": {
    "mode": "in_distribution",
    "status": "READY"
  }
}
```

### 2. Client Sends Observation (Client $\to$ Server)
```json
{
  "timestamp": "2026-08-28T12:00:01.200",
  "domain_name": "a9f3b2.exfiltration.tunnel.example.com",
  "subdomain_length": 18.0,
  "entropy": 2.45
}
```

### 3. Intermediate Event (Server $\to$ Client)
- When between evaluation horizons:
```json
{
  "event": "OBSERVATION_BUFFERED",
  "stream_id": "session-991",
  "data": { "observations_count": 3 }
}
```
- When horizon reached but confidence uncertain:
```json
{
  "event": "EVIDENCE_UPDATE",
  "stream_id": "session-991",
  "data": {
    "horizon": 10,
    "attack_probability": 0.6700,
    "benign_probability": 0.3300,
    "status": "CONTINUE",
    "evidence_history": { "5": 0.2100, "10": 0.6700 }
  }
}
```

### 4. Terminal Decision Event (Server $\to$ Client)
```json
{
  "event": "DECISION",
  "stream_id": "session-991",
  "data": {
    "decision": "ATTACK",
    "confidence": 0.9921,
    "attack_probability": 0.9921,
    "stopping_horizon": 10,
    "observations_consumed": 10,
    "observations_saved": 20,
    "query_savings_pct": 66.67,
    "is_early_decision": true,
    "stop_reason": "ATTACK_THRESHOLD_MET (p=0.9921 >= 0.95)"
  }
}
```

---

## 5. Stream Lifecycle & Concurrency Isolation

- **State Machine**: `CREATED` $\to$ `RECEIVING` $\to$ `EVALUATING` $\to$ `DECISION_REACHED`.
- **Termination Guard**: Once a stream reaches `DECISION_REACHED`, any subsequent observations sent to that stream receive an `ERROR` event.
- **Memory Isolation**: Each stream operates with its own `DNSStream` and `asyncio.Lock`. Evidence from one client stream cannot contaminate another.

---

## 6. Running the Test Suite

```powershell
python -m pytest -v tests/test_serving.py
```
*(All 76 repository tests pass deterministically with 100% pass rate).*
