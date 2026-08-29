# DeepDNS Production Web Dashboard

## 1. Architecture & Design Philosophy

The **DeepDNS Production Frontend** is a scientific, high-precision detection instrument engineered to expose the real-time sequential multi-view inference and Adaptive Evidence Controller (AEC) of DeepDNS.

```
                ┌────────────────────────────────────────────────────────┐
                │                    DeepDNS Web App                     │
                │              (React + Vite + TypeScript)               │
                └───────────────────────────┬────────────────────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
             REST API Client                               WebSocket Client
             (src/api/client.ts)                           (src/api/websocket.ts)
                     │                                             │
          GET /health, POST /detect                    ws://localhost:8000/ws/stream/{id}
                     │                                             │
                     └──────────────────────┬──────────────────────┘
                                            ▼
                ┌────────────────────────────────────────────────────────┐
                │                 FastAPI Serving Layer                  │
                │                 (src/serving/app.py)                   │
                └───────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
                ┌────────────────────────────────────────────────────────┐
                │               Canonical Inference Engine               │
                │               (src/inference/engine.py)                │
                └────────────────────────────────────────────────────────┘
```

---

## 2. Key Features

- **Sequential Telemetry & Bounded Horizons**: Visualizes query progression across $K \in \{5, 10, 15, 20, 25, 30\}$.
- **Dynamic Evidence Trajectory Chart**: Real-time SVG chart plotting $P(\text{Attack})$ against calibrated operating thresholds ($\tau_{\text{attack}}$ and $\tau_{\text{benign}}$) with stopping horizon markers and query savings zones.
- **Decision & Query Savings Panel**: Highlights whether detection stopped early, exact query count consumed vs saved, percentage reduction ($50\% - 83.3\%$), confidence score, and AEC stopping rationale.
- **Live WebSocket Stream Simulator**: Streams real DNS observations progressively with realistic network timing or single-query stepping.
- **Batch & File Input**: Supports instant batch analysis (`/detect`) and custom CSV/JSON file uploads.
- **Benchmark Presets**: One-click loaders for CIC-Bell Light Text Attack (In-Distribution), Enterprise CDN Traffic (Benign), and Leave-One-Modality-Out Video Exfiltration (OOD).

---

## 3. Getting Started

### Development Server
From the `frontend/` directory:
```powershell
npm run dev
```
Access the application at `http://localhost:5173`.

### Production Build
```powershell
npm run build
npm run preview
```

---

## 4. Environment Configuration

The frontend connects to the FastAPI backend using standard environment variables:
- `VITE_API_BASE_URL`: Default `http://localhost:8000`
- `VITE_WS_BASE_URL`: Default `ws://localhost:8000`
