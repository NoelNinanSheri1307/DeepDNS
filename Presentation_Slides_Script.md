# DeepDNS: Slide-by-Slide Presentation Script & Speaker Notes

Use this guide for presenting DeepDNS step-by-step. Each slide includes the key visual points, technical talking points, and answers to expect.

---

### Slide 1: Title & Overview
* **Title**: **DeepDNS: Adaptive Evidence-Driven Dual-View DNS Exfiltration Detection**
* **Subtitle**: High-Precision, Low-Latency Covert Channel Detection in Enterprise DNS Traffic
* **Speaker Script**:
  > *"Good morning/afternoon. Today I am presenting DeepDNS, an adaptive, multi-view sequential framework designed to detect DNS data exfiltration in real-time. Traditional systems require buffering a large fixed window of queries—causing delayed alarms and high false positive rates. DeepDNS transforms detection into an adaptive sequential stopping process, cutting detection latency and query overhead by over 65% while delivering 99.2% recall on unseen attack modalities."*

---

### Slide 2: The Problem: Why DNS Exfiltration is Dangerous
* **Key Visuals**: Diagram showing Port 53 UDP traffic passing firewalls uninspected with embedded data in subdomains (e.g. `base64_chunk.tunnel.attacker.com`).
* **Core Pain Points**:
  1. **Unrestricted Port 53**: Enterprise firewalls cannot easily block DNS without breaking Internet resolution.
  2. **Static Window Delay**: Legacy detectors wait for fixed windows of 30–50 queries. By the time detection triggers, the sensitive file is already stolen.
  3. **High False Positive Rates on CDNs**: Benign services (Cloudflare, Akamai, Antivirus) use random-looking UUIDs, causing simple lexical detectors to trigger false alarms on up to 38% of legitimate queries.

---

### Slide 3: DeepDNS Core Architecture & Dual-View Methodology
* **Key Visuals**: Dual-View Architecture diagram:
  * **Branch 1 (Lexical)**: Character-CNN processing raw domain strings.
  * **Branch 2 (Behavioral)**: 2-Layer Temporal GRU processing 12 causal sequential features.
  * **Fusion Layer**: Concatenation + Multi-Layer Perceptron (MLP) generating step probability $p_k$.
* **Speaker Script**:
  > *"To solve the trade-off between false alarms and missed detections, DeepDNS processes two complementary views of DNS traffic:*
  > *First, the **Lexical View** uses a Character-CNN with 1D convolutions (kernel sizes 3, 5, 7) to extract n-gram orthographic patterns from domain strings.*
  > *Second, the **Behavioral View** uses a 2-Layer Temporal GRU with 12 causal behavioral features—such as inter-arrival time ($\Delta t$), burstiness, and Shannon entropy.*
  > *These two views are fused by an MLP into a time-step attack probability $p_k \in [0, 1]$ as each query arrives."*

---

### Slide 4: Novelty: The Adaptive Evidence Controller (AEC)
* **Key Visuals**: Step probability curve $p_k$ crossing dynamic thresholds $\tau_{\text{attack}}$ and $\tau_{\text{benign}}$ at checkpoints $K \in \{5, 10, 15, 20, 25, 30\}$.
* **Speaker Script**:
  > *"Our primary novelty is the **Adaptive Evidence Controller**. Rather than waiting for all 30 queries, the controller evaluates confidence at dynamic checkpoints ($K=5, 10, 15, 20, 25, 30$).*
  > *If confidence crosses our calibrated attack threshold ($\tau_{\text{atk}} = 0.95$), observation terminates immediately with an early alarm.*
  > *If confidence drops below our benign threshold ($\tau_{\text{ben}} = 0.15$), the stream is dismissed as benign.*
  > *Ambiguous streams continue accumulating evidence up to a bounded maximum horizon of 30 queries. This guarantees optimal latency without sacrificing accuracy."*

---

### Slide 5: Experimental Results & Key Deliverables
* **Key Metrics Table**:
  * **In-Distribution**: $98.17\%$ Recall, $0.12\%$ FPR, $0.9894$ F1, Mean Horizon $\bar{K}^* = 13.13$ queries (**$56.2\%$ Query Savings**).
  * **Sequential CUSUM**: $99.12\%$ Recall, $0.58\%$ FPR, Mean Horizon $\bar{K}^* = 9.22$ queries (**$69.3\%$ Query Savings**).
  * **Out-of-Distribution (LOMO)**: $99.21\%$ Recall, $0.50\%$ FPR, $0.9941$ F1, Mean Horizon $\bar{K}^* = 10.30$ queries (**$65.7\%$ Query Savings**).
* **Speaker Script**:
  > *"Our empirical evaluation on over 33,000 sequence windows confirms outstanding gains:*
  > *On completely unseen Out-of-Distribution attack modalities (such as video and text exfiltration), DeepDNS achieved a 99.21% detection rate in an average of just 10.3 queries.*
  > *This represents a 65.7% reduction in observation delay compared to static 30-query baselines."*

---

### Slide 6: Master Ablation Study (Why Both Views Matter)
* **Key Visuals**: Bar chart showing Lexical-Only vs Behavioral-Only vs Dual-View Fusion:
  * **Lexical-Only**: FPR of **$38.26\%$** (F1 = 0.7048) — Fails on benign CDNs.
  * **Behavioral-Only**: FPR $0.10\%$, but misses 47 attack sequences.
  * **Dual-View Fusion**: Synergistically reduces missed attacks by **$57.4\%$** while keeping FPR at $0.22\%$.
* **Speaker Script**:
  > *"Our ablation experiments validate the core hypothesis:*
  > *Using character patterns alone causes a 38% false positive rate because modern CDNs look random.*
  > *Using behavioral timing alone misses low-and-slow attacks.*
  > *Only the dual-view fusion achieves both low false alarms and near-perfect recall."*

---

### Slide 7: Live Web Dashboard & Serving Architecture
* **Key Visuals**: Frontend screenshot showing the Live Streaming Simulator, Interactive SVG Trajectory Chart, Custom Flow Builder, and Decision Banner.
* **Speaker Script**:
  > *"We translated this scientific model into an enterprise production system:*
  > *The backend is built with FastAPI and WebSockets, delivering real-time sub-millisecond inference.*
  > *The frontend is an interactive React dashboard where operators can simulate live traffic, upload batch captures, inspect all 12 causal features in real time, and generate custom synthetic exfiltration streams with auto-computed entropy."*

---

### Slide 8: Conclusion & Summary
* **Takeaway Bullet Points**:
  1. **Novel Adaptive Framework**: Early-stopping sequential evidence accumulation.
  2. **Superior Performance**: $99.21\%$ Recall, $0.50\%$ FPR on unseen OOD data.
  3. **Operational Efficiency**: $65.7\%$ reduction in time-to-detection and query buffering overhead.
  4. **Fully Verified & Tested**: Production FastAPI backend, React UI, and 76/76 passing test suite.
