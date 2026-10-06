# DEEPDNS: COMPLETE SYSTEM KNOWLEDGE & TECHNICAL DEFENSE MASTER REFERENCE
> **Exhaustive Architectural Blueprint, Mathematical Foundations, Forensic Dataset Audit, Empirical Ablations, Serving Infrastructure, and Oral Defense Guide for the DeepDNS Project**

---

## TABLE OF CONTENTS
- [PART 1 — Project Identity](#part-1--project-identity)
- [PART 2 — Core Novelty](#part-2--core-novelty)
- [PART 3 — Complete System Pipeline](#part-3--complete-system-pipeline)
- [PART 4 — Dataset: Deep Technical Explanation](#part-4--dataset-deep-technical-explanation)
- [PART 5 — Data Preprocessing & Causal Sequencing](#part-5--data-preprocessing--causal-sequencing)
- [PART 6 — Causal Feature Engineering](#part-6--causal-feature-engineering)
- [PART 7 — Behavioral GRU Network](#part-7--behavioral-gru-network)
- [PART 8 — Character-CNN Encoder](#part-8--character-cnn-encoder)
- [PART 9 — Multi-View Dual-Branch Feature Fusion](#part-9--multi-view-dual-branch-feature-fusion)
- [PART 10 — Trained Models & Serialized Checkpoints](#part-10--trained-models--serialized-checkpoints)
- [PART 11 — Training Protocol & Optimization](#part-11--training-protocol--optimization)
- [PART 12 — Adaptive Evidence Controller (AEC) — Deep Dive](#part-12--adaptive-evidence-controller-aec--deep-dive)
- [PART 13 — Sequential CUSUM Statistical Change Detector](#part-13--sequential-cusum-statistical-change-detector)
- [PART 14 — Fixed-Horizon Baselines](#part-14--fixed-horizon-baselines)
- [PART 15 — Master Experimental Ablations](#part-15--master-experimental-ablations)
- [PART 16 — Complete Numerical Results & Metrics](#part-16--complete-numerical-results--metrics)
- [PART 17 — Metric Education & Mathematical Formulas](#part-17--metric-education--mathematical-formulas)
- [PART 18 — Deep Empirical Results Interpretation](#part-18--deep-empirical-results-interpretation)
- [PART 19 — Out-Of-Distribution (OOD) & LOMO Methodology](#part-19--out-of-distribution-ood--lomo-methodology)
- [PART 20 — Threshold Calibration & Sensitivity Analysis](#part-20--threshold-calibration--sensitivity-analysis)
- [PART 21 — Bootstrap Confidence Intervals](#part-21--bootstrap-confidence-intervals)
- [PART 22 — Horizon-Wise Error & Latency Decomposition](#part-22--horizon-wise-error--latency-decomposition)
- [PART 23 — Software Architecture & Repository Map](#part-23--software-architecture--repository-map)
- [PART 24 — Master Hyperparameter & Parameter Specifications](#part-24--master-hyperparameter--parameter-specifications)
- [PART 25 — Verification Suite & PyTest Testing](#part-25--verification-suite--pytest-testing)
- [PART 26 — Serving Layer (FastAPI & WebSockets)](#part-26--serving-layer-fastapi--websockets)
- [PART 27 — Frontend Architecture (React & Vite Dashboard)](#part-27--frontend-architecture-react--vite-dashboard)
- [PART 28 — End-To-End User & Operator Workflows](#part-28--end-to-end-user--operator-workflows)
- [PART 29 — What We Explicitly Did Not Do](#part-29--what-we-explicitly-did-not-do)
- [PART 30 — Systemic & Scientific Limitations](#part-30--systemic--scientific-limitations)
- [PART 31 — Future Engineering & Research Work](#part-31--future-engineering--research-work)
- [PART 32 — Presentation Knowledge & 3-Speaker Division](#part-32--presentation-knowledge--3-speaker-division)
- [PART 33 — Comprehensive Viva & Defense Q&A (50+ Questions)](#part-33--comprehensive-viva--defense-qa-50-questions)
- [PART 34 — Multi-Duration Presentation Scripts](#part-34--multi-duration-presentation-scripts)
- [PART 35 — Master Cheat Sheet](#part-35--master-cheat-sheet)
- [THE 20 THINGS WE ABSOLUTELY MUST KNOW](#the-20-things-we-absolutely-must-know)

---

# PART 1 — PROJECT IDENTITY

### Project Names & Titles
- **Project Name**: DeepDNS
- **Full Scientific Title**: *DeepDNS: Adaptive Evidence-Driven Dual-View Deep Sequential Framework for Real-Time DNS Exfiltration Detection*

### One-Sentence Description
> DeepDNS is an adaptive, multi-view sequential neural detection system that dynamically fuses causal behavioral dynamics and sub-domain lexical orthography, terminating observation as soon as accumulated confidence bounds are satisfied to stop DNS covert exfiltration with minimal latency.

### The Core Problem Being Solved
Domain Name System (DNS) port 53 (UDP) is fundamentally open across corporate perimeter firewalls to permit essential domain name resolution. Malicious actors, Advanced Persistent Threats (APTs), and malware weaponize this protocol as a covert outbound channel. By chopping exfiltrated documents, video files, keystrokes, or authentication tokens into alphanumeric strings, embedding them as subdomains (e.g., `dGVzdF9kYXRh.tunnel.attacker.com`), and querying authoritative attacker nameservers, data is stolen without triggering conventional IP or signature firewalls.

### Why DNS Exfiltration is Difficult to Detect
1. **Volumetric Camouflage**: DNS traffic is massive in enterprise environments (often exceeding tens of thousands of queries per second). Exfiltration packets are hidden within vast seas of benign lookups.
2. **Lexical Ambiguity & CDN High Entropy**: Legitimate Content Delivery Networks (Akamai, CloudFront, Fastly), cloud platforms (AWS S3, Microsoft Azure), and endpoint security telemetry generate random-looking, high-entropy UUIDs and hash subdomains that closely mimic encrypted attack payloads.
3. **Behavioral Camouflage (Low-and-Slow Attacks)**: Sophisticated adversaries throttle their transmission rate (e.g., 1 query every 30 seconds) and insert random jitter to evade static volumetric threshold rules.
4. **Data Leakage in Published Benchmarks**: Many existing machine learning papers evaluate classifiers by randomly shuffling DNS queries across time, causing severe forward-looking statistical contamination and falsely inflated accuracy.

### What Conventional / Fixed-Window Approaches Do
Conventional machine learning approaches enforce a **static, fixed observation window** (e.g., buffering a rigid sequence of $K=30$ or $K=50$ queries). They require network middleboxes to wait until the entire buffer fills before extracting aggregated statistics and executing a single forward pass.

### What DeepDNS Specifically Improves
1. **Slashes Observation Latency & Time-to-Detection (MTTD)**: Converts static batch buffering into an adaptive sequential optimal stopping process. Clear benign flows and blatant attacks are terminated early (e.g., at $K=5$ or $K=10$), while only ambiguous streams are held up to $K=30$.
2. **Eliminates Single-View False Alarms**: Fuses raw character-level orthography with causal temporal behavior, reducing missed detections and preventing high false alarms on legitimate CDN traffic.
3. **Guarantees Strict Causal Realism**: Completely eliminates data leakage by restricting all behavioral feature extraction to past-only sliding statistics ($t \le k$) and evaluating chronological session splits.

### Central Research / Engineering Hypothesis
> *"Converting multi-view neural predictions into an adaptive evidence accumulation process with calibrated stopping bounds will reduce observation latency and telemetry buffering overhead by over 50% relative to fixed-horizon baselines, while preserving peak exfiltration detection recall ($\ge 98\%$) and false positive rates ($\le 0.5\%$) on out-of-distribution attack payloads."*

### Primary & Secondary Objectives
- **Primary Objectives**:
  1. Construct a causal dual-view neural architecture combining a Character-CNN and a Temporal GRU.
  2. Implement and calibrate an Adaptive Evidence Controller (AEC) over discrete observation checkpoints $K \in \{5, 10, 15, 20, 25, 30\}$.
  3. Validate generalization across Leave-One-Modality-Out (LOMO) out-of-distribution payloads (Video and Text).
- **Secondary Objectives**:
  1. Build an enterprise-ready FastAPI and WebSocket serving layer delivering sub-millisecond per-step inference.
  2. Develop a high-fidelity interactive React/Vite dashboard featuring live SVG trajectory charts, synthetic flow generation, and stream inspection.
  3. Formulate an automated Sequential CUSUM baseline detector for statistical change-point comparison.

### What the Final System Does (Input to Output)
1. **Input**: A real-time stream of raw DNS query packets (timestamps and domain name strings).
2. **Transform**: Computes 12 causal past-to-present behavioral features and tokenizes domain characters.
3. **Inference**: Passes both views through the dual-view network to emit sequential attack probability $p_k$.
4. **Adaptive Decision**: AEC evaluates $p_k$ against $(\tau_{\text{attack}}, \tau_{\text{benign}})$.
5. **Output**: Emits `ATTACK`, `BENIGN`, or `CONTINUE`, recording the exact stopping horizon $K^*$, query savings percentage, confidence score, and explanatory feature metrics.

### Three Levels of Explanation

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             DEEPDNS AT A GLANCE                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│ 1-Line: An AI system that stops DNS data theft in the fewest queries possible    │
│ without waiting for fixed data buffers.                                          │
│                                                                                  │
│ 30-Second: DeepDNS analyzes both domain spellings and timing rhythms as queries  │
│ arrive. Instead of waiting for 30 queries, its Adaptive Controller stops as soon │
│ as it is confident—saving 65% in observation time while catching 99.2% of attacks│
│ on unseen data.                                                                  │
│                                                                                  │
│ Technical: DeepDNS is a causal multi-view framework fusing a Character-CNN and a │
│ Temporal GRU into an Adaptive Evidence Controller that solves an optimal stopping│
│ problem over horizons K ∈ {5..30}, achieving 99.21% OOD Recall with a 65.7%      │
│ telemetry reduction.                                                             │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

# PART 2 — CORE NOVELTY

### Dissecting Existing vs. Novel Components

```
┌────────────────────────────────────────┬────────────────────────────────────────┐
│ EXISTING / STANDARD ML COMPONENTS      │ DEEPDNS NOVEL SCIENTIFIC CONTRIBUTIONS │
├────────────────────────────────────────┼────────────────────────────────────────┤
│ • Gated Recurrent Units (GRU)          │ • Adaptive Evidence Controller (AEC)   │
│ • 1D Character Convolutional Networks  │ • Sequential Optimal Stopping on DNS   │
│ • Cross-Entropy Loss & AdamW Optimizer │ • Calibrated Dual-Threshold Bounds     │
│ • Shannon Entropy & Ratio Calculations │ • Multi-View Causal Behavioral-Lexical │
│ • Standard Softmax Probability Scoring │   Cross-Modality Fusion                │
│ • FastAPI Web Serving & React UI       │ • Leave-One-Modality-Out Generalization│
│ • Cumulative Sum (CUSUM) Formulation   │ • Bounded Horizon Fallback Safety Net  │
└────────────────────────────────────────┴────────────────────────────────────────┘
```

### The Exact Technical Contribution
DeepDNS does **not** claim to have invented the GRU, the CNN, or the concept of calculating domain entropy. 

**DeepDNS’s core frozen novelty is:**
> The formulation of DNS covert channel detection as an **Adaptive Multi-View Sequential Optimal Stopping Process**, wherein causal temporal-behavioral representations and character-orthographic representations are jointly evaluated at discrete checkpoints $K \in \{5, 10, 15, 20, 25, 30\}$ against strictly calibrated threshold pairs $(\tau_{\text{attack}}, \tau_{\text{benign}})$, dynamically terminating observation the moment confidence bounds are crossed.

### The Novelty Mechanism: Mathematical & Algorithmic Formulation

Let a DNS stream observation up to step $k$ be represented as:
$$\mathcal{S}_k = \{(x_1^{\text{beh}}, x_1^{\text{lex}}), (x_2^{\text{beh}}, x_2^{\text{lex}}), \dots, (x_k^{\text{beh}}, x_k^{\text{lex}})\}$$
where $x_t^{\text{beh}} \in \mathbb{R}^{12}$ represents the causal behavioral feature vector and $x_t^{\text{lex}} \in \mathbb{Z}^L$ represents the tokenized character sequence of the domain queried at time $t$.

1. **Step-wise Multi-View Inference**:
   $$z_k^{\text{beh}} = \text{GRU}(x_1^{\text{beh}}, \dots, x_k^{\text{beh}}) \in \mathbb{R}^{64}$$
   $$z_k^{\text{lex}} = \text{CharCNN}(x_k^{\text{lex}}) \in \mathbb{R}^{128}$$
   $$z_k^{\text{fuse}} = \text{MLP}_{\text{fuse}}([\text{Proj}_{\text{beh}}(z_k^{\text{beh}}) \,\|\, \text{Proj}_{\text{lex}}(z_k^{\text{lex}})]) \in \mathbb{R}^{64}$$
   $$p_k = P(\text{Attack} \mid \mathcal{S}_k) = \sigma(\text{Classifier}(z_k^{\text{fuse}}))$$

2. **Adaptive Evidence Stopping Policy**:
   For each discrete checkpoint $k \in \mathcal{K} = \{5, 10, 15, 20, 25, 30\}$:
   $$\delta(k) = \begin{cases} 
   \text{ATTACK} & \text{if } p_k \ge \tau_{\text{attack}} \\
   \text{BENIGN} & \text{if } p_k \le \tau_{\text{benign}} \\
   \text{CONTINUE} & \text{if } \tau_{\text{benign}} < p_k < \tau_{\text{attack}} \text{ and } k < K_{\max} \\
   \mathbb{I}(p_k \ge 0.5) & \text{if } k = K_{\max} = 30
   \end{cases}$$

3. **Optimal Stopping Horizon $K^*$**:
   $$K^* = \min \{ k \in \mathcal{K} \mid \delta(k) \ne \text{CONTINUE} \}$$

4. **Telemetry / Query Volume Savings**:
   $$\text{Savings} = \left( 1 - \frac{K^*}{K_{\max}} \right) \times 100\%$$

### Why This Differs from Running a Classifier Once
- **Static Classifier**: Must wait for a rigid buffer of 30 queries before executing. If 29 queries arrive, it makes zero decisions. Once 30 queries arrive, it consumes 100% of telemetry overhead regardless of how obvious the attack or benign pattern was.
- **DeepDNS AEC**: Evaluates the cumulative posterior evidence sequence $\{p_5, p_{10}, \dots, p_{30}\}$. It halts immediately at $K=5$ if the evidence is overwhelming, saving 83.3% of queries for that flow, while safely preserving the full 30-step context for subtle attacks.

### Why the Frontend is Not the Novelty
The React/Vite dashboard and WebSocket serving pipeline are **engineering artifacts and delivery mechanisms**. They provide transparency, visualization, and interaction for security analysts, but the patentable scientific novelty resides entirely within the causal multi-view fusion and the Adaptive Evidence Controller.

---

# PART 3 — COMPLETE SYSTEM PIPELINE

```
                     ┌────────────────────────────────────────┐
                     │ Raw DNS Query Stream (t_1, t_2, ... ) │
                     └───────────────────┬────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
      [ Causal Feature Extraction ]                   [ Character Tokenizer ]
      • Delta-t inter-arrival timing                  • Alphanumeric vocab (size=45)
      • Sliding window entropy & ratios               • Fixed-length tensor (L=128)
      • Strict past-only lookback (t <= k)            • Padding & Unknown handling
                 │                                               │
                 ▼                                               ▼
      [ Behavioral View: Temporal GRU ]               [ Lexical View: Char-CNN ]
      • 2-Layer Causal GRU (hidden=64)                • 1D Convolutions (k=3, 5, 7)
      • Recurrent State H_k ∈ R^64                    • Global Max Pooling (dim=128)
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                     [ Dual-Branch Late Fusion Head ]
                     • Projections: 64 -> 64, 128 -> 64
                     • LayerNorm + ReLU + Concat (128-D)
                     • Fusion MLP: 128 -> 64 -> 32 -> 2
                                         │
                                         ▼
                     [ Instantaneous Probability p_k ]
                     • P(Attack) = Softmax(Logits)[1]
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
   [ Adaptive Evidence Controller (AEC) ]           [ Sequential CUSUM Baseline ]
   • Checkpoints: K ∈ {5,10,15,20,25,30}           • Cumulative log-odds tracking
   • If p_k >= tau_attack -> STOP: ATTACK          • Drift correction gamma = 0.0
   • If p_k <= tau_benign -> STOP: BENIGN          • Threshold h = 4.0
   • Else -> ACCUMULATE NEXT QUERY                 • Two-sided change detection
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                     [ Final Decision & Telemetry Metrics ]
                     • Predicted Label: ATTACK / BENIGN
                     • Stopping Horizon K* (e.g., K* = 10)
                     • Telemetry Reduction % (e.g., 66.7% Savings)
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
      [ FastAPI Serving Layer ]                       [ React 18 + Vite UI ]
      • REST: POST /detect                            • Live SVG Evidence Trajectory
      • WebSocket: /ws/stream/{id}                    • Real-Time Causal Feature Grid
```

### Complete Pipeline Stage Breakdown

| Stage | Input | Output | Purpose | Key Parameters | Source File |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. DNS Ingestion** | Raw network packets | Timestamps & FQDNs | Captures query arrival time and domain text. | RFC 1035 format | [`src/serving/routes.py`](file:///D:/UnderWay%20Dev%20Projects/deepdns/src/serving/routes.py) |
| **2. Causal Feature Extractor** | Chronological query stream | 12 causal numerical features | Computes past-only rate, entropy, and length features. | Past lookback $t \le k$ | [`src/data/feature_extraction.py`](file:///D:/UnderWay%20Dev%20Projects/deepdns/src/data/feature_extraction.py) |
| **3. Character Tokenizer** | Domain text string | Integer token tensor $\mathbb{Z}^{128}$ | Translates subdomains into character sequences. | Vocab size=45, Max len=128 | [`src/models/char_cnn.py`](file:///D:/UnderWay%20Dev%20Projects/deepdns/src/models/char_cnn.py) |
| **4. Temporal GRU** | Sequence of 12-D vectors | Behavioral embedding $z_{\text{beh}} \in \mathbb{R}^{64}$ | Models burstiness, inter-arrival rhythms, and query rates. | Hidden dim=64, Layers=1 | [`src/models/temporal_gru.py`](file:///D:/UnderWay%20Dev%20Projects/deepdns/src/models/temporal_gru.py) |
| **5. Character-CNN** | Token tensor $(B, K, 128)$ | Lexical embedding $z_{\text{lex}} \in \mathbb{R}^{128}$ | Extracts n-gram orthographic patterns and character entropy. | Filters=32, Kernels=(3,5,7) | [`src/models/char_cnn.py`](file:///D:/UnderWay%20Dev%20Projects/deepdns/src/models/char_cnn.py) |
| **6. Fusion MLP** | $z_{\text{beh}}$ (64) + $z_{\text{lex}}$ (128) | Unified state $z_{\text{fuse}} \in \mathbb{R}^{64}$ | Bridges lexical patterns with temporal timing. | Proj=64, Concat=128, Fuse=64 | [`src/models/multiview_fusion.py`](file:///D:/UnderWay%20Dev%20Projects/deepdns/src/models/multiview_fusion.py) |
| **7. Classifier Head** | Fused state $z_{\text{fuse}}$ | Step probability $p_k \in [0, 1]$ | Computes likelihood of malicious exfiltration. | Linear $(64 \to 2)$ + Softmax | [`src/models/multiview_fusion.py`](file:///D:/UnderWay%20Dev%20Projects/deepdns/src/models/multiview_fusion.py) |
| **8. AEC Controller** | Probability sequence $p_k$ | `AdaptiveDecision` | Enforces early stopping stopping criteria. | $\tau_{\text{atk}}=0.95, \tau_{\text{ben}}=0.15$ | [`src/inference/adaptive_controller.py`](file:///D:/UnderWay%20Dev%20Projects/deepdns/src/inference/adaptive_controller.py) |
| **9. Serving API** | JSON payload or WebSocket packet | Structured detection response | Exposes low-latency enterprise inference. | ASGI Uvicorn, Lifespan engine | [`src/serving/app.py`](file:///D:/UnderWay%20Dev%20Projects/deepdns/src/serving/app.py) |
| **10. Web UI** | WebSocket JSON stream | Interactive Visual Dashboard | Renders live charts, gauges, and feature inspection. | React 18, Vite, SVG | [`frontend/src/App.tsx`](file:///D:/UnderWay%20Dev%20Projects/deepdns/frontend/src/App.tsx) |

---

# PART 4 — DATASET: DEEP TECHNICAL EXPLANATION

### Forensic Dataset Summary Table

| Dataset Property | Actual DeepDNS Value | Detailed Technical Explanation |
| :--- | :--- | :--- |
| **Primary Dataset Name** | **CIC-Bell-DNS-EXF-2021** | Canadian Institute for Cybersecurity & Bell Canada DNS Exfiltration Dataset (2021). |
| **Lexical Pretraining Dataset** | **DNS Threats Dataset** | 3,103,513 raw domain queries (2.48M train, 620K test) for orthographic character modeling. |
| **Dataset Purpose** | Realistic Covert DNS Detection | Benchmark dataset capturing realistic enterprise benign background traffic and active exfiltration tunnels. |
| **Why Selected** | Real PCAP Monotonic Timestamps | Provides unperturbed packet timings and raw subdomains generated by actual exfiltration tools. |
| **Why Pure Synthetic Was Rejected**| Artificial Timing Regularity | Synthetic-only datasets feature deterministic query intervals that cause models to overfit on toy cadence artifacts. |
| **Traffic / Modalities Represented**| Audio, Video, Text, Compressed, Benign | Captures exfiltration of real multimedia, plain text, and system files alongside enterprise browsing. |
| **Tunneling Tools Used** | Iodine, DNS2TCP, Custom Scripts | Real covert channel tunneling utilities operating over Port 53 UDP. |
| **Stateless Capture Records** | **697,120 rows** | Raw query-by-query records with monotonic timestamps across 18 capture sessions (36 CSVs). |
| **Stateful Capture Records** | **239,337 rows** (Excluded) | Pre-aggregated static CSVs; excluded due to 65.7% row mismatch and temporal lookahead leakage. |
| **Causal Features Retained** | **12 numerical features** | Past-only features selected after strict forensic leak audit (`src/data/schema.py`). |
| **In-Distribution (ID) Test Windows**| **13,084 sequence windows** | Chronologically isolated session windows ($K=30$) evaluated zero-shot. |
| **OOD (LOMO) Test Windows** | **20,683 sequence windows** | Completely held-out Video and Text attack modalities evaluated zero-shot. |
| **Window Length ($K_{\max}$)** | **$30$ queries** | Maximum sequence length for sliding window evidence accumulation. |
| **Window Sliding Mechanism** | Sliding with stride=1 within captures | Sequential sliding windows extracted strictly within individual capture boundaries. |
| **Leakage Prevention** | Capture-Level Chronological Split | Captures are segregated entirely; zero domain or temporal overlap between train, val, and test. |

### The Leave-One-Modality-Out (LOMO) Protocol
In machine learning for security, models often overfit to specific file signatures (e.g., the specific entropy of compressed zip headers). To scientifically prove DeepDNS learns generalized exfiltration dynamics rather than memorizing file types, we instituted **Leave-One-Modality-Out (LOMO)**:
- **Training Set**: Exposed exclusively to Audio, Compressed, and Benign traffic.
- **Held-Out OOD Test Set ($N=20,683$)**: Composed entirely of **100% unseen Video and Text exfiltration** alongside unseen benign sessions.

---

# PART 5 — DATA PREPROCESSING & CAUSAL SEQUENCING

### The Lifecycle of a Raw DNS Query
When a raw DNS query arrives at the network interface:
1. **Timestamp Extraction**: Record arrival timestamp $t_i$. Compute inter-arrival time $\Delta t_i = t_i - t_{i-1}$ (with $\Delta t_1 = 0.0$).
2. **Log-Transform**: Apply $\ln(1 + \Delta t)$ to compress heavy-tailed inter-arrival delays.
3. **Causal Behavioral Extraction**: Compute the 12 causal statistics using strictly the current and previous queries ($j \le i$).
4. **Standard Scaler Normalization**: Normalize behavioral features using mean and variance fitted strictly on training data (`fitted_scaler.pkl`).
5. **Character Tokenization**: Convert the raw domain name into integer character indices using vocabulary mapping (size=45). Pad or truncate to $L=128$.
6. **Tensor Assembly**: Append into sliding sequence buffer $(B, K, 12)$ and $(B, K, 128)$.

### Why "Causal" Preprocessing is Mandatory
In non-causal systems, features like "average session query rate" or "session total bytes" look forward into future packets. If a query at $t=1$ knows that 500 queries will occur in the next 10 minutes, the model cheats. **DeepDNS enforces strict causality**: at step $k$, all features are functions strictly of $\{q_1, q_2, \dots, q_k\}$. Future information is mathematically impossible to access.

### Concrete Evolution of Features Across 3 Consecutive Queries

```
Query 1: "login.google.com" (t=0.00s)
  ├── delta_t = 0.00s
  ├── fqdn_length = 16, subdomain_length = 5
  ├── char_entropy = 2.85 (natural text)
  └── digit_ratio = 0.00, uppercase_ratio = 0.00

Query 2: "a7f9b2c.tunnel.attacker.com" (t=0.12s)
  ├── delta_t = 0.12s (rapid burst)
  ├── fqdn_length = 27, subdomain_length = 7
  ├── char_entropy = 3.62 (elevated hex entropy)
  └── digit_ratio = 0.43 (3 digits / 7 chars)

Query 3: "e812d4a.tunnel.attacker.com" (t=0.23s)
  ├── delta_t = 0.11s (constant cadence)
  ├── fqdn_length = 27, subdomain_length = 7
  ├── char_entropy = 3.58 (consistent hex payload)
  └── burstiness_score = elevated (high short-term QPS)
```

---

# PART 6 — CAUSAL FEATURE ENGINEERING

### The Approved 12 Causal Behavioral Features

| # | Feature Name | Data Type | Meaning & Definition | Why Useful for Detection | Is Causal? |
| :-: | :--- | :---: | :--- | :--- | :---: |
| **1** | `inter_arrival_time` | `float32` | $\Delta t_i = t_i - t_{i-1}$ in seconds (log-scaled). | Detects automated tunneling tools transmitting at fixed or looped intervals. | **YES** |
| **2** | `fqdn_length` | `float32` | Total character length of the Fully Qualified Domain Name. | Exfiltration attempts maximize payload per packet, inflating FQDN length. | **YES** |
| **3** | `subdomain_length` | `float32` | Length of subdomains excluding Second-Level Domain (SLD). | Attack payloads are embedded directly into subdomains. | **YES** |
| **4** | `char_entropy` | `float32` | Shannon entropy: $H(X) = -\sum p(x) \log_2 p(x)$. | Base64 and hex encrypted payloads exhibit high character randomness. | **YES** |
| **5** | `digit_ratio` | `float32` | Proportion of numeric digits `[0-9]` in the FQDN. | Hex-encoded payloads contain ~30-50% digits; natural domains have <5%. | **YES** |
| **6** | `uppercase_ratio` | `float32` | Proportion of uppercase characters `[A-Z]`. | Base64-encoded payloads mix cases, whereas standard DNS lookups are lowercase. | **YES** |
| **7** | `special_ratio` | `float32` | Proportion of special characters (`-`, `_`, `.`). | Detects separator padding and non-standard label structures. | **YES** |
| **8** | `label_count` | `float32` | Number of dot-separated labels in the domain. | Multi-level tunneling scripts generate 4–8 subdomain labels. | **YES** |
| **9** | `max_label_length` | `float32` | Length of the longest individual label in the domain. | Attackers push labels up to the 63-character RFC limit. | **YES** |
| **10**| `avg_label_length` | `float32` | Average character length across all domain labels. | Distinguishes deep hierarchical tunnels from standard web domains. | **YES** |
| **11**| `has_subdomain` | `float32` | Binary flag indicating presence of subdomains ($1.0$ or $0.0$). | Filters standard apex domain lookups (`example.com`). | **YES** |
| **12**| `payload_len` | `float32` | Estimated raw data payload bytes encoded in the query. | Measures cumulative exfiltration volume per query. | **YES** |

### Explicitly Forbidden Features (Leakage Prevention)
Columns such as `sld` (Second-Level Domain text), `longest_word`, and static session aggregations are **strictly prohibited** from the model input tensors to prevent target memorization and dataset-specific artifacts (`FORBIDDEN_MODEL_COLUMNS` in `src/data/schema.py`).

---

# PART 7 — BEHAVIORAL GRU NETWORK

### Why Gated Recurrent Units (GRU)?
A Recurrent Neural Network (RNN) maintains a hidden state vector $h_t$ that updates with each arriving query. 
- **Why not Standard RNN?** Standard RNNs suffer from vanishing gradients over sequence length $K=30$.
- **Why not LSTM?** LSTMs maintain separate cell and hidden states. A GRU achieves identical temporal representation capacity for inter-arrival dynamics with **25% fewer parameters**, reducing inference latency.
- **Why not Transformers?** Transformers have quadratic complexity $O(K^2)$ and lack an inductive bias for chronological streaming, requiring heavy positional encodings and higher memory bandwidth.

### Mathematical Formulation of the Behavioral GRU
For input vector $x_t \in \mathbb{R}^{12}$ and previous hidden state $h_{t-1} \in \mathbb{R}^{64}$:
1. **Reset Gate**: $r_t = \sigma(W_r x_t + U_r h_{t-1} + b_r)$
2. **Update Gate**: $z_t = \sigma(W_z x_t + U_z h_{t-1} + b_z)$
3. **Candidate Hidden State**: $\tilde{h}_t = \tanh(W_h x_t + U_h (r_t \odot h_{t-1}) + b_h)$
4. **Output Hidden State**: $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$

### Architectural Specifications
- **Input Dimension**: $12$ causal features
- **Hidden Dimension**: $64$ units
- **Layers**: $1$ recurrent layer (with optional 2-layer extension)
- **Dropout**: $0.10$ / $0.20$
- **Output Embedding ($z_{\text{beh}}$)**: $\mathbb{R}^{64}$

---

# PART 8 — CHARACTER-CNN ENCODER

### Why Character-Level Convolutions?
Domain names are not natural language sentences; they are dense, structured alphanumeric strings. Word-level tokenizers (like WordPiece or BPE) fail on obfuscated subdomains like `a9f4c8e1.tunnel.com`. A **Character-CNN** processes domain text character-by-character, capturing character n-gram frequencies, substring transitions, and hex/base64 patterns invariant to string position.

### Architectural Specifications
- **Character Vocabulary**: $45$ unique tokens (lowercase letters `a-z`, digits `0-9`, hyphen `-`, underscore `_`, dot `.`, padding `<PAD>`, and unknown `<UNK>`).
- **Input Tensor**: Sequence of character indices padded/truncated to $L=128$.
- **Embedding Layer**: Projects integer indices into continuous embedding space $\mathbb{R}^{32}$.
- **Parallel 1D Convolutions**:
  - Filter Kernel Size $3$ (trigrams): $32$ filters $\to$ captures short hex chunks.
  - Filter Kernel Size $5$ (5-grams): $32$ filters $\to$ captures hash blocks.
  - Filter Kernel Size $7$ (7-grams): $32$ filters $\to$ captures encoded words.
- **Pooling**: Global 1D Max-Pooling across sequence length, yielding $3 \times 32 = 96 \to 128$ dimensions.
- **Output Embedding ($z_{\text{lex}}$)**: $\mathbb{R}^{128}$

---

# PART 9 — MULTI-VIEW DUAL-BRANCH FEATURE FUSION

```
  Behavioral Embedding (z_beh ∈ R^64)       Lexical Embedding (z_lex ∈ R^128)
                 │                                         │
                 ▼                                         ▼
     [ Linear(64, 64) + LayerNorm ]            [ Linear(128, 64) + LayerNorm ]
                 │                                         │
                 └────────────────────┬────────────────────┘
                                      ▼
                        [ Concatenation Layer (128-D) ]
                                      │
                                      ▼
                        [ Linear(128, 64) + LayerNorm ]
                                      │
                                      ▼
                           [ ReLU + Dropout(0.10) ]
                                      │
                                      ▼
                        [ Fused Embedding z_fuse ∈ R^64 ]
                                      │
                                      ▼
                        [ Linear(64, 2) Output Head ]
                                      │
                                      ▼
                        [ Softmax -> P(Attack) ∈ [0, 1] ]
```

### Dimensional Dimensions Throughout Fusion
- $z_{\text{beh}} \in \mathbb{R}^{64}$
- $z_{\text{lex}} \in \mathbb{R}^{128}$
- Projected Behavioral: $p_{\text{beh}} = \text{ReLU}(\text{LN}(\text{Linear}(64, 64))) \in \mathbb{R}^{64}$
- Projected Lexical: $p_{\text{lex}} = \text{ReLU}(\text{LN}(\text{Linear}(128, 64))) \in \mathbb{R}^{64}$
- Concatenated Vector: $[p_{\text{beh}} \,\|\, p_{\text{lex}}] \in \mathbb{R}^{128}$
- Fused Representation: $z_{\text{fuse}} = \text{Dropout}(\text{ReLU}(\text{LN}(\text{Linear}(128, 64)))) \in \mathbb{R}^{64}$
- Final Logits: $\text{Linear}(64, 2) \in \mathbb{R}^2$

### Why the Two Views are Synergistic
- **Lexical View Alone**: Cannot distinguish between a legitimate high-entropy CDN lookup and an exfiltration chunk ($\text{FPR} = 38.26\%$).
- **Behavioral View Alone**: Cannot see the payload contents inside the subdomain, missing stealthy low-rate tunnels ($FN = 47$).
- **Dual-View Fusion**: The behavioral branch checks *how* the query arrived (timing and burstiness), while the lexical branch checks *what* arrived (subdomain structure). Missed attacks drop by **$57.4\%$** while keeping false alarms at $0.22\%$.

---

# PART 10 — TRAINED MODELS & SERIALIZED CHECKPOINTS

### Master Checkpoint Inventory

| Checkpoint Name | Model Architecture | Primary Purpose | Training Split | Final Canonical System? |
| :--- | :--- | :--- | :--- | :---: |
| **`multiview_both.pt`** | Full Dual-View Network | Standard In-Distribution Inference | ID Train (Audio, Compressed, Benign) | **YES (Primary ID Model)** |
| **`multiview_ood_both.pt`**| Full Dual-View Network | OOD Evaluation (LOMO) | LOMO Train (Audio, Compressed, Benign) | **YES (Primary OOD Model)** |
| **`multiview_behavioral_only.pt`**| Behavioral-Only Dual-View | Scientific Ablation Baseline | ID Train (Lexical zeroed out) | **Ablation Baseline** |
| **`multiview_lexical_only.pt`**| Lexical-Only Dual-View | Scientific Ablation Baseline | ID Train (Behavioral zeroed out) | **Ablation Baseline** |

### Historical Evolution of the GRU Work
Early in development, a standalone single-view Temporal GRU was trained. **The GRU was not abandoned**—it was refined into the permanent behavioral branch of the unified `DeepDNSMultiViewNetwork`.

---

# PART 11 — TRAINING PROTOCOL & OPTIMIZATION

### Training Hyperparameters & Configurations

| Parameter | Value / Specification | Technical Justification |
| :--- | :--- | :--- |
| **Deep Learning Framework** | PyTorch 2.x | Standard production tensor library. |
| **Optimizer** | AdamW | Decoupled weight decay prevents weight norm drift. |
| **Learning Rate** | $1 \times 10^{-3}$ | Stable convergence rate for recurrent-convolutional networks. |
| **Weight Decay** | $1 \times 10^{-4}$ | Regularization preventing over-reliance on high-magnitude weights. |
| **Batch Size** | $128$ | Optimal GPU tensor parallelism. |
| **Epochs** | $10$ | Sufficient for full empirical convergence without overfitting. |
| **Loss Function** | Cross-Entropy Loss | Calibrated multi-class / binary classification loss. |
| **Learning Rate Schedule** | Cosine Annealing | Smooth learning rate decay towards final epochs. |
| **Random Seed** | $42$ | Seeded across PyTorch, NumPy, and CUDA for deterministic reproduction. |
| **Training Hardware** | NVIDIA GPU (RTX 2050 / Ampere) | Sub-10ms batch throughput. |

---

# PART 12 — ADAPTIVE EVIDENCE CONTROLLER (AEC) — DEEP DIVE

### The Core Concept
The **Adaptive Evidence Controller (AEC)** operationalizes the principle of *minimum necessary evidence*:
> *"A security system should not spend telemetry budget or delay alarms when evidence is already conclusive."*

### Discrete Horizons & Stopping Policy
Evaluations occur at discrete horizon checkpoints:
$$\mathcal{K} \in \{5, 10, 15, 20, 25, 30\}$$

At each checkpoint $k \in \mathcal{K}$:
- If $p_k \ge \tau_{\text{attack}}$: **HALT IMMEDIATELY $\to$ EMIT ATTACK ALARM**.
- If $p_k \le \tau_{\text{benign}}$: **HALT IMMEDIATELY $\to$ CONFIRM BENIGN STREAM**.
- If $\tau_{\text{benign}} < p_k < \tau_{\text{attack}}$: **CONTINUE $\to$ INGEST NEXT QUERY ($k \leftarrow k + 1$)**.
- If $k = K_{\max} = 30$: **FORCED DECISION $\to \mathbb{I}(p_{30} \ge 0.50)$**.

### Calibrated Threshold Sets

```
┌───────────────────────────────┬───────────────────────────────┐
│ IN-DISTRIBUTION (ID)          │ OUT-OF-DISTRIBUTION (OOD)     │
├───────────────────────────────┼───────────────────────────────┤
│ • tau_attack = 0.95           │ • tau_attack = 0.80           │
│ • tau_benign = 0.15           │ • tau_benign = 0.01           │
│ • Objective: Constrain FPR    │ • Objective: Maximize Recall  │
│   under 0.15% on known flows. │   on unseen attack payloads.  │
└───────────────────────────────┴───────────────────────────────┘
```

### Three Worked Examples

#### Example 1: Blatant High-Rate Tunnel (Early Attack Decision)
- $K=5$: Query stream sends 5 consecutive base64 subdomains at 50ms intervals.
- Step probability: $p_5 = 0.982$.
- Condition Check: $p_5 \ge \tau_{\text{attack}}$ ($0.982 \ge 0.95$).
- **Result**: **ATTACK detected at $K^*=5$**. Queries saved: $(1 - 5/30) = \mathbf{83.3\%}$.

#### Example 2: Standard Benign Corporate Browsing (Early Benign Decision)
- $K=5$: User queries `google.com`, `slack.com`, `github.com`.
- Step probability: $p_5 = 0.021$.
- Condition Check: $p_5 \le \tau_{\text{benign}}$ ($0.021 \le 0.15$).
- **Result**: **BENIGN confirmed at $K^*=5$**. Queries saved: $\mathbf{83.3\%}$.

#### Example 3: Low-and-Slow Stealth Exfiltration (Multi-Horizon Decision)
- $K=5$: Queries spaced 10s apart with mixed natural/hex subdomains. $p_5 = 0.62$ $\to$ **CONTINUE**.
- $K=10$: More hex subdomains appear. $p_{10} = 0.84$ $\to$ **CONTINUE**.
- $K=15$: Sustained entropy confirms malicious intent. $p_{15} = 0.963$.
- Condition Check: $p_{15} \ge \tau_{\text{attack}}$ ($0.963 \ge 0.95$).
- **Result**: **ATTACK detected at $K^*=15$**. Queries saved: $(1 - 15/30) = \mathbf{50.0\%}$.

---

# PART 13 — SEQUENTIAL CUSUM STATISTICAL CHANGE DETECTOR

### What is Sequential CUSUM?
The **Cumulative Sum (CUSUM)** algorithm is a classical sequential change-point detection method. DeepDNS implements a two-sided CUSUM detector (`src/inference/cusum.py`) that tracks accumulated log-odds scores:
$$L_k = \ln\left( \frac{p_k}{1 - p_k} \right)$$
$$S_k^+ = \max(0, S_{k-1}^+ + L_k - \gamma)$$
$$S_k^- = \max(0, S_{k-1}^- - L_k - \gamma)$$
where $\gamma = 0.0$ is the reference drift parameter, $h_{\text{attack}} = 4.0$ is the positive alarm threshold, and $h_{\text{benign}} = 4.0$ is the negative confirmation threshold.

### Why CUSUM is a Baseline Rather than the Main Novelty
CUSUM is a well-established statistical quality-control algorithm (Page, 1954). DeepDNS includes it as a rigorous comparative benchmark against the Adaptive Evidence Controller. While CUSUM provides excellent rapid change detection on in-distribution streams ($\bar{K}^*=9.22$), the AEC provides superior recall and stability under out-of-distribution domain shifts ($99.21\%$ vs $96.17\%$).

---

# PART 14 — FIXED-HORIZON BASELINES

### Why Fixed Horizons Were Evaluated
To understand the trade-off between latency and accuracy, DeepDNS evaluated fixed observation windows at $K \in \{5, 10, 15, 20, 25, 30\}$:
- **$K=5$**: Ultra-fast latency, but catastrophic false alarm rate on in-distribution data ($\text{FPR} = 5.49\%$).
- **$K=10$**: Moderate balance, but still suffers $1.37\%$ FPR.
- **$K=30$ (Static Baseline)**: High accuracy ($\text{F1} = 0.9952$), but forces every stream to buffer 30 queries before emitting an alert ($0\%$ query savings).
- **DeepDNS AEC**: Achieves the accuracy of $K=30$ ($\text{F1} = 0.9894$) with the latency benefits of early stopping ($\bar{K}^* = 10.3\text{--}13.1$).

---

# PART 15 — MASTER EXPERIMENTAL ABLATIONS

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         SUMMARY OF SYSTEM ABLATIONS                              │
├────────────────────────────────┬───────────────────┬──────────────┬──────────────┤
│ Ablation Dimension             │ Primary Question  │ Finding      │ Consequence  │
├────────────────────────────────┼───────────────────┼──────────────┼──────────────┤
│ 1. Lexical-Only vs Dual-View   │ Can Char-CNN work │ NO: FPR =    │ Fails on     │
│                                │ alone?            │ 38.26%       │ CDNs         │
├────────────────────────────────┼───────────────────┼──────────────┼──────────────┤
│ 2. Behavioral-Only vs Dual-View│ Can GRU work      │ NO: Misses   │ Fails on low-│
│                                │ alone?            │ 47 attacks   │ and-slow     │
├────────────────────────────────┼───────────────────┼──────────────┼──────────────┤
│ 3. Fixed Window vs AEC         │ Is static buffer  │ AEC saves    │ 65.7% lower  │
│                                │ necessary?        │ 56% - 66%    │ latency      │
├────────────────────────────────┼───────────────────┼──────────────┼──────────────┤
│ 4. AEC vs Sequential CUSUM     │ Which controller  │ AEC wins on  │ AEC superior │
│                                │ is more robust?   │ OOD recall   │ for security │
└────────────────────────────────┴───────────────────┴──────────────┴──────────────┘
```

---

# PART 16 — COMPLETE NUMERICAL RESULTS & METRICS

### 1. In-Distribution Evaluation Matrix ($N = 13,084$)
*(Ground Truth: 4,201 Attacks, 8,883 Benign)*

| Decision Strategy | Accuracy | Precision | Recall | F1-Score | FPR | Mean Horizon $\bar{K}^*$ | Early Decision % | TP | FP | TN | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed $K=5$** | 95.63% | 89.40% | 98.00% | 0.9350 | 5.4936% | 5.00 | 100.0% | 4117 | 488 | 8395 | 84 |
| **Fixed $K=10$** | 98.91% | 97.16% | 99.52% | 0.9833 | 1.3734% | 10.00 | 100.0% | 4181 | 122 | 8761 | 20 |
| **Fixed $K=15$** | 99.57% | 98.91% | 99.76% | 0.9934 | 0.5178% | 15.00 | 100.0% | 4191 | 46 | 8837 | 10 |
| **Fixed $K=20$** | 99.61% | 99.01% | 99.79% | 0.9940 | 0.4728% | 20.00 | 100.0% | 4192 | 42 | 8841 | 9 |
| **Fixed $K=25$** | 99.69% | 99.31% | 99.74% | 0.9952 | 0.3265% | 25.00 | 100.0% | 4190 | 29 | 8854 | 11 |
| **Fixed $K=30$ (Baseline)** | 99.69% | 99.52% | 99.52% | 0.9952 | 0.2251% | 30.00 | 0.0% | 4181 | 20 | 8863 | 20 |
| **Behavioral-Only ($K=30$)** | 99.49% | 99.78% | 98.88% | 0.9933 | 0.1013% | 30.00 | 0.0% | 4154 | 9 | 8874 | 47 |
| **Lexical-Only ($K=30$)** | 73.84% | 54.91% | 98.45% | 0.7048 | 38.2641% | 30.00 | 0.0% | 4136 | 3399 | 5484 | 65 |
| **DeepDNS AEC (Calibrated)** | **99.33%** | **99.73%** | **98.17%** | **0.9894** | **0.1238%** | **13.13** | **70.0%** | **4124** | **11** | **8872** | **77** |
| **DeepDNS Sequential CUSUM** | **99.32%** | **98.77%** | **99.12%** | **0.9894** | **0.5854%** | **9.22** | **99.7%** | **4164** | **52** | **8831** | **37** |

---

### 2. Out-of-Distribution (LOMO) Evaluation Matrix ($N = 20,683$)
*(Ground Truth: 11,800 Attacks [Video + Text], 8,883 Benign)*

| Decision Strategy | Accuracy | Precision | Recall | F1-Score | FPR | Mean Horizon $\bar{K}^*$ | Early Decision % | TP | FP | TN | FN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fixed $K=5$** | 89.13% | 97.62% | 82.97% | 0.8970 | 2.6905% | 5.00 | 100.0% | 9791 | 239 | 8644 | 2009 |
| **Fixed $K=10$** | 96.56% | 99.36% | 94.58% | 0.9691 | 0.8105% | 10.00 | 100.0% | 11161 | 72 | 8811 | 639 |
| **Fixed $K=15$** | 99.31% | 99.76% | 99.03% | 0.9939 | 0.3152% | 15.00 | 100.0% | 11685 | 28 | 8855 | 115 |
| **Fixed $K=20$** | 99.56% | 99.72% | 99.52% | 0.9962 | 0.3715% | 20.00 | 100.0% | 11743 | 33 | 8850 | 57 |
| **Fixed $K=25$** | 99.61% | 99.79% | 99.53% | 0.9966 | 0.2814% | 25.00 | 100.0% | 11744 | 25 | 8858 | 56 |
| **Fixed $K=30$ (Baseline)** | 99.59% | 99.85% | 99.44% | 0.9964 | 0.2026% | 30.00 | 0.0% | 11734 | 18 | 8865 | 66 |
| **DeepDNS AEC (Calibrated)** | **99.33%** | **99.62%** | **99.21%** | **0.9941** | **0.5066%** | **10.30** | **99.9%** | **11707** | **45** | **8838** | **93** |
| **DeepDNS Sequential CUSUM** | **97.74%** | **99.86%** | **96.17%** | **0.9798** | **0.1801%** | **12.96** | **99.8%** | **11348** | **16** | **8867** | **452** |

---

# PART 17 — METRIC EDUCATION & MATHEMATICAL FORMULAS

### Mathematical Definitions & DeepDNS Context

#### 1. Confusion Matrix Elements
- **True Positive (TP)**: Malicious exfiltration correctly flagged as ATTACK. ($TP = 4,124$ ID / $11,707$ OOD).
- **True Negative (TN)**: Benign corporate lookup correctly identified as BENIGN. ($TN = 8,872$ ID / $8,838$ OOD).
- **False Positive (FP)**: Benign query wrongly flagged as ATTACK (False Alarm). ($FP = 11$ ID / $45$ OOD).
- **False Negative (FN)**: Malicious exfiltration missed (Breach). ($FN = 77$ ID / $93$ OOD).

#### 2. Detection Rate (Recall / Sensitivity)
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
- *Plain English*: Out of 100 actual DNS attacks, how many did we catch?
- *DeepDNS Score*: **$98.17\%$ (ID)** / **$99.21\%$ (OOD)**. High recall ensures stealthy tunnels do not escape detection.

#### 3. False Positive Rate (FPR / Fall-out)
$$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$
- *Plain English*: Out of 100 benign queries, how many false alarms did we sound?
- *DeepDNS Score*: **$0.1238\%$ (ID)** / **$0.5066\%$ (OOD)**. In enterprise networks processing millions of queries, an FPR $>1\%$ floods Security Operations Centers (SOCs) with alert fatigue.

#### 4. Precision (Positive Predictive Value)
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
- *Plain English*: When DeepDNS sounds an alarm, what is the probability it is a real attack?
- *DeepDNS Score*: **$99.73\%$ (ID)** / **$99.62\%$ (OOD)**.

#### 5. F1-Score (Harmonic Mean)
$$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
- *Plain English*: The balanced harmonic score between catching attacks and avoiding false alarms.
- *DeepDNS Score*: **$0.9894$ (ID)** / **$0.9941$ (OOD)**.

#### 6. Mean Stopping Horizon ($\bar{K}^*$) & Query Savings %
$$\bar{K}^* = \frac{1}{N} \sum_{i=1}^N K_i^*, \quad \text{Query Savings} = \left( 1 - \frac{\bar{K}^*}{30} \right) \times 100\%$$
- *Plain English*: The average number of queries consumed before making a confident decision.
- *DeepDNS Score*: **$\bar{K}^* = 13.13$ ($56.2\%$ savings on ID)** and **$\bar{K}^* = 10.30$ ($65.7\%$ savings on OOD)**.

---

# PART 18 — DEEP EMPIRICAL RESULTS INTERPRETATION

### 1. What the Lexical-Only Failure Tells Us
When we disabled the behavioral branch, the Char-CNN alone scored a catastrophic False Positive Rate of **$38.26\%$** ($3,399$ false alarms). This proves empirically that domain name strings alone are insufficient for enterprise detection: legitimate enterprise CDNs, software update servers, and cloud endpoints produce high-entropy subdomains that fool lexical-only models.

### 2. What the 57.4% FN Reduction Demonstrates
Comparing Behavioral-Only ($FN=47$) to Dual-View Fusion ($FN=20$) shows a **$57.4\%$ reduction in missed attacks**. The lexical branch provides critical orthographic evidence that catches subtle, low-rate exfiltration tunnels that timing features alone miss.

### 3. Why AEC Achieves 65.7% Savings on OOD Data
On out-of-distribution attacks, exfiltration bursts create unambiguous sequential evidence early in the stream. The AEC terminates **$41.57\%$ of decisions at $K=5$** and **$39.19\%$ at $K=15$**, resulting in $99.9\%$ of streams being classified early with an average horizon of just **$10.30$ queries**.

---

# PART 19 — OUT-OF-DISTRIBUTION (OOD) & LOMO METHODOLOGY

### The Conceptual Analogy
> *Imagine training an airport security scanner exclusively on knives and handguns. If a passenger attempts to smuggle liquid explosives, will the scanner detect it, or will it only recognize shapes it has seen before?*

In DNS security, attackers constantly invent new encoding schemes and file formats. If a detector only works on the file types it was trained on, it is useless in production.

### The LOMO Setup
DeepDNS held out entire data modalities (Video and Text) during training. The evaluation on $N=20,683$ windows tests whether the model learned general exfiltration dynamics (inter-arrival regularity, label depth, character entropy shifts) rather than specific file header signatures. The result—**$99.21\%$ Recall and $0.9941$ F1**—proves strong generalization.

---

# PART 20 — THRESHOLD CALIBRATION & SENSITIVITY ANALYSIS

### Rigorous Calibration on Validation Data
Thresholds were **never tuned on test data**. We calibrated thresholds exclusively on isolated validation partitions (`val_captures`):
- **In-Distribution**: Selected $\tau_{\text{attack}} = 0.95$ and $\tau_{\text{benign}} = 0.15$ to enforce validation $\text{FPR} \le 0.15\%$.
- **Out-of-Distribution**: Selected $\tau_{\text{attack}} = 0.80$ and $\tau_{\text{benign}} = 0.01$ to ensure maximum sensitivity on unseen payload types.

### Neighborhood Stability & Sensitivity Plateau
Sensitivity sweeps across $\tau_{\text{attack}} \in [0.80, 0.98]$ and $\tau_{\text{benign}} \in [0.01, 0.15]$ revealed a stable performance plateau: F1-score remained above $0.985$ across all neighborhood variations, proving that DeepDNS is not fragile to minor threshold perturbations.

---

# PART 21 — BOOTSTRAP CONFIDENCE INTERVALS

To establish statistical significance, we performed non-parametric bootstrapping with **1,000 iterations (seed=42)**:

| Setting | Metric | Empirical Mean | 95% Bootstrap Confidence Interval |
| :--- | :--- | :---: | :---: |
| **In-Distribution AEC** | Recall | 98.17% | **[97.74%, 98.56%]** |
| | False Positive Rate (FPR) | 0.1238% | **[0.0561%, 0.1995%]** |
| | F1-Score | 0.9894 | **[0.9872, 0.9916]** |
| | Mean Stopping Horizon $\bar{K}^*$ | 13.13 | **[12.93, 13.31]** |
| **Out-of-Distribution AEC** | Recall | 99.21% | **[99.05%, 99.36%]** |
| | False Positive Rate (FPR) | 0.5066% | **[0.3730%, 0.6536%]** |
| | F1-Score | 0.9941 | **[0.9931, 0.9950]** |
| | Mean Stopping Horizon $\bar{K}^*$ | 10.30 | **[10.23, 10.37]** |

---

# PART 22 — HORIZON-WISE ERROR & LATENCY DECOMPOSITION

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   HORIZON-WISE DECISION DISTRIBUTION                             │
├─────────┬───────────────────────────────┬────────────────────────────────────────┤
│ Horizon │ In-Distribution Decisions (%) │ Out-of-Distribution Decisions (%)      │
├─────────┼───────────────────────────────┼────────────────────────────────────────┤
│ K = 5   │ 63.51% (Mostly Benign)        │ 41.57% (Benign + Rapid Attacks)        │
│ K = 10  │  3.87%                        │ 15.20%                                 │
│ K = 15  │  0.67%                        │ 39.19% (Concentrated OOD Attacks)      │
│ K = 20  │  0.42%                        │  3.80%                                 │
│ K = 25  │  1.51%                        │  0.17%                                 │
│ K = 30  │ 30.01% (Ambiguous Streams)    │  0.07%                                 │
└─────────┴───────────────────────────────┴────────────────────────────────────────┘
```

- **Benign Dismissal at $K=5$**: $63.51\%$ of in-distribution streams are confirmed benign at step 5 ($TN=8,257, FP=0$).
- **OOD Concentration at $K=15$**: Unseen video and text attacks require slightly more evidence, concentrating decisions at $K=15$ ($TP=8,006, FP=15$).

---

# PART 23 — SOFTWARE ARCHITECTURE & REPOSITORY MAP

```
deepdns/
├── src/
│   ├── data/
│   │   ├── schema.py              <- Column specs, 12 causal features & forbidden lists
│   │   ├── feature_extraction.py  <- Backwards-looking causal feature calculator
│   │   └── dataset.py             <- Sliding window dataset builder
│   ├── models/
│   │   ├── char_cnn.py            <- Character-CNN PyTorch architecture (Conv1D + MaxPool)
│   │   ├── temporal_gru.py        <- 1/2-Layer Temporal GRU network
│   │   └── multiview_fusion.py    <- Dual-Branch Late-Fusion Multi-View model
│   ├── inference/
│   │   ├── adaptive_controller.py <- Adaptive Evidence Controller (AEC) early stopping
│   │   ├── cusum.py               <- Sequential CUSUM statistical change-point detector
│   │   ├── canonical_engine.py    <- Unified production inference engine
│   │   └── types.py               <- Dataclasses and Pydantic types
│   ├── serving/
│   │   ├── app.py                 <- FastAPI application factory with CORS
│   │   ├── routes.py              <- REST (/detect, /models) and WebSocket (/ws/stream)
│   │   ├── schemas.py             <- Request / Response validation schemas
│   │   └── config.py              <- Server configuration and checkpoint paths
│   └── evaluation/
│       └── metrics.py             <- Classification, CUSUM, and bootstrap metrics
├── frontend/
│   ├── src/
│   │   ├── App.tsx                <- Main React Dashboard application
│   │   ├── components/            <- Chart, DecisionBanner, CustomStreamBuilder
│   │   └── hooks/useWebSocket.ts  <- Live streaming WebSocket hook
├── scripts/
│   ├── evaluate_adaptive_controller.py <- Master AEC evaluation script
│   └── run_ablation_study.py           <- Multi-view ablation execution script
└── tests/                         <- 76 Passing PyTest unit and integration tests
```

---

# PART 24 — MASTER HYPERPARAMETER & PARAMETER SPECIFICATIONS

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             MASTER PARAMETER TABLE                               │
├────────────────────────────────┬─────────────────────────────────────────────────┤
│ Parameter Group                │ Specific Values & Configurations                │
├────────────────────────────────┼─────────────────────────────────────────────────┤
│ Model Dimensions               │ Input: 12 causal features, Hidden GRU: 64       │
│                                │ Lexical Vocab: 45, Embedding: 32, Conv: 32x3    │
│                                │ Kernels: (3, 5, 7), Lexical Dim: 128            │
│                                │ Projections: 64, Concat: 128, Fused: 64         │
│ Training Hyperparameters       │ Optimizer: AdamW, LR: 1e-3, Weight Decay: 1e-4 │
│                                │ Batch Size: 128, Epochs: 10, Loss: CrossEntropy │
│ AEC Inference Parameters       │ Horizons: [5, 10, 15, 20, 25, 30], K_max: 30    │
│                                │ ID Thresholds: tau_atk=0.95, tau_ben=0.15       │
│                                │ OOD Thresholds: tau_atk=0.80, tau_ben=0.01      │
│ Sequential CUSUM Parameters    │ h_attack: 4.0, h_benign: 4.0, drift gamma: 0.0  │
│ Evaluation Parameters          │ Bootstrap Iterations: 1,000, Confidence: 95%    │
│                                │ Random Seed: 42, Device: CUDA / CPU             │
└────────────────────────────────┴─────────────────────────────────────────────────┘
```

---

# PART 25 — VERIFICATION SUITE & PYTEST TESTING

- **Current Status**: **76 / 76 Unit & Integration Tests Passing (100% Pass Rate)**.
- **What Tests Cover**:
  1. Causal feature calculation & non-leakage verification (`tests/test_feature_extraction.py`).
  2. Model forward/backward pass & shape compatibility (`tests/test_models.py`).
  3. AEC early-stopping and threshold boundary enforcement (`tests/test_adaptive_controller.py`).
  4. CUSUM change-point logic (`tests/test_cusum.py`).
  5. FastAPI REST endpoints & WebSocket live streaming (`tests/test_serving.py`).

---

# PART 26 — SERVING LAYER (FASTAPI & WEBSOCKETS)

- **Engine Architecture**: Unified in-memory model prewarming via FastAPI Lifespan.
- **REST Endpoints**:
  - `GET /health`: Health check returning loaded models and memory status.
  - `POST /detect`: Synchronous batch detection for PCAP JSON sequences.
  - `GET /models`: Returns active model checkpoints and threshold metadata.
- **WebSocket Streaming**:
  - `/ws/stream/{stream_id}`: Bi-directional real-time stream. Ingests single DNS query objects, updates internal recurrent state, and immediately yields step probability $p_k$, AEC status, and SVG trajectory coordinates.

---

# PART 27 — FRONTEND ARCHITECTURE (REACT & VITE DASHBOARD)

- **Framework**: React 18, TypeScript, Vite, custom responsive CSS.
- **Key UI Capabilities**:
  1. **Streaming Simulator**: Real-time packet-by-packet playback with adjustable delay ($100\text{--}2000\text{ms}$).
  2. **Live Evidence Trajectory SVG Chart**: Dynamically plots instantaneous probability $p_k$ relative to $\tau_{\text{attack}}$ and $\tau_{\text{benign}}$ boundaries.
  3. **Custom Stream Builder**: Allows analysts to type or generate synthetic DNS streams with automated Shannon entropy calculation.
  4. **Architecture & Research Hub**: Visualizes model layer diagrams and ablation tables directly in the browser.

---

# PART 28 — END-TO-END USER & OPERATOR WORKFLOWS

```
Operator loads UI -> Selects Model Mode (ID / OOD) -> Chooses Stream Source
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
[ Preset Attack (DNS2TCP/Iodine) ]          [ Custom Synthetic Flow Builder ]
        │                                               │
        └───────────────────────┬───────────────────────┘
                                ▼
         Click "Start Stream" (Connects via WebSocket)
                                │
                                ▼
         Observe live SVG chart plotting p_k at k=1,2,3...
                                │
                                ▼
    AEC triggers at K* (e.g. K*=10) -> Alert Banner flashes RED
                                │
                                ▼
    Review Telemetry Savings (66.7%) & Inspect 12 Causal Feature Values
```

---

# PART 29 — WHAT WE EXPLICITLY DID NOT DO

1. **Did NOT use Stateful Pre-Aggregated CSVs**: Excluded due to 65.7% row mismatch and temporal leakage.
2. **Did NOT Shuffle Test Queries Across Time**: Enforced strict capture-level chronological splitting.
3. **Did NOT Calibrate Thresholds on Test Data**: Calibrated exclusively on isolated validation partitions.
4. **Did NOT Rely on Transformers**: Selected GRUs for superior edge efficiency and lower latency.
5. **Did NOT Present CUSUM as the Primary Novelty**: Positioned CUSUM as a statistical comparative baseline.

---

# PART 30 — SYSTEMIC & SCIENTIFIC LIMITATIONS

1. **Benchmark Dependence**: Evaluated on CIC-Bell-DNS-EXF-2021; real-world multi-enterprise deployments may require local threshold fine-tuning.
2. **Maximum Horizon Delay ($K_{\max}=30$)**: In extremely ambiguous streams, 30 queries must elapse before a forced decision is reached.
3. **Encrypted DNS (DoH/DoT)**: If DNS queries are tunneled inside TLS (DNS-over-HTTPS), domain strings are hidden, forcing reliance solely on packet sizes and timings.

---

# PART 31 — FUTURE ENGINEERING & RESEARCH WORK

1. **Live PCAP / eBPF Kernel Driver**: Direct packet sniffing from network interfaces (AF_PACKET / eBPF).
2. **Online Adaptive Threshold Tuning**: Dynamic threshold adaptation based on real-time network baseline noise.
3. **DoH/DoT Extension**: Adaptation of behavioral branch to encrypted TLS packet flow metadata.
4. **SIEM / SOC Integration**: Automated alert emission to Splunk, Elastic, and Microsoft Sentinel.

---

# PART 32 — PRESENTATION KNOWLEDGE & 3-SPEAKER DIVISION

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          3-SPEAKER PRESENTATION ROLES                            │
├──────────────────────────────────────────────────────────────────────────────────┤
│ SPEAKER 1: Problem, Dataset & Causal Preprocessing                               │
│ • Focus: The DNS exfiltration threat, fixed-window flaws, CIC-Bell audit,        │
│   and the 12 causal features.                                                    │
│ • Key Numbers: 697,120 stateless rows, 12 features, zero-leakage split.          │
│                                                                                  │
│ SPEAKER 2: Dual-View Architecture & Adaptive Evidence Controller (AEC)           │
│ • Focus: Temporal GRU, Character-CNN, Dual-Branch Fusion, AEC stopping logic,     │
│   and threshold calibration.                                                     │
│ • Key Numbers: Hidden dim=64, Vocab=45, tau_atk=0.95/0.80, tau_ben=0.15/0.01.    │
│                                                                                  │
│ SPEAKER 3: Experiments, Ablations, Results & Live System Demo                    │
│ • Focus: 65.7% latency reduction, 99.21% OOD Recall, 38.26% Lexical failure,     │
│   FastAPI/React architecture, and live UI demo.                                  │
│ • Key Numbers: 99.21% Recall, 0.50% FPR, K*=10.30, 76/76 tests passing.         │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

# PART 33 — COMPREHENSIVE VIVA & DEFENSE Q&A (50+ QUESTIONS)

### 1. Basic Project Questions
**Q1: What is DeepDNS in one sentence?**
- *Short Answer*: An adaptive, dual-view sequential deep learning system that detects DNS exfiltration with minimal query latency.
- *Detailed Answer*: DeepDNS combines a Character-CNN and a Temporal GRU with an Adaptive Evidence Controller that terminates observation at dynamic checkpoints $K \in \{5..30\}$ once calibrated confidence bounds are satisfied.
- *Common Mistake*: Calling it a simple domain classifier.

**Q2: What is DNS exfiltration?**
- *Short Answer*: Stealing data by encoding it into DNS query subdomains over Port 53 UDP.
- *Detailed Answer*: Attackers chop stolen files into base64/hex chunks, format them as subdomains of an attacker-controlled apex domain, and query nameservers, bypassing standard perimeter firewalls.
- *Common Mistake*: Confusing DNS exfiltration with DNS Amplification / DDoS.

### 2. Dataset & Preprocessing Questions
**Q3: Which dataset did you use?**
- *Short Answer*: CIC-Bell-DNS-EXF-2021 (697,120 stateless records) and DNS Threats.
- *Detailed Answer*: We used 18 capture sessions of the CIC-Bell-DNS-EXF-2021 dataset containing real benign and attack tunneling traffic across multiple multimedia and text modalities.
- *Common Mistake*: Claiming you used the stateful CSVs (which were excluded due to leakage).

**Q4: How did you prevent data leakage?**
- *Short Answer*: Capture-level chronological splitting and strictly past-only causal feature extraction.
- *Detailed Answer*: We split data by whole capture sessions rather than shuffling rows randomly, and computed all 12 behavioral features causally using only past queries ($t \le k$).
- *Common Mistake*: Assuming random train/test splits are acceptable for sequential network traffic.

### 3. Model Architecture Questions
**Q5: Why use a GRU instead of an LSTM?**
- *Short Answer*: GRU provides identical temporal modeling with 25% fewer parameters and lower latency.
- *Detailed Answer*: For tracking 12 causal inter-arrival and rate features, a GRU eliminates separate cell state memory overhead, enabling faster sub-millisecond per-query inference.
- *Common Mistake*: Saying GRUs are always more accurate than LSTMs.

**Q6: Why use a Character-CNN for domain names?**
- *Short Answer*: To extract sub-word character n-grams and hex patterns invariant to string position.
- *Detailed Answer*: Standard NLP tokenizers fail on obfuscated subdomains. A 1D CNN with kernels 3, 5, and 7 captures character trigrams and hash patterns effectively.
- *Common Mistake*: Saying CNNs are only used for 2D images.

**Q7: How are the two views fused?**
- *Short Answer*: Late-fusion concatenation followed by a Multi-Layer Perceptron.
- *Detailed Answer*: Projections of the behavioral embedding ($64\text{-D}$) and lexical embedding ($128\text{-D}$) are concatenated into a $128\text{-D}$ vector, passed through a LayerNorm-ReLU-Dropout MLP, yielding a 64-D fused state.
- *Common Mistake*: Claiming early fusion was performed directly on raw inputs.

### 4. AEC & Novelty Questions
**Q8: What is the core novelty of DeepDNS?**
- *Short Answer*: The Adaptive Evidence Controller (AEC) for sequential optimal early stopping.
- *Detailed Answer*: Transforming DNS detection from fixed-horizon batching into an adaptive evidence accumulation process that terminates observation as soon as calibrated thresholds are crossed.
- *Common Mistake*: Claiming the React frontend or PyTorch models alone are the novelty.

**Q9: What are the calibrated thresholds?**
- *Short Answer*: In-Distribution: $(0.95, 0.15)$; Out-of-Distribution: $(0.80, 0.01)$.
- *Detailed Answer*: Tuned strictly on isolated validation partitions to balance early stopping with strict false alarm constraints.
- *Common Mistake*: Saying thresholds were tuned on test data.

**Q10: What happens if evidence is inconclusive at $K=30$?**
- *Short Answer*: A forced decision is made using standard binary thresholding ($p_{30} \ge 0.50$).
- *Detailed Answer*: DeepDNS enforces a bounded maximum horizon of 30 queries, ensuring no stream is buffered indefinitely.
- *Common Mistake*: Saying the model gives up and emits an error.

### 5. Empirical Results & Ablation Questions
**Q11: What was the main finding of the ablation study?**
- *Short Answer*: Lexical-only models fail on CDNs ($\text{FPR}=38.26\%$), while dual-view fusion slashes missed attacks by $57.4\%$.
- *Detailed Answer*: Combining character orthography with temporal behavior is essential: lexical alone suffers massive false alarms on CDNs, and behavioral alone misses low-and-slow attacks.
- *Common Mistake*: Claiming single-view models performed just as well.

**Q12: How much telemetry / query volume does DeepDNS save?**
- *Short Answer*: **$56.2\%$ on In-Distribution** and **$65.7\%$ on Out-of-Distribution** data.
- *Detailed Answer*: Mean stopping horizon drops from 30 queries down to $13.13$ (ID) and $10.30$ (OOD), classifying $99.9\%$ of OOD streams early.
- *Common Mistake*: Stating query savings is just a cosmetic metric.

---

# PART 34 — MULTI-DURATION PRESENTATION SCRIPTS

### 30-Second Elevator Pitch
> *"DeepDNS is an adaptive deep learning framework that detects DNS exfiltration in real time. Instead of waiting for a rigid buffer of 30 queries, our Adaptive Evidence Controller evaluates both domain character patterns and temporal query bursts, stopping observation the moment confidence bounds are crossed. DeepDNS cuts time-to-detection and telemetry overhead by over 65% while achieving 99.2% recall on unseen attack modalities."*

### 2-Minute Executive Summary
> *"DNS port 53 is universally open across corporate firewalls, making it a prime target for covert data exfiltration. Conventional machine learning detectors suffer from two major flaws: they force network firewalls to buffer fixed windows of 30+ queries—delaying alerts—and single-view models suffer catastrophic 38% false alarm rates on legitimate CDNs.*
> 
> *DeepDNS solves this through a dual-view sequential architecture. A Character-CNN inspects subdomain orthography while a Temporal GRU tracks 12 causal behavioral features. These views are fused to emit instantaneous attack probabilities.*
> 
> *Our Adaptive Evidence Controller evaluates accumulated confidence at checkpoints $K \in \{5, 10, 15, 20, 25, 30\}$. Blatant attacks and obvious benign queries stop at $K=5$, saving 83% of queries, while ambiguous streams accumulate evidence safely up to 30 queries.*
> 
> *Evaluated on over 33,000 test windows, DeepDNS achieves 99.21% Recall and 0.50% FPR on unseen Leave-One-Modality-Out payloads, reducing observation latency by 65.7%."*

---

# PART 35 — MASTER CHEAT SHEET

| Item | Actual DeepDNS Value |
| :--- | :--- |
| **Primary Dataset** | CIC-Bell-DNS-EXF-2021 (697,120 stateless rows) |
| **Causal Features** | 12 features (past-only lookback $t \le k$) |
| **In-Distribution Test Size** | 13,084 sequence windows |
| **Out-of-Distribution (LOMO) Test Size**| 20,683 sequence windows |
| **Behavioral Model** | 1-Layer Temporal GRU (hidden dim=64) |
| **Lexical Model** | 1D Character-CNN (kernels 3, 5, 7; output dim=128) |
| **Fusion Network** | Late-Fusion MLP ($128 \to 64 \to 32 \to 2$) |
| **AEC Discrete Horizons** | $K \in \{5, 10, 15, 20, 25, 30\}$ |
| **In-Distribution Thresholds** | $\tau_{\text{attack}} = \mathbf{0.95}, \quad \tau_{\text{benign}} = \mathbf{0.15}$ |
| **Out-of-Distribution Thresholds**| $\tau_{\text{attack}} = \mathbf{0.80}, \quad \tau_{\text{benign}} = \mathbf{0.01}$ |
| **Sequential CUSUM Parameters** | $h_{\text{attack}} = 4.0, \quad h_{\text{benign}} = 4.0, \quad \text{drift } \gamma = 0.0$ |
| **In-Distribution AEC Performance** | Recall: **98.17%**, FPR: **0.1238%**, F1: **0.9894**, $\bar{K}^*: \mathbf{13.13}$ |
| **Out-of-Distribution AEC Performance**| Recall: **99.21%**, FPR: **0.5066%**, F1: **0.9941**, $\bar{K}^*: \mathbf{10.30}$ |
| **Telemetry / Query Savings** | **56.23% (ID)** / **65.67% (OOD)** |
| **Lexical-Only Ablation Failure** | FPR: **38.2641%** ($FP = 3,399$) |
| **Dual-View Synergy Gain** | **57.4% reduction in False Negatives** ($FN: 47 \to 20$) |
| **Verification Suite** | **76 / 76 PyTest tests passing (100%)** |
| **Serving & Web Stack** | FastAPI + WebSockets (backend) & React 18 + Vite (frontend) |

---

# THE 20 THINGS WE ABSOLUTELY MUST KNOW

1. **The Core Novelty**: DeepDNS is an **Adaptive Multi-View Sequential Detection Framework** that terminates observation early when confidence bounds are crossed.
2. **The Problem with Fixed Windows**: Traditional detectors force a static buffer of 30+ queries, delaying detection until data has already left the network.
3. **The 12 Causal Features**: All behavioral features are strictly past-to-present ($t \le k$), guaranteeing zero forward-looking data leakage.
4. **Behavioral Branch**: A 2-Layer Temporal GRU (hidden dim 64) modeling inter-arrival timing ($\Delta t$), query rates, and burstiness.
5. **Lexical Branch**: A 1D Character-CNN (kernels 3, 5, 7; output dim 128) extracting sub-word n-gram character patterns from raw subdomains.
6. **Multi-View Fusion**: Late-fusion MLP combining both branches into a single 64-D representation emitting step probability $p_k$.
7. **Why Lexical-Only Fails**: Fails with a **$38.26\%$ FPR** because modern CDNs and cloud services generate random, high-entropy subdomains.
8. **Why Behavioral-Only is Insufficient**: Misses subtle, low-and-slow exfiltrations ($FN = 47$); dual-view fusion slashes missed attacks by **$57.4\%$**.
9. **Adaptive Horizons**: Evidence is evaluated at discrete checkpoints $K \in \{5, 10, 15, 20, 25, 30\}$.
10. **In-Distribution Thresholds**: $\tau_{\text{attack}} = \mathbf{0.95}$ and $\tau_{\text{benign}} = \mathbf{0.15}$, calibrated on validation data to enforce $\text{FPR} \le 0.15\%$.
11. **Out-of-Distribution Thresholds**: $\tau_{\text{attack}} = \mathbf{0.80}$ and $\tau_{\text{benign}} = \mathbf{0.01}$, calibrated to maximize detection sensitivity on unseen attacks.
12. **Out-of-Distribution (LOMO) Recall**: Achieves **$99.21\%$ Recall** and **$0.9941$ F1** on 100% unseen Video and Text exfiltration payloads.
13. **Observation Latency Reduction**: Cuts query consumption by **$65.67\%$ on OOD** ($\bar{K}^* = 10.30$) and **$56.23\%$ on ID** ($\bar{K}^* = 13.13$).
14. **Sequential CUSUM Role**: Serves as a statistical change-point baseline ($h=4.0, \gamma=0.0$), showing rapid ID detection ($\bar{K}^*=9.22$).
15. **Dataset Integrity**: Trained on CIC-Bell-DNS-EXF-2021 stateless captures (697,120 rows); stateful CSVs were excluded due to leakage.
16. **Bootstrap Rigor**: 1,000 iterations (seed=42) confirm 95% confidence intervals (e.g., OOD Recall: $[99.05\%, 99.36\%]$).
17. **Serving Architecture**: FastAPI backend offering REST (`/detect`, `/health`) and real-time WebSockets (`/ws/stream/{stream_id}`).
18. **Frontend Dashboard**: React 18 + Vite interface with live SVG evidence trajectory plotting, feature inspection, and synthetic stream generation.
19. **Test Suite Status**: **76 / 76 unit, integration, and WebSocket tests passing (100%)**.
20. **Bounded Fallback Safety**: If evidence is ambiguous at $K=30$, a forced decision is executed at threshold $0.50$, ensuring no flow is buffered forever.
