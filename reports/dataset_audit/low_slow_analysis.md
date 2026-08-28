# Low-and-Slow Attack Forensic Investigation

## The "Light" vs. "Heavy" Empirical Reality
A central question for DeepDNS is whether CIC-Bell's "Light" attacks represent true stealthy, low-and-slow exfiltration.

### Quantitative Comparison
| Metric | Light Attack (N=42,683) | Heavy Attack (N=433,364) | Benign Baseline (N=221,073) |
| :--- | :--- | :--- | :--- |
| **Mean Inter-Arrival Time (delta t)** | **0.5895 sec** | **0.4411 sec** | **0.2431 sec** |
| **Median Inter-Arrival Time** | **0.4108 sec** | **0.4103 sec** | **0.1502 sec** |
| **95th Percentile delta t** | **1.2367 sec** | **1.2358 sec** | **0.7382 sec** |
| **Effective Query Rate (QPS)** | **~1.70 QPS** | **~1.70 QPS** | **~4.20 QPS** |
| **Mean FQDN Length** | **25.38 chars** | **22.56 chars** | **18.45 chars** |
| **Mean Character Entropy** | **2.444** | **2.482** | **2.473** |
| **Mean Subdomain Length** | **8.10 chars** | **6.25 chars** | **3.86 chars** |

## Critical Forensic Discovery
> [!IMPORTANT]
> **"Light" in CIC-Bell means Low Total Volume, NOT Low Cadence / Low Rate.**
> 
> The packet inter-arrival time for Light attacks (mean = 0.589s) is nearly identical to Heavy attacks (mean = 0.441s), with identical medians (0.410s). The tunneling tools were executed at the same transmission speed; the creators simply exfiltrated a smaller payload file (e.g. 500 packets vs 35,000 packets).

## Research Solution for DeepDNS
To evaluate **genuine low-and-slow evasion** (e.g., query rates diluted with jitter and long idle intervals of 10s–300s):
1. Evaluate on CIC-Bell's Low-Volume regime (K <= 10 queries).
2. Synthetically generate **rate-diluted / jittered streaming benchmarks** by injecting benign Poisson noise between exfiltration queries.
