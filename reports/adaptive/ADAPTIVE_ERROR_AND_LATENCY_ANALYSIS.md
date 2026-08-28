# DeepDNS Adaptive Evidence Controller: Horizon-Wise Error & Latency Analysis

## 1. Executive Summary
This study provides a rigorous forensic decomposition of the **Adaptive Evidence Controller (AEC)** and **Sequential CUSUM Detector**, analyzing:
1. **Horizon-Wise Decision Distribution**: At which observation horizons ($K \in \{5, 10, 15, 20, 25, 30\}$) decisions are triggered.
2. **Error Concentration**: Where False Positives (FP) and False Negatives (FN) predominantly occur.
3. **Observation Latency & Query Savings**: Exact query volume reduction relative to static $K=30$ buffering.
4. **Statistical Uncertainty**: 1,000-iteration non-parametric Bootstrap 95% Confidence Intervals for all primary metrics.

## 2. Global Performance & Statistical Confidence Matrix

| Split | Strategy | Detection Rate (Recall) [95% CI] | FPR [95% CI] | F1-Score [95% CI] | Mean Horizon ($\bar{K}$) [95% CI] | Query Savings |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| In-Distribution | **AEC** (Calibrated) | 98.17% [97.74%, 98.56%] | 0.1238% [0.0561%, 0.1995%] | 0.9894 [0.9872, 0.9916] | 13.13 [12.93, 13.31] | **56.2%** |
| In-Distribution | **CUSUM** | 99.12% [98.84%, 99.38%] | 0.5854% [0.4396%, 0.7422%] | 0.9894 [0.9872, 0.9915] | 9.22 [9.11, 9.31] | **69.3%** |
| Held-Out OOD | **AEC** (Calibrated) | 99.21% [99.05%, 99.36%] | 0.5066% [0.373%, 0.6536%] | 0.9941 [0.9931, 0.995] | 10.30 [10.23, 10.37] | **65.7%** |
| Held-Out OOD | **CUSUM** | 96.17% [95.81%, 96.47%] | 0.1801% [0.1012%, 0.2704%] | 0.9798 [0.9779, 0.9814] | 12.96 [12.86, 13.06] | **56.8%** |

## 3. In-Distribution Horizon Error Decomposition (AEC)
| Horizon ($K$) | Total Stopped | % of Total | True Positives ($TP$) | True Negatives ($TN$) | False Positives ($FP$) | False Negatives ($FN$) | Horizon Error Rate | Primary Traffic Stopped |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $K=5$ | 8,310 | 63.51% | 0 | 8,257 | 0 | 53 | 0.64% | video: 50, text: 3 |
| $K=10$ | 507 | 3.87% | 0 | 502 | 1 | 4 | 0.99% | video: 4, none: 503 |
| $K=15$ | 88 | 0.67% | 4 | 80 | 1 | 3 | 4.55% | video: 7, none: 81 |
| $K=20$ | 55 | 0.42% | 35 | 18 | 0 | 2 | 3.64% | video: 34, text: 3 |
| $K=25$ | 197 | 1.51% | 183 | 9 | 2 | 3 | 2.54% | video: 166, text: 20 |
| $K=30$ | 3,927 | 30.01% | 3,902 | 6 | 7 | 12 | 0.48% | video: 3566, text: 348 |

## 4. Held-Out OOD Horizon Error Decomposition (AEC)
| Horizon ($K$) | Total Stopped | % of Total | True Positives ($TP$) | True Negatives ($TN$) | False Positives ($FP$) | False Negatives ($FN$) | Horizon Error Rate | Primary Traffic Stopped |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| $K=5$ | 8,597 | 41.57% | 272 | 8,264 | 20 | 41 | 0.71% | text: 207, video: 106 |
| $K=10$ | 3,144 | 15.20% | 2,633 | 491 | 10 | 10 | 0.64% | text: 1708, video: 935 |
| $K=15$ | 8,106 | 39.19% | 8,006 | 72 | 15 | 13 | 0.35% | text: 5119, video: 2900 |
| $K=20$ | 786 | 3.80% | 768 | 9 | 0 | 9 | 1.15% | text: 450, video: 327 |
| $K=25$ | 36 | 0.17% | 25 | 1 | 0 | 10 | 27.78% | text: 22, video: 13 |
| $K=30$ | 14 | 0.07% | 3 | 1 | 0 | 10 | 71.43% | text: 4, video: 9 |

## 5. Key Forensic Findings
1. **Early Benign Dismissal**: Under AEC, **$63.51\%$ of In-Distribution traffic and $41.57\%$ of OOD traffic is classified decisively at $K=5$**, overwhelmingly consisting of benign traffic dismissed early with near-zero false alarms.
2. **Sequential Evidence Saturation**: For unseen attack modalities (Video, Text), high confidence is achieved primarily at **$K=10$ and $K=15$** ($54.39\%$ of decisions), proving that recurrent evidence accumulation quickly resolves domain ambiguity.
3. **CUSUM vs AEC Trade-Off**: CUSUM is a more aggressive filter on In-Distribution data (stopping in $\bar{K} = 9.22$ queries with $99.12\%$ recall), whereas AEC provides superior OOD detection sensitivity ($99.21\%$ Recall vs $96.17\%$ for CUSUM).
4. **Computational & Query Savings**: AEC reduces observation buffer requirements by **$56.2\%$ (ID)** and **$65.7\%$ (OOD)** relative to static $K=30$ buffering.
