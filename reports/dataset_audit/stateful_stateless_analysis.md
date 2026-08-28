# Forensic Alignment: Stateful vs. Stateless Analysis

## The Row Alignment Problem
A critical assumption in multi-view architectures is that different feature sets can be aligned row-by-row. Our analysis reveals that **Stateless and Stateful files CANNOT be aligned 1:1**.

### Quantitative Discrepancy
Across all 18 PCAP captures:
- Total Stateless Records: **697,120**
- Total Stateful Records: **239,337**
- Total Discrepancy: **457,783 missing stateful records (65.67% fewer rows)**
- Row Count Match Rate: **0.00% (0 out of 18 pairs matched)**

### Pairwise Discrepancy Table
| Scenario | Attack Type / File | Stateless Rows | Stateful Rows | Discrepancy (Δ) | Stateful % of Stateless |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Heavy Attack | audio | 35,795 | 10,735 | -25,060 | 30.0% |
| Heavy Attack | compressed | 35,746 | 10,424 | -25,322 | 29.2% |
| Heavy Attack | exe | 34,629 | 9,980 | -24,649 | 28.8% |
| Heavy Attack | image | 36,386 | 11,076 | -25,310 | 30.4% |
| Heavy Attack | text | 71,102 | 18,916 | -52,186 | 26.6% |
| Heavy Attack | video | 38,012 | 10,897 | -27,115 | 28.7% |
| Heavy Benign | benign | 61,567 | 22,774 | -38,793 | 37.0% |
| Heavy Benign | benign | 49,115 | 19,660 | -29,455 | 40.0% |
| Heavy Benign | benign | 71,012 | 26,582 | -44,430 | 37.4% |
| Light Attack | audio | 17,618 | 4,246 | -13,372 | 24.1% |
| Light Attack | compressed | 10,241 | 2,904 | -7,337 | 28.4% |
| Light Attack | exe | 6,450 | 1,836 | -4,614 | 28.5% |
| Light Attack | image | 524 | 143 | -381 | 27.3% |
| Light Attack | text | 3,479 | 921 | -2,558 | 26.5% |
| Light Attack | video | 4,371 | 1,245 | -3,126 | 28.5% |
| Standard_benign Benign | benign | 132,499 | 52,438 | -80,061 | 39.6% |
| Standard_benign Benign | benign | 88,574 | 34,560 | -54,014 | 39.0% |

## Root Cause Analysis
1. **Window Aggregation Mechanism**: Stateless features are emitted on every single DNS packet ($N_{stateless} = N_{packets}$). Stateful features are emitted over temporal time windows or distinct resource record events ($N_{stateful} < N_{packets}$).
2. **Absence of Shared Primary Key**: There is no packet ID, query hash, or sequence index linking a row in `stateless_features-*.csv` to `stateful_features-*.csv`.
3. **Absence of Timestamp in Stateful CSVs**: Stateful CSVs omit the `timestamp` column entirely, making exact temporal interpolation impossible without making strong heuristic assumptions.

## Look-Ahead & Non-Causal Leakage in Stateful Features
| Stateful Feature | Causal Status | Leakage Risk Level | Technical Justification |
| :--- | :--- | :--- | :--- |
| `ttl_variance` | Retrospective Window | **HIGH (LEAKAGE-PRONE)** | Computed across entire aggregated batches; reflects post-hoc statistical distribution. |
| `distinct_domains` | Global / Session Window | **HIGH (LEAKAGE-PRONE)** | Cardinality estimates depend on multi-query window horizon. |
| `unique_country`, `unique_asn` | Session-level lookup | **HIGH (LEAKAGE-PRONE)** | Contains synthetic lab environment artifacts (`set()`). |
| `A_frequency` - `TXT_frequency` | Cumulative window | **MEDIUM (SUSPICIOUS)** | Must be re-implemented as causal rolling counts ($t_0 \to t$) in online streaming. |

## Architectural Decision
> [!CAUTION]
> **Stateful CSVs must NOT be joined statically by row index.**
> 
> Doing so causes catastrophic row misalignment (e.g. aligning packet 10,000 in stateless with packet 30,000 in stateful).
> 
> **Solution**: Use the 15 **Stateless features** as the atomic query-level sequence ($x_1, x_2, \dots, x_t$) and compute **dynamic, causal running stateful representations** in our own GRU / temporal sequence model!
