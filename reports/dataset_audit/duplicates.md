# Forensic Duplication Analysis

## Multi-Level Duplicate Audit

| Duplication Level | Total Records | Duplicate Count | Duplicate Rate | Assessment |
| :--- | :--- | :--- | :--- | :--- |
| **Exact Row (Timestamp + Features)** | 697,120 | **518** | **0.07%** | Negligible; caused by millisecond-identical dual queries. |
| **Feature Vector Only (Excl. Timestamp)** | 697,120 | **594,789** | **85.32%** | High repetition of structural domain shapes. |
| **Cross-Class Collision (Attack vs. Benign)** | - | **79** | **0.01%** | Minimal collision (e.g., standard short PTR / root queries). |

## Forensic Implications
1. **Why 85.32% Feature Vector Repetition Occurs**:
   - Tunneling tools (e.g. `iodine`, `dnscat2`) fragment data into fixed-size chunks (e.g., exactly 64 base32 characters per label). Consequently, consecutive queries have identical `FQDN_count`, `subdomain_length`, `entropy`, and `numeric` counts.
   - Benign DNS traffic frequently issues identical repetitive queries for background OS telemetry (e.g., `time.windows.com`, `connectivitycheck.gstatic.com`).
2. **Train/Test Split Danger**:
   - Splitting randomly across rows (k-fold row shuffle) will cause massive **data leakage** because identical feature vectors with neighboring timestamps will sit in both train and test partitions.
   - **Mandatory Solution**: Must perform **Group-by-PCAP (Session-based) splitting** or **Temporal block splitting** (Train time < Val time < Test time).
