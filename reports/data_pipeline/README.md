# DeepDNS Data Pipeline & Engineering Documentation

## 1. Data Sources & Physical Layout
The DeepDNS data pipeline ingests two heterogeneous DNS security datasets:
1. **CIC-Bell-DNS-EXF-2021**:
   - Location: `data/raw/cic_bell_dns_exf_2021/`
   - Content: 18 PCAP capture sessions organized into 36 CSV files (18 stateless and 18 stateful).
   - Stateless volume: 697,120 individual query/response records with microsecond timestamps.
   - Stateful volume: 239,337 window-aggregated records (excluded from direct model input).
2. **DNS Threats Dataset**:
   - Location: `data/raw/dns_threats/original/`
   - Content: `train_combined_multiclass.csv.gz` (2,482,810 domains) and `test_combined_multiclass.csv.gz` (620,703 domains).
   - Used strictly for Lexical Character-CNN representation learning and out-of-distribution domain validation.

## 2. Directory Assumptions & Label Parsing
CIC-Bell encodes ground truth via filesystem directory hierarchy and filename conventions rather than explicit label columns:
- Directories containing `/Benign/` or filenames containing `benign` map to **Label 0 (Benign)**.
- Directories containing `/Attacks/` without benign indicators map to **Label 1 (Attack/Exfiltration)**.
- Attack modality (`text`, `audio`, `compressed`, `video`, `executable`, `image`) is parsed deterministically from filename tokens.
- Capture intensity (`light`, `heavy`, `standard`) is extracted from folder and file prefix tokens.
- Handled deterministically by `src/data/labels.py::parse_cic_bell_label()`.

## 3. The 12 Approved Causal Features
Every query $x_i$ is mapped to a 12-dimensional numerical vector derived exclusively from information available at or before timestamp $t_i$:

| Index | Feature Name | Mathematical Derivation | Causal Justification |
| :--- | :--- | :--- | :--- |
| 1 | `inter_arrival_time` | $\ln(1 + \max(t_i - t_{i-1}, 0))$ with $\Delta t_0 = 0.0$ | Reflects query cadence strictly backwards from past query. |
| 2 | `fqdn_length` | Raw count of characters in FQDN | Computed immediately upon packet inspection. |
| 3 | `subdomain_length` | Character length of subdomain labels | Computed immediately upon packet inspection. |
| 4 | `char_entropy` | Shannon character entropy of FQDN | Computed on query string at arrival time $t_i$. |
| 5 | `digit_ratio` | $\text{numeric} / (\text{FQDN\_count} + 10^{-6})$ | Proportional ratio at query arrival. |
| 6 | `uppercase_ratio` | $\text{upper} / (\text{FQDN\_count} + 10^{-6})$ | Proportional ratio at query arrival. |
| 7 | `special_ratio` | $\text{special} / (\text{FQDN\_count} + 10^{-6})$ | Proportional ratio at query arrival. |
| 8 | `label_count` | Number of dot-separated labels | Structural query property at arrival. |
| 9 | `max_label_length` | Maximum length of single label | Structural query property at arrival. |
| 10 | `avg_label_length` | Average length across labels | Structural query property at arrival. |
| 11 | `has_subdomain` | Binary indicator $\{0, 1\}$ | Structural query property at arrival. |
| 12 | `payload_len` | Packet payload byte length | Observed on wire at arrival $t_i$. |

## 4. Why Specific Features Are Excluded (Leakage & Non-Causality Prevention)
- **`sld`**: Excluded because the attack exfiltration traffic in CIC-Bell was generated against fixed synthetic test domains (e.g. `tunnel.example.com`). Including `sld` causes models to memorize the lab domain hash rather than generalizable tunneling behavior.
- **`longest_word`**: Excluded due to inconsistent string parsing and dictionary lookup heuristics that introduce ungrounded noise.
- **`timestamp` (Absolute)**: Excluded as a raw feature to prevent the model from memorizing the capture date/time. Only relative $\Delta t$ is permitted.
- **All 27 raw `stateful_features-*.csv` columns**: Excluded because they are retrospective window summaries ($N_{stateful} < N_{stateless}$ with a 65.7% row mismatch) and serialize raw Python set literals (`"{'PTR'}"`, `"set()"`). Stateful temporal memory is learned causally inside our GRU.

## 5. Sequence Windowing ($K \in [5, 30]$)
- Observation streams are processed per PCAP capture in strict chronological order.
- Prefix windows start at $K_{min} = 5$ observations: $[x_1 \dots x_5], [x_1 \dots x_6], \dots, [x_1 \dots x_{30}]$.
- Sliding horizon windows continue across the stream: $[x_i \dots x_{i+30}]$.
- Window tensors are padded to length 30 with trailing zeros and accompanied by exact `seq_len` scalars.

## 6. Leakage-Safe Grouped Dataset Splitting
- Partitioning is performed strictly at the **PCAP capture level (`capture_id`)**, never across shuffled rows.
- **Train (69.2%, 12 captures, 524,027 rows)**: Model training and FeatureScaler parameter fitting.
- **Validation (13.6%, 3 captures, 103,119 rows)**: Hyperparameter tuning and early stopping (`heavy_image`, `light_audio`, `benign_heavy_2`).
- **Test (17.2%, 3 captures, 130,065 rows)**: Out-of-distribution evaluation (`heavy_video`, `light_text`, `benign_2`).
- Cross-split capture leakage is cryptographically verified to be 0.00%.

## 7. DNS Threats Lexical Processing
- `DNSThreatsLexicalTokenizer` maps raw domain ASCII strings to fixed-size token ID sequences ($L=128$).
- Token vocabulary: Index 0 (`<pad>`), Index 1 (`<unk>`), Indices 2..42 (ASCII alphanumeric + punctuation).

## 8. Running the Pipeline
To execute the complete pipeline and regenerate all manifests and scalers:
```bash
python scripts/dataset_loader.py
```

To execute the unit test suite:
```bash
python -m pytest -v tests/
```

## 9. Generated Artifacts
- `reports/data_pipeline/dataset_inventory.csv`: Complete filesystem audit metadata.
- `reports/data_pipeline/stateless_schema_validation.csv`: Schema conformance per file.
- `reports/data_pipeline/split_manifest.json`: Verified session-grouped partition manifest.
- `reports/data_pipeline/dataset_validation.json`: Summary execution metrics.
- `data/processed/feature_scaler.json`: Scaler parameters (mean/std) fit strictly on train captures.
