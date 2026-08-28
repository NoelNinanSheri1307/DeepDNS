# CIC-Bell Label & Class Distribution Analysis

## Label Representation
In CIC-Bell, labels are **implicitly encoded in file paths and filenames**, rather than stored as explicit columns in the raw CSV files.

### Binary Classification Distribution (Stateless)
- **Benign Records (Class 0)**: 402,767 (57.78%)
- **Attack / Exfiltration Records (Class 1)**: 294,353 (42.22%)
- **Total Records**: 697,120
- **Imbalance Ratio**: 1.37 : 1 (Well-balanced for binary classification).

### Multi-Class Modality Distribution
| Modality / Data Exfiltrated | Record Count | Percentage | Class Role |
| :--- | :--- | :--- | :--- |
| `Benign` (Normal Web / DNS) | 402,767 | 57.78% | Negative Class |
| `Text` (ASCII / Code / Docs) | 74,581 | 10.70% | High-entropy Base64/Base32 chunks |
| `Audio` (MP3 / WAV chunks) | 53,413 | 7.66% | Binary stream exfiltration |
| `Compressed` (ZIP / GZ) | 45,987 | 6.60% | Maximum entropy payloads |
| `Video` (MP4 streaming) | 42,383 | 6.08% | Sustained high-volume tunneling |
| `Executable` (EXE / DLL) | 41,079 | 5.89% | Binary header + payload |
| `Image` (JPEG / PNG) | 36,910 | 5.29% | Structured binary data |

### Traffic Intensity Breakdown
- **Heavy Attacks / Background**: 433,364 records (62.17%)
- **Standard Benign (Baseline)**: 221,073 records (31.71%)
- **Light Attacks / Background**: 42,683 records (6.12%)

> [!NOTE]
> Ground truth labels must be mapped during dataset ingestion using regex parsing of filename paths: `is_attack = ('attack' in path.lower() and 'benign' not in filename.lower())`.
