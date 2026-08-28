# DeepDNS Validation Repository & Infrastructure Audit

## 1. Dataset & Capture Inventory

### CIC-Bell DNS Exfiltration Dataset (2021)
The primary sequential DNS dataset consists of 18 capture sessions (PCAP-derived stateless CSVs) containing 757,211 total DNS query records.

| Capture ID | File Name | Modality | Intensity | Class Label | Query Count | Current Split |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `heavy_audio` | `stateless_features-heavy_audio.pcap.csv` | Audio | Heavy | 1 (Attack) | 35,795 | Train |
| `heavy_compressed` | `stateless_features-heavy_compressed.pcap.csv` | Compressed | Heavy | 1 (Attack) | 35,746 | Train |
| `heavy_exe` | `stateless_features-heavy_exe.pcap.csv` | Exe | Heavy | 1 (Attack) | 34,629 | Train |
| `heavy_text` | `stateless_features-heavy_text.pcap.csv` | Text | Heavy | 1 (Attack) | 71,102 | Train |
| `benign_heavy_1` | `stateless_features-benign_heavy_1.pcap.csv` | None | Heavy | 0 (Benign) | 61,567 | Train |
| `benign_heavy_3` | `stateless_features-benign_heavy_3.pcap.csv` | None | Heavy | 0 (Benign) | 71,012 | Train |
| `light_compressed` | `stateless_features-light_compressed.pcap.csv` | Compressed | Light | 1 (Attack) | 10,241 | Train |
| `light_exe` | `stateless_features-light_exe.pcap.csv` | Exe | Light | 1 (Attack) | 6,450 | Train |
| `light_image` | `stateless_features-light_image.pcap.csv` | Image | Light | 1 (Attack) | 524 | Train |
| `light_video` | `stateless_features-light_video.pcap.csv` | Video | Light | 1 (Attack) | 4,371 | Train |
| `light_benign` | `stateless_features-light_benign.pcap.csv` | None | Light | 0 (Benign) | 60,091 | Train |
| `benign_1` | `stateless_features-benign_1.pcap.csv` | None | Standard | 0 (Benign) | 132,499 | Train |
| `heavy_image` | `stateless_features-heavy_image.pcap.csv` | Image | Heavy | 1 (Attack) | 36,386 | Validation |
| `benign_heavy_2` | `stateless_features-benign_heavy_2.pcap.csv` | None | Heavy | 0 (Benign) | 49,115 | Validation |
| `light_audio` | `stateless_features-light_audio.pcap.csv` | Audio | Light | 1 (Attack) | 17,618 | Validation |
| `heavy_video` | `stateless_features-heavy_video.pcap.csv` | Video | Heavy | 1 (Attack) | 38,012 | Test |
| `light_text` | `stateless_features-light_text.pcap.csv` | Text | Light | 1 (Attack) | 3,479 | Test |
| `benign_2` | `stateless_features-benign_2.pcap.csv` | None | Standard | 0 (Benign) | 88,574 | Test |

### External Dataset: `dns_threats`
- Path: `data/raw/dns_threats/original/test_combined_multiclass.csv.gz` (620,703 rows)
- Path: `data/raw/dns_threats/original/train_combined_multiclass.csv.gz` (2,482,810 rows)
- Format: Static pointwise `['domain', 'class']` without timing or PCAP sequential packets.

---

## 2. Modality Inventory & Cross-Partition Distribution

The 6 attack modalities are distributed as follows across captures:
- **Audio**: `heavy_audio` (Train), `light_audio` (Val)
- **Compressed**: `heavy_compressed` (Train), `light_compressed` (Train)
- **Exe**: `heavy_exe` (Train), `light_exe` (Train)
- **Image**: `light_image` (Train), `heavy_image` (Val)
- **Text**: `heavy_text` (Train), `light_text` (Test)
- **Video**: `light_video` (Train), `heavy_video` (Test)

In the standard `split_manifest.json`, modalities were stratified across partitions. To perform strict **Leave-One-Modality-Out (LOMO) OOD evaluation**, we must hold out entire modalities completely from training (e.g. holding out ALL `video` and ALL `text` captures so the model has 0 training exposure to those modalities).

---

## 3. Existing Reusable Components

1. **Feature Extraction**: `CausalFeatureExtractor` in `src/data/features.py` (guarantees 0 forbidden columns).
2. **Feature Scaler**: `FeatureScaler` in `src/data/features.py` (fitted strictly on training partitions).
3. **Lexical Tokenizer**: `CharacterTokenizer` in `src/models/char_cnn.py` (deterministic 45-character ASCII vocabulary).
4. **Sequence Construction**: `SequenceBuilder` in `src/data/sequences.py` (enforces capture-boundary isolation).
5. **Models**:
   - `TemporalGRUClassifier` in `src/models/temporal_gru.py`
   - `CharacterCNNEncoder` in `src/models/char_cnn.py`
   - `DeepDNSMultiViewClassifier` in `src/models/multiview_fusion.py`
6. **Evaluation Engine**: `compute_classification_metrics` in `src/evaluation/metrics.py`.

---

## 4. OOD & Evaluation Infrastructure Gaps

- **Missing**:
  1. Formal LOMO OOD split manifest builder (`src/data/ood_splits.py`).
  2. OOD split audit script (`scripts/audit_ood_split.py`).
  3. Evaluation unit audit script (`scripts/audit_evaluation_unit.py`).
  4. Capture-level robustness aggregation script (`scripts/audit_capture_level_eval.py`).
  5. Dedicated OOD training/eval script (`scripts/train_multiview_ood.py`).
  6. Zero-shot external `dns_threats` lexical evaluation script (`scripts/evaluate_dns_threats.py`).

---

## 5. Recommended Experiment Order

1. **Audit & Build OOD Split**: Construct a strict Leave-One-Modality-Out split holding out Video and Text completely from training.
2. **Run Evaluation-Unit & Capture-Level Audits**: Verify sample correlation and capture-level robustness on existing Dual-View results.
3. **Run Zero-Shot `dns_threats` Lexical Test**: Evaluate the pre-trained Character-CNN on 620k real-world external domains without retraining.
4. **Prepare `train_multiview_ood.py`**: Await user confirmation before launching OOD training.
