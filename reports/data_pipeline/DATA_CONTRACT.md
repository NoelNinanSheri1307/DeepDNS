# DeepDNS Data Contract for Downstream Modeling Code

## Purpose
This document defines the strict mathematical and software contract that all downstream DeepDNS models (Character-CNN, Behavioral MLP, GRU, Multi-view Fusion, Deep Ensembles, Adaptive Controller) can rely upon.

---

## 1. Sequential Behavioral Stream Contract (`StreamingSequenceDataset`)

### Batch Tensors Definition
When iterating over a PyTorch DataLoader wrapping `StreamingSequenceDataset`:
```python
for padded_seqs, seq_lens, labels, meta in dataloader:
    ...
```

| Output Component | PyTorch Tensor Shape | Data Type | Value Range / Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| `padded_seqs` | `(Batch_Size, 30, 12)` | `torch.float32` | Standardized (clipped $\pm 8\sigma$) | 12 causal normalized features padded with 0.0 for $t > \text{seq\_len}$. |
| `seq_lens` | `(Batch_Size,)` | `torch.long` / `int` | $5 \le \text{seq\_len} \le 30$ | Actual unpadded evidence horizon length $k$. |
| `labels` | `(Batch_Size,)` | `torch.float32` | Binary: $\{0.0, 1.0\}$ | Ground truth (0 = Benign, 1 = Attack/Exfiltration). |
| `meta` | Dictionary | Python primitives | Keys detailed below | Contextual capture information for evaluation. |

### Metadata Dictionary Specification
```python
meta = {
    "capture_id": str,          # e.g., "heavy_audio", "benign_2"
    "start_idx": int,           # Starting row index within the capture PCAP
    "end_idx": int,             # Ending row index within the capture PCAP
    "seq_len": int,             # Horizon length k
    "attack_modality": str|None,# "text", "audio", "compressed", "video", "exe", "image", or None
    "intensity": str,           # "heavy", "light", or "standard"
}
```

### 12-Feature Channel Mapping (Dimension 2: Indices 0 to 11)
```python
CHANNEL_MAP = {
    0: "inter_arrival_time",  # ln(1 + dt)
    1: "fqdn_length",         # Standardized character length
    2: "subdomain_length",    # Standardized subdomain length
    3: "char_entropy",        # Standardized Shannon entropy
    4: "digit_ratio",         # numeric / (FQDN_count + 1e-6)
    5: "uppercase_ratio",     # upper / (FQDN_count + 1e-6)
    6: "special_ratio",       # special / (FQDN_count + 1e-6)
    7: "label_count",         # Standardized label count
    8: "max_label_length",    # Standardized max label length
    9: "avg_label_length",    # Standardized average label length
    10: "has_subdomain",      # Binary indicator {0.0, 1.0}
    11: "payload_len"         # Standardized record byte length
}
```

---

## 2. Lexical Domain Stream Contract (`DNSThreatsDataset`)

### Batch Tensors Definition
```python
for token_ids, labels in dns_threats_dataloader:
    ...
```

| Output Component | PyTorch Tensor Shape | Data Type | Value Range / Constraints | Description |
| :--- | :--- | :--- | :--- | :--- |
| `token_ids` | `(Batch_Size, 128)` | `torch.long` | $[0, 42]$ | ASCII character token IDs (0 = `<pad>`, 1 = `<unk>`). |
| `labels` | `(Batch_Size,)` | `torch.long` | $\{0, 1, 2\}$ | Multiclass target (0 = Benign, 1 = Malicious, 2 = Suspicious). |

---

## 3. Invariant Scientific Constraints
Downstream models and training scripts **MUST NOT**:
1. Access or ingest columns outside the approved 12 causal features (`timestamp`, `sld`, `longest_word`, and all stateful CSV columns are strictly prohibited).
2. Shuffle sequences across PCAP capture sessions during cross-validation (splitting must use `reports/data_pipeline/split_manifest.json`).
3. Compute normalization statistics across test or validation sets (must use `data/processed/feature_scaler.json`).
4. Look ahead into future queries $t' > k$ during evidence accumulation (evidence controller must operate as an online causal filter).
