# Phase 3: Capture-Level Robustness Audit Report

- **Capture-Level Accuracy**: **100.0%** (100% Robustness)
- **Attack Capture Recall**: **100.0%** (4/4 Unseen Modality Attacks Detected)
- **Benign Capture FPR**: **0.0%** (0 False Alarms)

### Per-Capture OOD Performance Breakdown

| Capture ID | Modality | Intensity | Windows | Mean Prob | Median Prob | Attack Window % | Pred | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `heavy_text` | text | heavy | 7,136 | 0.919 | 0.9217 | 99.8% | 1 | **CORRECT_ATTACK** |
| `light_video` | video | light | 463 | 0.8927 | 0.9217 | 97.2% | 1 | **CORRECT_ATTACK** |
| `heavy_video` | video | heavy | 3,827 | 0.9156 | 0.9217 | 99.4% | 1 | **CORRECT_ATTACK** |
| `light_text` | text | light | 374 | 0.8886 | 0.9216 | 96.8% | 1 | **CORRECT_ATTACK** |
| `benign_2` | benign | standard | 8,883 | 0.0018 | 0.0 | 0.2% | 0 | **CORRECT_BENIGN** |
