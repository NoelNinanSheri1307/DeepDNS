# Computational Feasibility (NVIDIA RTX 2050 4GB VRAM)

## Resource Consumption Estimates

| Component | Memory / VRAM Footprint | CPU RAM Footprint | Compute Time (Estimate) | Feasibility on RTX 2050 |
| :--- | :--- | :--- | :--- | :--- |
| **Data Ingestion (Chunked)** | 0 MB (CPU) | ~1.2 GB RAM | ~15 sec | **100% FEASIBLE** |
| **DNS Threats Char-CNN** | ~450 MB VRAM (Batch 256) | ~2.0 GB RAM | ~4 min / epoch | **100% FEASIBLE** |
| **CIC-Bell GRU Encoder** | ~280 MB VRAM (Batch 128) | ~1.5 GB RAM | ~45 sec / epoch | **100% FEASIBLE** |
| **3-Model Deep Ensemble** | ~850 MB VRAM Total | ~2.5 GB RAM | ~2.5 min / epoch | **100% FEASIBLE** |
| **Inference Latency** | < 10 MB VRAM | Minimal | **< 3.2 ms per window** | **Real-Time SOC Capable** |

### Execution Strategy
- Use PyTorch with AMP (`torch.cuda.amp.autocast()`).
- Keep maximum sequence length K_{max} <= 30.
- Batch size = 128 or 256 for smooth 4GB VRAM utilization.
