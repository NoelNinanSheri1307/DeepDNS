# Sequential & Temporal Structure Reconstruction

## Feasibility of Sequence Reconstruction
Can realistic DNS query sequences be reconstructed from CIC-Bell for adaptive evidence accumulation?
**Answer: YES, but exclusively from Stateless data using chronological sorting.**

## Candidate Sequence Construction Approaches

### Option A: Fixed Packet Window (W = 10, 20, 50 consecutive queries)
- **Feasibility**: High (100% supported).
- **Advantages**: Uniform tensor shapes for GRU/Transformer batching (B x T x D).
- **Disadvantages**: Ignores variable inter-arrival times unless delta t is included as an explicit feature.
- **Leakage Risk**: Low (strictly causal if ordered by timestamp).

### Option B: Time-Based Horizon Windows (Delta T = 10s, 30s, 60s)
- **Feasibility**: Moderate (Requires padding/masking due to variable query count per window).
- **Advantages**: Faithfully mirrors real SOC sensor buffers.
- **Disadvantages**: Sparse windows during low-and-slow periods.

### Option C: Adaptive Online Evidence Accumulation (The DeepDNS Novelty)
- **Feasibility**: **Optimal**.
- **Mechanics**: Sequence begins at t=1 (initial window k_0 = 5). GRU updates hidden state h_t. If entropy/uncertainty U(y_t) > tau, ingest observation t+1 and update h_{t+1} until confidence threshold is reached or maximum budget K_{max} is exhausted.
- **Compatibility**: Perfectly supported by stateless timestamp ordering.

## Selected Strategy
Use **Causal Temporal Streaming** with a sliding input buffer of size K in [5, 30], feeding relative delta t_i alongside the 11 verified stateless features.
