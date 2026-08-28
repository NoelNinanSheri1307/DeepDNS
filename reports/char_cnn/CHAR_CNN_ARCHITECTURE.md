# DeepDNS Character-Level Lexical Encoder (Character-CNN) Architecture

## 1. Executive Summary & Design Rationale
The **Character-CNN** provides DeepDNS with an orthogonal lexical view ($z_{lex} \in \mathbb{R}^{128}$) of DNS traffic by inspecting raw character-level orthography in domain names. While the Temporal GRU models inter-arrival intervals and sequential feature dynamics, the Character-CNN identifies:
- Base32/Base64 chunking anomalies in subdomains.
- High Shannon character entropy and unusual consonant-to-vowel distributions.
- Hexadecimal encoding fragments from tunneling tools (e.g. `dnscat2`, `iodine`).

By combining lexical domain structure with behavioral temporal modeling in the Multi-View Fusion stage, DeepDNS eliminates reliance on synthetic testbed timing cadences.

---

## 2. Character Vocabulary & Tokenization Specification

The tokenizer ([`CharacterTokenizer`](file:///c:/Users/VICTUS/deepdns/src/models/char_cnn.py)) enforces a deterministic, immutable ASCII vocabulary constructed strictly without target labels or test-set statistics:

- **Maximum Domain Length ($L$)**: 128 characters (RFC 1035 specifies max 253 octets; 128 captures $>99.9\%$ of query labels).
- **Vocabulary Size ($V$)**: 45 tokens.
  - Index `0`: `<pad>` (zero-padding token)
  - Index `1`: `<unk>` (out-of-vocabulary / special non-standard characters)
  - Indices `2..44`: Printable ASCII characters:
    - Lowercase alphabet: `a-z` (26 tokens)
    - Numeric digits: `0-9` (10 tokens)
    - Punctuation & delimiters: `.-_~+/=` (7 tokens)

---

## 3. Neural Network Architecture Specification

```
                  Domain String (e.g. "x4f1a.tunnel.attacker.com")
                                       │
                                       ▼
                       CharacterTokenizer (Length L = 128)
                                       │
                                       ▼
                       Token Tensor: (Batch_Size, 128)
                                       │
                                       ▼
                       nn.Embedding(45, 32, padding_idx=0)
                                       │
                                       ▼
                       Embedded Tensor: (Batch_Size, 32, 128)
                                       │
                ┌──────────────────────┼──────────────────────┐
                │                      │                      │
                ▼                      ▼                      ▼
        Conv1D (k=3, pad=1)    Conv1D (k=5, pad=2)    Conv1D (k=7, pad=3)
        32 Filters             32 Filters             32 Filters
                │                      │                      │
                ▼                      ▼                      ▼
             ReLU()                 ReLU()                 ReLU()
                │                      │                      │
                ▼                      ▼                      ▼
        Global Max Pool        Global Max Pool        Global Max Pool
                │                      │                      │
                └──────────────────────┼──────────────────────┘
                                       │
                                       ▼
                     Concatenated Filter Maps: (Batch_Size, 96)
                                       │
                                       ▼
                     Linear(96, 128) ──► LayerNorm(128) ──► ReLU ──► Dropout(0.1)
                                       │
                                       ▼
                     Lexical Representation z_lex in R^128
```

---

## 4. Parameter Count & Tensor Dimensions

| Layer | Input Shape | Output Shape | Parameters | Computation |
| :--- | :--- | :--- | :--- | :--- |
| `embedding` | `(B, 128)` | `(B, 32, 128)` | $45 \times 32 = 1,440$ | Lookup Table |
| `conv_branch_1` ($k=3$) | `(B, 32, 128)` | `(B, 32, 128)` | $(32 \times 3 \times 32) + 32 = 3,104$ | Short n-gram morphemes |
| `conv_branch_2` ($k=5$) | `(B, 32, 128)` | `(B, 32, 128)` | $(32 \times 5 \times 32) + 32 = 5,152$ | Medium sub-label chunks |
| `conv_branch_3` ($k=7$) | `(B, 32, 128)` | `(B, 32, 128)` | $(32 \times 7 \times 32) + 32 = 7,200$ | Long payload blocks |
| `global_max_pool` | $3 \times (B, 32, 128)$ | `(B, 96)` | $0$ | Shift-invariant feature pooling |
| `projection` | `(B, 96)` | `(B, 128)` | $(96 \times 128) + 128 + 256 = 12,672$ | Dense 128D projection + LayerNorm |
| `classifier` (Ablation) | `(B, 128)` | `(B, 2)` | $(128 \times 2) + 2 = 258$ | Standalone classification head |
| **Total Parameters** | — | — | **29,826 Parameters** ($\approx 119.3\text{ KB}$) | Highly lightweight for RTX 2050 |

---

## 5. Sequential Multi-Observation Handling

When evaluated across an observation sequence of length $K \in [5, 30]$:
1. Input tensor has shape `(B, K, 128)` containing token IDs for $K$ consecutive queries.
2. The encoder flattens the sequence into $(B \cdot K, 128)$, extracts lexical embeddings $(B \cdot K, 128)$, and reshapes back to $(B, K, 128)$.
3. This produces a sequence of lexical representations $z_{lex}^{(1)}, z_{lex}^{(2)}, \dots, z_{lex}^{(K)}$ aligned synchronously with the Temporal GRU hidden states.
