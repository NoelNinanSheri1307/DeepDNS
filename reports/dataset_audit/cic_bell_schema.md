# CIC-Bell DNS Exfiltration 2021 Schema Forensic Analysis

## Executive Summary
The **CIC-Bell-DNS-EXF-2021** dataset consists of 36 CSV files (18 stateless and 18 stateful) capturing DNS traffic during data exfiltration attacks and benign network activity. Our audit reveals distinct schemas for stateless and stateful feature sets, significant data type anomalies, and complete absence of raw domain name strings.

## Stateless Features Schema (15 Columns)
Every stateless CSV file contains exactly 15 columns representing individual DNS queries/responses with microsecond-resolution timestamps and pre-extracted lexical/structural metrics.

| Column Name | Data Type | Null % | Unique Values (Total) | Description & Forensic Notes |
| :--- | :--- | :--- | :--- | :--- |
| `timestamp` | `object` (ISO8601 string) | 0.0% | 696,554 | Packet capture timestamp (YYYY-MM-DD HH:MM:SS.ffffff). Strictly monotonic per capture. |
| `FQDN_count` | `int64` | 0.0% | 125 | Total character length of the Fully Qualified Domain Name. |
| `subdomain_length` | `int64` | 0.0% | 114 | Total length of subdomain labels (excluding SLD/TLD). |
| `upper` | `int64` | 0.0% | 22 | Count of uppercase characters in FQDN. |
| `lower` | `int64` | 0.0% | 108 | Count of lowercase characters in FQDN. |
| `numeric` | `int64` | 0.0% | 68 | Count of numerical digits (0-9) in FQDN. |
| `entropy` | `float64` | 0.0% | 1,482 | Shannon entropy of characters in the domain string. |
| `special` | `int64` | 0.0% | 14 | Count of special characters (e.g. dots, hyphens, underscores). |
| `labels` | `int64` | 0.0% | 15 | Number of dot-separated labels in the domain. |
| `labels_max` | `int64` | 0.0% | 72 | Maximum length among individual labels. |
| `labels_average` | `float64` | 0.0% | 450 | Average length across labels. |
| `longest_word` | `int64`/`object` | 0.0% | 52 | Length of longest dictionary word detected in the query. |
| `sld` | `object` / `int64` | 0.0% | 1,241 | Encoded representation / length of the Second-Level Domain. |
| `len` | `int64` | 0.0% | 118 | Length of the query payload/record. |
| `subdomain` | `int64` (binary) | 0.0% | 2 | Binary indicator (1 if subdomain present, 0 otherwise). |

> [!WARNING]
> **Absence of Raw String Queries**: Raw FQDN strings (e.g., `exfil-01a9b.attacker.com`) were **discarded** during the creation of CIC-Bell. Only statistical character counts and entropy are preserved. A Character-level CNN cannot be applied directly to CIC-Bell unless synthetic domain reconstruction or raw DNS Threats data is used.

## Stateful Features Schema (27 Columns)
Stateful CSV files contain 27 columns representing aggregated traffic over time windows.

| Column Name | Raw Dtype | Anomalous Structures | Forensic Assessment |
| :--- | :--- | :--- | :--- |
| `rr` | `float64` | None | Resource Record count. |
| `A_frequency` to `OPT_frequency` (12 cols) | `int64` | None | Query type frequency counts (A, NS, CNAME, SOA, NULL, PTR, MX, TXT, AAAA, SRV, OPT, HINFO). |
| `rr_type` | `object` | Raw Python `set` strings (e.g., `"{'PTR'}"`, `"{'A', 'TXT'}"`) | Non-standard string serialization; requires AST parsing or one-hot decoding. |
| `rr_count` | `int64` | None | Total RR count in window. |
| `rr_name_entropy` | `float64` | None | Entropy across RR names. |
| `rr_name_length` | `int64` | None | Cumulative/mean RR name length. |
| `distinct_ns` | `int64` | None | Number of distinct Name Servers. |
| `distinct_ip` | `object` | Raw Python `set` literals (e.g., `"set()"`, `"{'192.168.1.1'}"`) | Serialization artifact. |
| `unique_country` | `object` | Raw Python `set` literals (e.g., `"set()"`) | Empty or unresolvable in lab environment. |
| `unique_asn` | `object` | Raw Python `set` literals (e.g., `"set()"`) | Empty or unresolvable in lab environment. |
| `distinct_domains` | `object` | Raw Python `dict`/`set` string literals (`"{}"`) | High-risk string artifact. |
| `reverse_dns` | `object` | Categorical strings (`"unknown"`, `"resolved"`) | High percentage constant. |
| `a_records` | `int64` | None | Count of resolved A records. |
| `unique_ttl` | `object` | Raw Python `list` literals (e.g., `"[1, 1]"`, `"[300]"`) | Requires array parsing. |
| `ttl_mean` | `float64` | None | Mean TTL across window. |
| `ttl_variance` | `float64` | None | Variance of TTL across window. |

## Key Findings
1. **Missing Flow Identifiers**: Neither stateless nor stateful files contain Client IP, Server IP, Client Port, or Transaction ID (XID). Traffic is grouped solely by the PCAP file capture session.
2. **Data Type Corruption**: Stateful files store complex data structures as raw Python string reprs rather than numeric vectors.
3. **Completeness**: 0% null values across all numerical columns; no `inf` values present.
