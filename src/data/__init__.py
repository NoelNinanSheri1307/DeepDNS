"""
DeepDNS Data Engineering Package.

Provides robust, causal, leakage-safe data ingestion, schema validation,
feature extraction, sequence windowing, and grouped dataset splitting.
"""

from src.data.schema import (
    CIC_BELL_STATELESS_COLUMNS,
    FORBIDDEN_MODEL_COLUMNS,
    CAUSAL_FEATURE_NAMES,
    DNS_THREATS_COLUMNS,
    ValidationResult,
    validate_stateless_dataframe,
)

from src.data.labels import (
    CaptureMetadata,
    parse_cic_bell_label,
    VALID_MODALITIES,
    VALID_INTENSITIES,
)

from src.data.features import (
    CausalFeatureExtractor,
    FeatureScaler,
)

from src.data.sequences import (
    SequenceWindow,
    SequenceBuilder,
    StreamingSequenceDataset,
    DEFAULT_MIN_SEQ_LEN,
    DEFAULT_MAX_SEQ_LEN,
)

from src.data.splits import (
    SplitManifest,
    GroupedSplitter,
)

from src.data.lexical import (
    DNSThreatsLexicalTokenizer,
    DNSThreatsDataset,
    DEFAULT_MAX_DOMAIN_LEN,
)

__all__ = [
    "CIC_BELL_STATELESS_COLUMNS",
    "FORBIDDEN_MODEL_COLUMNS",
    "CAUSAL_FEATURE_NAMES",
    "DNS_THREATS_COLUMNS",
    "ValidationResult",
    "validate_stateless_dataframe",
    "CaptureMetadata",
    "parse_cic_bell_label",
    "VALID_MODALITIES",
    "VALID_INTENSITIES",
    "CausalFeatureExtractor",
    "FeatureScaler",
    "SequenceWindow",
    "SequenceBuilder",
    "StreamingSequenceDataset",
    "DEFAULT_MIN_SEQ_LEN",
    "DEFAULT_MAX_SEQ_LEN",
    "SplitManifest",
    "GroupedSplitter",
    "DNSThreatsLexicalTokenizer",
    "DNSThreatsDataset",
    "DEFAULT_MAX_DOMAIN_LEN",
]
