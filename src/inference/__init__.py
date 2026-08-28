"""
DeepDNS Canonical Inference and Sequential Evidence Control Package.
"""

from src.inference.types import (
    DNSObservation,
    DNSStream,
    DetectionResult,
    DecisionEnum,
    InferenceMode,
)
from src.inference.engine import DeepDNSInferenceEngine
from src.inference.adaptive_controller import (
    AdaptiveEvidenceController,
    AdaptiveDecision,
    AdaptiveEvaluationResult,
)
from src.inference.cusum import SequentialCUSUMDetector

__all__ = [
    "DeepDNSInferenceEngine",
    "DNSObservation",
    "DNSStream",
    "DetectionResult",
    "DecisionEnum",
    "InferenceMode",
    "AdaptiveEvidenceController",
    "AdaptiveDecision",
    "AdaptiveEvaluationResult",
    "SequentialCUSUMDetector",
]
