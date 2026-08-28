"""
Models package for DeepDNS.
"""

from src.models.rule_based import RuleBasedDetector
from src.models.random_forest import RandomForestBaseline
from src.models.mlp_baseline import MLPNetwork, MLPClassifierWrapper
from src.models.temporal_gru import TemporalGRUNetwork, TemporalGRUClassifier
from src.models.char_cnn import CharacterTokenizer, CharacterCNNEncoder, CharacterCNNClassifier
from src.models.multiview_fusion import (
    MultiViewFusionHead,
    DeepDNSMultiViewNetwork,
    DeepDNSMultiViewClassifier,
)

__all__ = [
    "RuleBasedDetector",
    "RandomForestBaseline",
    "MLPNetwork",
    "MLPClassifierWrapper",
    "TemporalGRUNetwork",
    "TemporalGRUClassifier",
    "CharacterTokenizer",
    "CharacterCNNEncoder",
    "CharacterCNNClassifier",
    "MultiViewFusionHead",
    "DeepDNSMultiViewNetwork",
    "DeepDNSMultiViewClassifier",
]
