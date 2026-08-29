"""
Dependency injection and lifecycle management for DeepDNS inference engines.
"""

import logging
from typing import Dict, Optional
from src.inference.engine import DeepDNSInferenceEngine
from src.inference.types import InferenceMode
from src.serving.config import settings

logger = logging.getLogger("DeepDNSServing")

# Global in-memory cache of initialized inference engines
_ENGINES: Dict[str, DeepDNSInferenceEngine] = {}


def initialize_engines():
    """Initializes and caches the primary canonical engines at application startup."""
    logger.info("Pre-warming DeepDNS Canonical Inference Engines...")
    
    # 1. In-Distribution Dual-View Engine (Primary)
    if "in_distribution" not in _ENGINES:
        _ENGINES["in_distribution"] = DeepDNSInferenceEngine(
            mode=InferenceMode.IN_DISTRIBUTION,
            device=settings.DEVICE,
            project_root=settings.PROJECT_ROOT,
        )
        logger.info("Loaded In-Distribution Engine (multiview_both.pt, tau=[0.95, 0.15])")

    # 2. Out-of-Distribution Dual-View Engine (LOMO)
    if "ood" not in _ENGINES:
        _ENGINES["ood"] = DeepDNSInferenceEngine(
            mode=InferenceMode.OOD,
            device=settings.DEVICE,
            project_root=settings.PROJECT_ROOT,
        )
        logger.info("Loaded OOD Engine (multiview_ood_both.pt, tau=[0.80, 0.01])")


def get_engine(mode: str = "in_distribution") -> DeepDNSInferenceEngine:
    """
    Returns the cached engine for the requested mode.
    
    Raises ValueError if the mode is unsupported.
    """
    normalized_mode = mode.lower()
    if normalized_mode not in _ENGINES:
        if normalized_mode in ["behavioral_only", "lexical_only"]:
            # Lazily load ablation engines if requested
            _ENGINES[normalized_mode] = DeepDNSInferenceEngine(
                mode=InferenceMode(normalized_mode),
                device=settings.DEVICE,
                project_root=settings.PROJECT_ROOT,
            )
        elif normalized_mode in ["in_distribution", "ood"]:
            _ENGINES[normalized_mode] = DeepDNSInferenceEngine(
                mode=InferenceMode(normalized_mode),
                device=settings.DEVICE,
                project_root=settings.PROJECT_ROOT,
            )
        else:
            raise ValueError(
                f"Unsupported inference mode '{mode}'. "
                f"Valid modes: ['in_distribution', 'ood', 'behavioral_only', 'lexical_only']"
            )
    return _ENGINES[normalized_mode]
