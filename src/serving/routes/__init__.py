"""
Routes package for DeepDNS Serving Layer.
"""

from src.serving.routes.health import router as health_router
from src.serving.routes.detection import router as detection_router
from src.serving.routes.streaming import router as streaming_router

__all__ = ["health_router", "detection_router", "streaming_router"]
