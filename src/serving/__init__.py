"""
DeepDNS Production Serving Package.
"""

from src.serving.app import app, create_app

__all__ = ["app", "create_app"]
