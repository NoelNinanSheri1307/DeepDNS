"""
Configuration settings for the DeepDNS Serving Layer.
"""

import os
from pathlib import Path
from typing import List


class Settings:
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
    HOST: str = os.getenv("DEEPDNS_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("DEEPDNS_PORT", "8000"))
    DEVICE: str = os.getenv("DEEPDNS_DEVICE", "cuda" if os.getenv("CUDA_VISIBLE_DEVICES") else "cpu")
    LOG_LEVEL: str = os.getenv("DEEPDNS_LOG_LEVEL", "INFO")
    
    # CORS Configuration
    CORS_ORIGINS: List[str] = os.getenv(
        "DEEPDNS_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
    ).split(",")


settings = Settings()
