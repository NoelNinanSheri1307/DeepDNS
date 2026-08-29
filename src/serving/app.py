"""
FastAPI Application Entry Point for DeepDNS Serving Layer.
"""

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.serving.config import settings
from src.serving.dependencies import initialize_engines
from src.serving.routes import health_router, detection_router, streaming_router

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger("DeepDNSService")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for application startup and shutdown."""
    logger.info("Initializing DeepDNS Production Serving Layer...")
    initialize_engines()
    logger.info("DeepDNS Inference Engines Ready. Service is operational.")
    yield
    logger.info("Shutting down DeepDNS Serving Layer.")


def create_app() -> FastAPI:
    """Factory function to build and configure the FastAPI application."""
    app = FastAPI(
        title="DeepDNS Canonical Serving API",
        version="1.0.0",
        description=(
            "Enterprise Real-Time DNS Tunneling Detection & Exfiltration Mitigation "
            "powered by Multi-View Neural Fusion and Adaptive Evidence Control (AEC)."
        ),
        lifespan=lifespan,
    )

    # Configure CORS for development and production frontend integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include Routes
    app.include_router(health_router)
    app.include_router(detection_router)
    app.include_router(streaming_router)

    return app


app = create_app()
