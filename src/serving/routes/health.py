"""
Health check and engine readiness routes.
"""

from fastapi import APIRouter
from src.serving.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def get_health():
    """Returns the operational status of the DeepDNS inference service."""
    return HealthResponse(
        status="ok",
        service="deepdns",
        engine="ready",
        available_modes=["in_distribution", "ood", "behavioral_only", "lexical_only"],
    )
