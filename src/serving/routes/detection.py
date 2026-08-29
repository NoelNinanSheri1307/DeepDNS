"""
Batch and sequence detection routes for DeepDNS.
"""

import logging
from fastapi import APIRouter, HTTPException, status
from src.serving.schemas import DetectRequest, DetectResponse
from src.serving.dependencies import get_engine
from src.inference.types import DNSStream, DNSObservation

logger = logging.getLogger("DeepDNSDetectionRoute")
router = APIRouter(tags=["Detection"])


@router.post("/detect", response_model=DetectResponse, status_code=status.HTTP_200_OK)
async def detect_sequence(request: DetectRequest):
    """
    Executes sequential Multi-View inference with Adaptive Evidence Early Stopping on an ordered DNS sequence.
    
    Minimum observations: K_min = 5
    Maximum observations: K_max = 30
    """
    obs_count = len(request.observations)
    if obs_count < 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"DeepDNS requires at least 5 observations for sequential evaluation (received {obs_count}).",
        )

    # Resolve cached canonical inference engine
    try:
        engine = get_engine(request.mode)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        logger.error(f"Failed to load engine for mode '{request.mode}': {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is currently unavailable.",
        )

    # Build canonical DNSStream
    stream = DNSStream(stream_id=request.stream_id or "api_stream")
    for obs_schema in request.observations:
        obs_dict = obs_schema.model_dump() if hasattr(obs_schema, "model_dump") else obs_schema.dict()
        obs = DNSObservation.from_dict(obs_dict)
        stream.add_observation(obs)

    # Run Canonical Inference
    try:
        result = engine.detect(stream, stream_id=stream.stream_id)
    except Exception as e:
        logger.error(f"Inference execution failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference processing failed.",
        )

    # Serialize to schema
    return DetectResponse(
        decision=result.decision,
        confidence=result.confidence,
        attack_probability=result.attack_probability,
        benign_probability=result.benign_probability,
        stopping_horizon=result.stopping_horizon,
        max_horizon=result.max_horizon,
        observations_consumed=result.observations_consumed,
        observations_saved=result.observations_saved,
        query_savings_pct=result.query_savings_pct,
        is_early_decision=result.is_early_decision,
        stop_reason=result.stop_reason,
        evidence_history={str(k): v for k, v in result.evidence_history.items()},
        model_name=result.model_name,
        mode=result.mode,
    )
