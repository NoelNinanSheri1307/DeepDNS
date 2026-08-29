"""
WebSocket streaming routes for progressive real-time DNS exfiltration detection.
"""

import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, status
from src.serving.dependencies import get_engine
from src.serving.websocket_manager import stream_manager, StreamState
from src.inference.types import DecisionEnum

logger = logging.getLogger("DeepDNSStreamingRoute")
router = APIRouter(tags=["Streaming"])


@router.websocket("/ws/stream/{stream_id}")
async def websocket_stream_endpoint(
    websocket: WebSocket,
    stream_id: str,
    mode: str = Query("in_distribution"),
):
    """
    WebSocket endpoint for real-time progressive DNS exfiltration streaming.
    
    Accepts individual DNS observations incrementally and evaluates AEC confidence
    at horizons K in {5, 10, 15, 20, 25, 30}.
    """
    await websocket.accept()
    logger.info(f"WebSocket client connected to stream [{stream_id}] (Mode: {mode})")

    try:
        engine = get_engine(mode)
    except Exception as e:
        await websocket.send_json({
            "event": "ERROR",
            "stream_id": stream_id,
            "data": {"message": f"Failed to initialize engine: {str(e)}"},
        })
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        return

    # Send initial connection event
    await websocket.send_json({
        "event": "CONNECTED",
        "stream_id": stream_id,
        "data": {"mode": mode, "status": "READY"},
    })

    try:
        while True:
            raw_text = await websocket.receive_text()
            try:
                obs_data = json.loads(raw_text)
            except Exception:
                await websocket.send_json({
                    "event": "ERROR",
                    "stream_id": stream_id,
                    "data": {"message": "Invalid JSON message format."},
                })
                continue

            try:
                result, eval_status = await stream_manager.process_stream_observation(
                    stream_id=stream_id,
                    obs_dict=obs_data,
                    engine=engine,
                )
            except RuntimeError as e:
                # Stream already terminated
                await websocket.send_json({
                    "event": "ERROR",
                    "stream_id": stream_id,
                    "data": {"message": str(e)},
                })
                continue
            except Exception as e:
                logger.error(f"Error processing stream observation: {e}")
                await websocket.send_json({
                    "event": "ERROR",
                    "stream_id": stream_id,
                    "data": {"message": f"Inference processing error: {str(e)}"},
                })
                continue

            # Event Dispatching
            if eval_status in [DecisionEnum.ATTACK.value, DecisionEnum.BENIGN.value]:
                # Terminal decision reached
                await websocket.send_json({
                    "event": "DECISION",
                    "stream_id": stream_id,
                    "data": result.to_dict(),
                })
            elif result is not None and eval_status == DecisionEnum.CONTINUE.value:
                # Horizon reached but decision uncertain
                await websocket.send_json({
                    "event": "EVIDENCE_UPDATE",
                    "stream_id": stream_id,
                    "data": {
                        "horizon": result.stopping_horizon,
                        "attack_probability": result.attack_probability,
                        "benign_probability": result.benign_probability,
                        "status": "CONTINUE",
                        "evidence_history": {str(k): v for k, v in result.evidence_history.items()},
                    },
                })
            else:
                # Observation buffered between evaluation horizons
                session = await stream_manager.get_session(stream_id)
                count = len(session.stream) if session else 0
                await websocket.send_json({
                    "event": "OBSERVATION_BUFFERED",
                    "stream_id": stream_id,
                    "data": {"observations_count": count},
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from stream [{stream_id}]")
        await stream_manager.remove_session(stream_id)
    except Exception as e:
        logger.error(f"Unexpected WebSocket error on stream [{stream_id}]: {e}")
        await stream_manager.remove_session(stream_id)
