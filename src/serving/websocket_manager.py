"""
WebSocket Session Manager and Stream Isolation Controller for DeepDNS.
"""

import asyncio
from enum import Enum
import logging
from typing import Any, Dict, Optional, Tuple
from src.inference.types import DNSObservation, DNSStream, DetectionResult, DecisionEnum
from src.inference.engine import DeepDNSInferenceEngine

logger = logging.getLogger("DeepDNSStreamManager")


class StreamState(str, Enum):
    CREATED = "CREATED"
    RECEIVING = "RECEIVING"
    EVALUATING = "EVALUATING"
    DECISION_REACHED = "DECISION_REACHED"


class StreamSession:
    """Represents an isolated streaming session with thread/async safe concurrency control."""
    def __init__(self, stream_id: str, mode: str = "in_distribution"):
        self.stream_id = stream_id
        self.mode = mode
        self.stream = DNSStream(stream_id=stream_id)
        self.state = StreamState.CREATED
        self.last_result: Optional[DetectionResult] = None
        self.lock = asyncio.Lock()


class StreamManager:
    """
    Manages active streaming sessions with strict memory isolation.
    """
    def __init__(self):
        self._sessions: Dict[str, StreamSession] = {}
        self._global_lock = asyncio.Lock()

    async def get_or_create_session(self, stream_id: str, mode: str = "in_distribution") -> StreamSession:
        async with self._global_lock:
            if stream_id not in self._sessions:
                self._sessions[stream_id] = StreamSession(stream_id=stream_id, mode=mode)
                logger.info(f"StreamSession [{stream_id}] created (Mode: {mode}).")
            return self._sessions[stream_id]

    async def get_session(self, stream_id: str) -> Optional[StreamSession]:
        async with self._global_lock:
            return self._sessions.get(stream_id)

    async def remove_session(self, stream_id: str):
        async with self._global_lock:
            if stream_id in self._sessions:
                del self._sessions[stream_id]
                logger.info(f"StreamSession [{stream_id}] cleaned up.")

    async def process_stream_observation(
        self,
        stream_id: str,
        obs_dict: Dict[str, Any],
        engine: DeepDNSInferenceEngine,
    ) -> Tuple[Optional[DetectionResult], str]:
        """
        Thread-safely appends observation to the isolated stream and executes AEC evaluation.
        
        Returns:
            Tuple of (DetectionResult if stopped/forced else None, status_string).
        """
        session = await self.get_or_create_session(stream_id, mode=engine.mode.value)
        
        async with session.lock:
            if session.state == StreamState.DECISION_REACHED:
                raise RuntimeError(
                    f"Stream [{stream_id}] has already terminated with decision '{session.last_result.decision if session.last_result else 'UNKNOWN'}'. "
                    f"Further observations are rejected."
                )

            session.state = StreamState.RECEIVING
            obs = DNSObservation.from_dict(obs_dict)
            session.stream.add_observation(obs)
            curr_len = len(session.stream)

            # Check if current observation count is an allowed AEC evaluation horizon
            if curr_len in engine.allowed_horizons:
                session.state = StreamState.EVALUATING
                result = engine.detect(session.stream, stream_id=stream_id)

                if result.is_early_decision or curr_len >= engine.max_horizon:
                    session.state = StreamState.DECISION_REACHED
                    session.last_result = result
                    logger.info(f"Stream [{stream_id}] reached terminal decision: {result.decision} at K={result.stopping_horizon}")
                    return result, result.decision

                # Intermediate evidence update
                return result, DecisionEnum.CONTINUE.value

            return None, DecisionEnum.CONTINUE.value


stream_manager = StreamManager()
