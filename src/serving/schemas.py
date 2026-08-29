"""
Pydantic Schemas and Request/Response Models for DeepDNS Serving Layer.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class DNSObservationSchema(BaseModel):
    """Schema representing a single incoming DNS query observation."""
    timestamp: Union[str, float, int] = Field(
        default=0.0,
        description="Observation timestamp (ISO string, unix epoch float, or integer)",
        json_schema_extra={"example": "2026-08-28T12:00:00.123456"},
    )
    domain_name: str = Field(
        ...,
        description="Queried Fully Qualified Domain Name (FQDN) or SLD",
        json_schema_extra={"example": "a9f3b2.exfiltration.tunnel.example.com"},
    )
    fqdn_count: Optional[float] = Field(default=None, description="Length or character count of FQDN")
    subdomain_length: Optional[float] = Field(default=None, description="Length of the subdomain label")
    upper: Optional[float] = Field(default=None, description="Count of uppercase characters")
    lower: Optional[float] = Field(default=None, description="Count of lowercase characters")
    numeric: Optional[float] = Field(default=None, description="Count of numeric digits")
    entropy: Optional[float] = Field(default=None, description="Shannon character entropy of domain")
    special: Optional[float] = Field(default=None, description="Count of special non-alphanumeric characters")
    labels: Optional[float] = Field(default=None, description="Total count of dot-separated domain labels")
    labels_max: Optional[float] = Field(default=None, description="Maximum label length")
    labels_average: Optional[float] = Field(default=None, description="Average label length")
    len: Optional[float] = Field(default=None, description="Total string length")
    subdomain: Optional[float] = Field(default=None, description="Binary indicator or subdomain length")


class DetectRequest(BaseModel):
    """Schema for batch/sequence detection request."""
    mode: str = Field(
        default="in_distribution",
        description="Inference mode: 'in_distribution' (standard dual-view) or 'ood' (LOMO model)",
        json_schema_extra={"example": "in_distribution"},
    )
    stream_id: Optional[str] = Field(
        default="api_stream_001",
        description="Identifier for the monitored host or client session",
        json_schema_extra={"example": "host-192-168-1-105"},
    )
    observations: List[DNSObservationSchema] = Field(
        ...,
        description="Ordered sequence of DNS query observations (5 <= K <= 30)",
        min_length=1,
    )


class DetectResponse(BaseModel):
    """Schema for structured DeepDNS detection response."""
    decision: str = Field(..., description="Final detection verdict: 'ATTACK' or 'BENIGN'", json_schema_extra={"example": "ATTACK"})
    confidence: float = Field(..., description="Decision confidence score in [0.0, 1.0]", json_schema_extra={"example": 0.9850})
    attack_probability: float = Field(..., description="Probability of DNS tunneling attack", json_schema_extra={"example": 0.9850})
    benign_probability: float = Field(..., description="Probability of benign network flow", json_schema_extra={"example": 0.0150})
    stopping_horizon: int = Field(..., description="Horizon K at which evaluation terminated", json_schema_extra={"example": 15})
    max_horizon: int = Field(default=30, description="Maximum bounded observation window", json_schema_extra={"example": 30})
    observations_consumed: int = Field(..., description="Number of queries observed before decision", json_schema_extra={"example": 15})
    observations_saved: int = Field(..., description="Number of queries saved relative to K=30", json_schema_extra={"example": 15})
    query_savings_pct: float = Field(..., description="Query volume savings percentage", json_schema_extra={"example": 50.0})
    is_early_decision: bool = Field(..., description="True if decision occurred before horizon K=30", json_schema_extra={"example": True})
    stop_reason: str = Field(..., description="Explanatory AEC reason string", json_schema_extra={"example": "ATTACK_THRESHOLD_MET (p=0.9850 >= 0.95)"})
    evidence_history: Dict[str, float] = Field(
        ...,
        description="Sequential P(Attack) values at evaluated horizons",
        json_schema_extra={"example": {"5": 0.21, "10": 0.67, "15": 0.985}},
    )
    model_name: str = Field(..., description="Trained model checkpoint name", json_schema_extra={"example": "multiview_both"})
    mode: str = Field(..., description="Evaluated inference mode", json_schema_extra={"example": "in_distribution"})


class HealthResponse(BaseModel):
    """Schema for service health check response."""
    status: str = Field(default="ok", json_schema_extra={"example": "ok"})
    service: str = Field(default="deepdns", json_schema_extra={"example": "deepdns"})
    engine: str = Field(default="ready", json_schema_extra={"example": "ready"})
    available_modes: List[str] = Field(
        default=["in_distribution", "ood", "behavioral_only", "lexical_only"],
        json_schema_extra={"example": ["in_distribution", "ood", "behavioral_only", "lexical_only"]},
    )


class StreamingEvent(BaseModel):
    """Schema for WebSocket streaming events."""
    event: str = Field(..., description="Event type: 'EVIDENCE_UPDATE', 'DECISION', 'ERROR', 'CONNECTED'", json_schema_extra={"example": "EVIDENCE_UPDATE"})
    stream_id: str = Field(..., description="Identifier of the active stream", json_schema_extra={"example": "stream-123"})
    data: Dict[str, Any] = Field(default_factory=dict, description="Payload attributes for the event")
