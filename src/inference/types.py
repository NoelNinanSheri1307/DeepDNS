"""
Canonical Data Types and Input Contracts for the DeepDNS Inference Engine.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd


class DecisionEnum(str, Enum):
    ATTACK = "ATTACK"
    BENIGN = "BENIGN"
    CONTINUE = "CONTINUE"


class InferenceMode(str, Enum):
    IN_DISTRIBUTION = "in_distribution"
    OOD = "ood"
    BEHAVIORAL_ONLY = "behavioral_only"
    LEXICAL_ONLY = "lexical_only"


@dataclass
class DNSObservation:
    """
    Canonical representation of a single DNS query observation in a stream.
    
    Can be constructed from raw network fields, a dictionary, or a DataFrame row.
    """
    timestamp: Union[str, pd.Timestamp, float]
    domain_name: str
    fqdn_count: float = 0.0
    subdomain_length: float = 0.0
    upper: float = 0.0
    lower: float = 0.0
    numeric: float = 0.0
    entropy: float = 0.0
    special: float = 0.0
    labels: float = 0.0
    labels_max: float = 0.0
    labels_average: float = 0.0
    len: float = 0.0
    subdomain: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DNSObservation":
        """Constructs a DNSObservation from a dictionary, mapping schema field variations."""
        domain = data.get("domain_name") or data.get("domain") or data.get("sld") or data.get("query") or ""
        return cls(
            timestamp=data.get("timestamp", 0.0),
            domain_name=str(domain),
            fqdn_count=float(data.get("FQDN_count", data.get("fqdn_count", len(str(domain))))),
            subdomain_length=float(data.get("subdomain_length", 0.0)),
            upper=float(data.get("upper", 0.0)),
            lower=float(data.get("lower", 0.0)),
            numeric=float(data.get("numeric", 0.0)),
            entropy=float(data.get("entropy", 0.0)),
            special=float(data.get("special", 0.0)),
            labels=float(data.get("labels", 0.0)),
            labels_max=float(data.get("labels_max", 0.0)),
            labels_average=float(data.get("labels_average", 0.0)),
            len=float(data.get("len", len(str(domain)))),
            subdomain=float(data.get("subdomain", 0.0)),
            metadata={k: v for k, v in data.items() if k not in [
                "timestamp", "domain_name", "FQDN_count", "subdomain_length", "upper", "lower",
                "numeric", "entropy", "special", "labels", "labels_max", "labels_average", "len", "subdomain"
            ]},
        )


@dataclass
class DNSStream:
    """
    A sequence of ordered DNS observations belonging to a monitored host / client session.
    """
    stream_id: str
    observations: List[DNSObservation] = field(default_factory=list)

    def add_observation(self, obs: Union[DNSObservation, Dict[str, Any]]):
        if isinstance(obs, dict):
            obs = DNSObservation.from_dict(obs)
        self.observations.append(obs)

    def __len__(self) -> int:
        return len(self.observations)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the internal observation list into a standard DataFrame."""
        records = []
        for o in self.observations:
            records.append({
                "timestamp": o.timestamp,
                "FQDN_count": o.fqdn_count,
                "subdomain_length": o.subdomain_length,
                "upper": o.upper,
                "lower": o.lower,
                "numeric": o.numeric,
                "entropy": o.entropy,
                "special": o.special,
                "labels": o.labels,
                "labels_max": o.labels_max,
                "labels_average": o.labels_average,
                "len": o.len,
                "subdomain": o.subdomain,
                "domain_name": o.domain_name,
            })
        return pd.DataFrame(records)


@dataclass
class DetectionResult:
    """
    Structured canonical detection output emitted by DeepDNS.
    """
    decision: str  # "ATTACK" or "BENIGN"
    confidence: float
    attack_probability: float
    benign_probability: float
    stopping_horizon: int
    max_horizon: int
    observations_consumed: int
    observations_saved: int
    query_savings_pct: float
    is_early_decision: bool
    stop_reason: str
    evidence_history: Dict[int, float]
    behavioral_evidence: Optional[Dict[str, float]] = None
    lexical_evidence: Optional[Dict[str, float]] = None
    model_name: str = "multiview_both"
    mode: str = "in_distribution"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "confidence": round(self.confidence, 4),
            "attack_probability": round(self.attack_probability, 4),
            "benign_probability": round(self.benign_probability, 4),
            "stopping_horizon": self.stopping_horizon,
            "max_horizon": self.max_horizon,
            "observations_consumed": self.observations_consumed,
            "observations_saved": self.observations_saved,
            "query_savings_pct": round(self.query_savings_pct, 2),
            "is_early_decision": self.is_early_decision,
            "stop_reason": self.stop_reason,
            "evidence_history": {k: round(v, 4) for k, v in self.evidence_history.items()},
            "model_name": self.model_name,
            "mode": self.mode,
        }
