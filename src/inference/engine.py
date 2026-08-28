"""
Canonical Inference Engine for DeepDNS.

Wraps the trained frozen Multi-View architecture, causal preprocessing,
and Adaptive Evidence Controller into an enterprise-ready callable API.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch

from src.data.features import CausalFeatureExtractor, FeatureScaler
from src.models.char_cnn import CharacterTokenizer
from src.models.multiview_fusion import DeepDNSMultiViewClassifier
from src.inference.types import (
    DNSObservation,
    DNSStream,
    DetectionResult,
    DecisionEnum,
    InferenceMode,
)
from src.inference.adaptive_controller import AdaptiveEvidenceController, AdaptiveDecision


class DeepDNSInferenceEngine:
    """
    Canonical DeepDNS Detector.
    
    Provides high-level, streaming-ready detection for DNS exfiltration streams.
    """

    def __init__(
        self,
        mode: Union[str, InferenceMode] = InferenceMode.IN_DISTRIBUTION,
        device: Optional[str] = None,
        project_root: Optional[Path] = None,
        tau_attack: Optional[float] = None,
        tau_benign: Optional[float] = None,
    ):
        self.project_root = project_root or Path(__file__).resolve().parent.parent.parent
        self.mode = InferenceMode(mode) if isinstance(mode, str) else mode
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Resolve Frozen Checkpoints & Calibrated Thresholds
        if self.mode == InferenceMode.IN_DISTRIBUTION:
            self.model_path = self.project_root / "data" / "processed" / "models" / "multiview_both.pt"
            self.scaler_path = self.project_root / "data" / "processed" / "feature_scaler.json"
            self.default_tau_atk = 0.95
            self.default_tau_ben = 0.15
            self.fusion_mode = "both"
        elif self.mode == InferenceMode.OOD:
            self.model_path = self.project_root / "data" / "processed" / "models" / "multiview_ood_both.pt"
            self.scaler_path = self.project_root / "data" / "processed" / "feature_scaler_lomo.json"
            self.default_tau_atk = 0.80
            self.default_tau_ben = 0.01
            self.fusion_mode = "both"
        elif self.mode == InferenceMode.BEHAVIORAL_ONLY:
            self.model_path = self.project_root / "data" / "processed" / "models" / "multiview_behavioral_only.pt"
            self.scaler_path = self.project_root / "data" / "processed" / "feature_scaler.json"
            self.default_tau_atk = 0.95
            self.default_tau_ben = 0.15
            self.fusion_mode = "behavioral_only"
        elif self.mode == InferenceMode.LEXICAL_ONLY:
            self.model_path = self.project_root / "data" / "processed" / "models" / "multiview_lexical_only.pt"
            self.scaler_path = self.project_root / "data" / "processed" / "feature_scaler.json"
            self.default_tau_atk = 0.95
            self.default_tau_ben = 0.15
            self.fusion_mode = "lexical_only"
        else:
            raise ValueError(f"Unsupported inference mode: {self.mode}")

        self.tau_attack = tau_attack if tau_attack is not None else self.default_tau_atk
        self.tau_benign = tau_benign if tau_benign is not None else self.default_tau_ben

        # 2. Initialize Preprocessors & Tokenizer
        self.extractor = CausalFeatureExtractor()
        self.scaler = FeatureScaler.from_json(self.scaler_path)
        self.tokenizer = CharacterTokenizer()

        # 3. Load Trained Model Checkpoint
        if not self.model_path.exists():
            raise FileNotFoundError(f"DeepDNS model checkpoint not found at: {self.model_path}")
        self.classifier = DeepDNSMultiViewClassifier.load(self.model_path, device=self.device)
        self.classifier.model.eval()

        # 4. Initialize Adaptive Evidence Controller
        self.allowed_horizons = [5, 10, 15, 20, 25, 30]
        self.min_horizon = 5
        self.max_horizon = 30
        self.aec = AdaptiveEvidenceController(
            tau_attack=self.tau_attack,
            tau_benign=self.tau_benign,
            horizons=self.allowed_horizons,
            min_horizon=self.min_horizon,
            max_horizon=self.max_horizon,
        )

    def preprocess_stream(self, stream: DNSStream) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Transforms a raw DNSStream into model-ready Behavioral and Lexical tensors.
        
        Returns:
            beh_tensor: (1, K_actual, 12)
            lex_tensor: (1, K_actual, 128)
            seq_len: int
        """
        df_raw = stream.to_dataframe()
        if len(df_raw) == 0:
            raise ValueError("Cannot preprocess an empty DNSStream.")

        # Limit to max horizon 30
        df_eval = df_raw.iloc[:self.max_horizon].copy()
        seq_len = len(df_eval)

        # 1. Behavioral Features
        df_feat = self.extractor.extract_from_dataframe(df_eval)
        feat_norm = self.scaler.transform(df_feat)
        beh_tensor = torch.tensor(feat_norm, dtype=torch.float32).unsqueeze(0)  # (1, K, 12)

        # 2. Lexical Character Tokens
        domain_series = df_eval["domain_name"].fillna("").astype(str)
        token_arrays = [self.tokenizer.encode(d) for d in domain_series]
        lex_tensor = torch.tensor(np.stack(token_arrays), dtype=torch.long).unsqueeze(0)  # (1, K, 128)

        return beh_tensor, lex_tensor, seq_len

    def detect(
        self,
        stream_input: Union[DNSStream, pd.DataFrame, List[Dict[str, Any]]],
        stream_id: str = "canonical_stream",
    ) -> DetectionResult:
        """
        Executes sequential Multi-View inference with Adaptive Evidence Early Stopping.
        
        Args:
            stream_input: DNSStream, pandas DataFrame, or list of observation dicts.
            stream_id: Identifier string for logging/tracking.
            
        Returns:
            DetectionResult dataclass with decision, confidence, and observation metrics.
        """
        # Convert input to canonical DNSStream
        if isinstance(stream_input, pd.DataFrame):
            stream = DNSStream(stream_id=stream_id)
            for _, row in stream_input.iterrows():
                stream.add_observation(row.to_dict())
        elif isinstance(stream_input, list):
            stream = DNSStream(stream_id=stream_id)
            for item in stream_input:
                stream.add_observation(item)
        elif isinstance(stream_input, DNSStream):
            stream = stream_input
        else:
            raise TypeError(f"Unsupported stream input type: {type(stream_input)}")

        if len(stream) < self.min_horizon:
            raise ValueError(
                f"DNSStream contains {len(stream)} observations. "
                f"DeepDNS requires at least {self.min_horizon} observations for sequential evaluation."
            )

        # Preprocess stream into tensors
        beh_tensor, lex_tensor, seq_len = self.preprocess_stream(stream)
        beh_tensor = beh_tensor.to(self.device)
        lex_tensor = lex_tensor.to(self.device)

        # Model Forward Pass
        with torch.no_grad():
            step_logits, _, z_beh_all, z_lex_all, _ = self.classifier.model(
                beh_tensor, lex_tensor, mode=self.fusion_mode
            )
            # Extract softmax probabilities for Attack class (index 1) across all steps
            step_probs = torch.softmax(step_logits, dim=-1)[0, :, 1].cpu().numpy()

        # Sequential AEC Evaluation
        decision = self.aec.evaluate_sequence_probabilities(step_probs)

        # Map to Canonical Result
        final_decision_str = DecisionEnum.ATTACK.value if decision.predicted_label == 1 else DecisionEnum.BENIGN.value
        p_atk = float(decision.probability_attack)
        p_ben = float(1.0 - p_atk)
        confidence = p_atk if decision.predicted_label == 1 else p_ben
        k_stop = decision.stopping_horizon
        consumed = k_stop
        saved = max(0, self.max_horizon - k_stop)
        savings_pct = (saved / self.max_horizon) * 100.0

        return DetectionResult(
            decision=final_decision_str,
            confidence=round(confidence, 4),
            attack_probability=round(p_atk, 4),
            benign_probability=round(p_ben, 4),
            stopping_horizon=k_stop,
            max_horizon=self.max_horizon,
            observations_consumed=consumed,
            observations_saved=saved,
            query_savings_pct=round(savings_pct, 2),
            is_early_decision=decision.is_early_stop,
            stop_reason=decision.stopping_reason,
            evidence_history={k: round(v, 4) for k, v in decision.step_probabilities.items()},
            model_name=self.model_path.stem,
            mode=self.mode.value,
        )

    def process_observation(
        self,
        stream: DNSStream,
        obs: Union[DNSObservation, Dict[str, Any]],
    ) -> Tuple[Optional[DetectionResult], str]:
        """
        Progressively appends a DNS observation to a stream and triggers AEC evaluation
        only when allowed observation horizons K in {5, 10, 15, 20, 25, 30} are reached.
        
        Returns:
            Tuple of (DetectionResult if stopped or forced, status string).
        """
        stream.add_observation(obs)
        curr_len = len(stream)

        # If not at an allowed horizon, continue accumulating
        if curr_len not in self.allowed_horizons:
            return None, DecisionEnum.CONTINUE.value

        # Evaluate current prefix
        result = self.detect(stream)

        # If stopped early or reached max horizon, return result
        if result.is_early_decision or curr_len >= self.max_horizon:
            return result, result.decision

        return None, DecisionEnum.CONTINUE.value
