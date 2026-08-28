"""
Multi-View Fusion Architecture for DeepDNS.

Fuses:
1. Temporal Behavioral Representation from Temporal GRU: z_beh in R^64
2. Lexical Domain Orthographic Representation from Character-CNN: z_lex in R^128

Produces a unified representation z_fuse in R^64 that retains high temporal evidence
accumulation while decoupling classification from artificial synthetic timing cadences.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.models.temporal_gru import TemporalGRUNetwork
from src.models.char_cnn import CharacterCNNEncoder, CharacterTokenizer
from src.evaluation.metrics import compute_classification_metrics, EvaluationReport

logger = logging.getLogger("MultiViewFusion")


class MultiViewFusionHead(nn.Module):
    """
    Late-Fusion Neural Network module combining behavioral and lexical embeddings.
    
    Architecture:
    z_beh (64) -> Linear(64, 64) -> LayerNorm -> ReLU ────────┐
                                                                ├─► Concat (128) ─► Linear(128, 64) ─► LayerNorm ─► ReLU ─► Dropout ─► z_fuse (64) ─► Linear(64, 2)
    z_lex (128) -> Linear(128, 64) -> LayerNorm -> ReLU ───────┘
    """

    def __init__(
        self,
        beh_dim: int = 64,
        lex_dim: int = 128,
        proj_dim: int = 64,
        fuse_dim: int = 64,
        num_classes: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.beh_dim = beh_dim
        self.lex_dim = lex_dim
        self.proj_dim = proj_dim
        self.fuse_dim = fuse_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Behavioral Projection Branch
        self.beh_proj = nn.Sequential(
            nn.Linear(beh_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )

        # Lexical Projection Branch
        self.lex_proj = nn.Sequential(
            nn.Linear(lex_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
        )

        # Fusion MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(proj_dim * 2, fuse_dim),
            nn.LayerNorm(fuse_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

        # Final Classification Head
        self.classifier = nn.Linear(fuse_dim, num_classes)

        # Auxiliary Ablation Heads (for behavioral-only and lexical-only modes)
        self.beh_only_head = nn.Linear(proj_dim, num_classes)
        self.lex_only_head = nn.Linear(proj_dim, num_classes)

    def forward(
        self,
        z_beh: torch.Tensor,
        z_lex: torch.Tensor,
        mode: str = "both",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes fused representation and logits.

        Args:
            z_beh: Tensor of shape (..., beh_dim).
            z_lex: Tensor of shape (..., lex_dim).
            mode: 'both', 'behavioral_only', or 'lexical_only'.

        Returns:
            logits: Tensor of shape (..., num_classes).
            z_fuse: Fused representation of shape (..., fuse_dim).
        """
        p_beh = self.beh_proj(z_beh)
        p_lex = self.lex_proj(z_lex)

        if mode == "behavioral_only":
            logits = self.beh_only_head(p_beh)
            return logits, p_beh
        elif mode == "lexical_only":
            logits = self.lex_only_head(p_lex)
            return logits, p_lex

        # Default: Full Dual-View Fusion
        concat_views = torch.cat([p_beh, p_lex], dim=-1)
        z_fuse = self.fusion_mlp(concat_views)
        logits = self.classifier(z_fuse)
        return logits, z_fuse


class DeepDNSMultiViewNetwork(nn.Module):
    """
    Unified Multi-View Neural Network for DeepDNS.
    
    Seamlessly integrates:
    - Temporal GRU (12 causal behavioral channels)
    - Character-CNN (raw lexical domain character tokens)
    - Multi-View Fusion Head
    """

    def __init__(
        self,
        beh_input_dim: int = 12,
        beh_hidden_dim: int = 64,
        lex_vocab_size: int = 45,
        lex_embedding_dim: int = 32,
        lex_output_dim: int = 128,
        fuse_dim: int = 64,
        num_classes: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.beh_input_dim = beh_input_dim
        self.beh_hidden_dim = beh_hidden_dim
        self.lex_vocab_size = lex_vocab_size
        self.lex_output_dim = lex_output_dim
        self.fuse_dim = fuse_dim
        self.num_classes = num_classes

        # 1. Temporal Behavioral View
        self.temporal_gru = TemporalGRUNetwork(
            input_dim=beh_input_dim,
            hidden_dim=beh_hidden_dim,
            num_layers=1,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )

        # 2. Lexical Orthographic View
        self.char_cnn = CharacterCNNEncoder(
            vocab_size=lex_vocab_size,
            embedding_dim=lex_embedding_dim,
            num_filters=32,
            kernel_sizes=(3, 5, 7),
            output_dim=lex_output_dim,
            dropout_rate=dropout_rate,
            num_classes=num_classes,
        )

        # 3. Multi-View Fusion Head
        self.fusion_head = MultiViewFusionHead(
            beh_dim=beh_hidden_dim,
            lex_dim=lex_output_dim,
            proj_dim=64,
            fuse_dim=fuse_dim,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
        )

    def forward(
        self,
        beh_seqs: torch.Tensor,
        lex_seqs: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
        mode: str = "both",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through both views and the fusion head.

        Args:
            beh_seqs: Tensor of shape (B, K, 12).
            lex_seqs: Integer tensor of shape (B, K, L) with character tokens.
            seq_lens: Optional tensor of shape (B,) with valid sequence lengths.
            mode: 'both', 'behavioral_only', or 'lexical_only'.

        Returns:
            step_logits: Tensor of shape (B, K, num_classes).
            final_logits: Tensor of shape (B, num_classes) at valid horizon length.
            z_beh_all: Behavioral embeddings across time (B, K, 64).
            z_lex_all: Lexical embeddings across time (B, K, 128).
            z_fuse_all: Fused representations across time (B, K, 64).
        """
        # 1. Behavioral GRU Forward Pass -> (B, K, 64)
        _, _, z_beh_all = self.temporal_gru(beh_seqs, seq_lens)

        # 2. Lexical Char-CNN Forward Pass -> (B, K, 128)
        z_lex_all = self.char_cnn.extract_embedding(lex_seqs)

        # 3. Fuse views at each observation step t in [1..K]
        step_logits, z_fuse_all = self.fusion_head(z_beh_all, z_lex_all, mode=mode)

        # 4. Extract final prediction at actual sequence length
        if seq_lens is not None:
            batch_size = beh_seqs.size(0)
            seq_lens_clamped = torch.clamp(seq_lens.long(), min=1, max=beh_seqs.size(1))
            batch_indices = torch.arange(batch_size, device=beh_seqs.device)
            final_logits = step_logits[batch_indices, seq_lens_clamped - 1]
        else:
            final_logits = step_logits[:, -1, :]

        return step_logits, final_logits, z_beh_all, z_lex_all, z_fuse_all


class DeepDNSMultiViewClassifier:
    """
    High-level orchestrator for training, evaluating, and running DeepDNS Multi-View Fusion.
    """

    def __init__(
        self,
        beh_input_dim: int = 12,
        beh_hidden_dim: int = 64,
        lex_vocab_size: int = 45,
        lex_embedding_dim: int = 32,
        lex_output_dim: int = 128,
        fuse_dim: int = 64,
        num_classes: int = 2,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        random_seed: int = 42,
        device: Optional[str] = None,
    ):
        self.beh_input_dim = beh_input_dim
        self.beh_hidden_dim = beh_hidden_dim
        self.lex_vocab_size = lex_vocab_size
        self.lex_embedding_dim = lex_embedding_dim
        self.lex_output_dim = lex_output_dim
        self.fuse_dim = fuse_dim
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.random_seed = random_seed

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.tokenizer = CharacterTokenizer()
        self._set_seed(self.random_seed)

        self.model = DeepDNSMultiViewNetwork(
            beh_input_dim=self.beh_input_dim,
            beh_hidden_dim=self.beh_hidden_dim,
            lex_vocab_size=self.lex_vocab_size,
            lex_embedding_dim=self.lex_embedding_dim,
            lex_output_dim=self.lex_output_dim,
            fuse_dim=self.fuse_dim,
            num_classes=self.num_classes,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

        self.is_fitted = False

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def fit(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        epochs: int = 10,
        mode: str = "both",
        verbose: bool = True,
        collate_fn: Optional[callable] = None,
    ) -> "DeepDNSMultiViewClassifier":
        """
        Trains the Multi-View Network using mini-batch AdamW optimizer.
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=(self.device == "cuda"),
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            num_batches = 0

            for batch in train_loader:
                beh_seqs, lex_seqs, lens, labels = batch[0], batch[1], batch[2], batch[3]
                beh_seqs = beh_seqs.to(self.device, non_blocking=True)
                lex_seqs = lex_seqs.to(self.device, non_blocking=True)
                lens = lens.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).long()

                optimizer.zero_grad()
                _, final_logits, _, _, _ = self.model(beh_seqs, lex_seqs, lens, mode=mode)
                loss = criterion(final_logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(1, num_batches)
            if verbose and (epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1):
                logger.info(f"  [MultiView Epoch {epoch+1:02d}/{epochs:02d}] Train Loss: {avg_loss:.4f}")

        self.is_fitted = True
        return self

    def predict_step_proba(
        self,
        beh_seqs: Union[np.ndarray, torch.Tensor],
        lex_seqs: Union[np.ndarray, torch.Tensor],
        mode: str = "both",
    ) -> np.ndarray:
        """Computes step-by-step prediction probabilities across time t=1..K."""
        beh_t = torch.as_tensor(beh_seqs, dtype=torch.float32, device=self.device)
        lex_t = torch.as_tensor(lex_seqs, dtype=torch.long, device=self.device)

        self.model.eval()
        with torch.no_grad():
            step_logits, _, _, _, _ = self.model(beh_t, lex_t, mode=mode)
            step_probs = F.softmax(step_logits, dim=-1)
        return step_probs.cpu().numpy()

    def predict_proba(
        self,
        beh_seqs: Union[np.ndarray, torch.Tensor],
        lex_seqs: Union[np.ndarray, torch.Tensor],
        seq_lens: Optional[Union[np.ndarray, torch.Tensor, List[int]]] = None,
        mode: str = "both",
    ) -> np.ndarray:
        """Computes final prediction probabilities at actual sequence length."""
        beh_t = torch.as_tensor(beh_seqs, dtype=torch.float32, device=self.device)
        lex_t = torch.as_tensor(lex_seqs, dtype=torch.long, device=self.device)
        lens_t = torch.as_tensor(seq_lens, dtype=torch.long, device=self.device) if seq_lens is not None else None

        self.model.eval()
        with torch.no_grad():
            _, final_logits, _, _, _ = self.model(beh_t, lex_t, lens_t, mode=mode)
            probs = F.softmax(final_logits, dim=-1)
        return probs.cpu().numpy()

    def predict(
        self,
        beh_seqs: Union[np.ndarray, torch.Tensor],
        lex_seqs: Union[np.ndarray, torch.Tensor],
        seq_lens: Optional[Union[np.ndarray, torch.Tensor, List[int]]] = None,
        mode: str = "both",
    ) -> np.ndarray:
        """Predicts binary class labels."""
        probs = self.predict_proba(beh_seqs, lex_seqs, seq_lens, mode=mode)
        return np.argmax(probs, axis=-1)

    def evaluate_at_horizons(
        self,
        dataset: Dataset,
        horizons: List[int] = [5, 10, 15, 20, 25, 30],
        mode: str = "both",
        split_name: str = "test",
        collate_fn: Optional[callable] = None,
    ) -> Dict[int, EvaluationReport]:
        """Evaluates Multi-View performance across expanding evidence horizons K."""
        self.model.eval()
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        all_step_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                beh_seqs, lex_seqs, lens, labels = batch[0], batch[1], batch[2], batch[3]
                beh_seqs = beh_seqs.to(self.device)
                lex_seqs = lex_seqs.to(self.device)

                step_logits, _, _, _, _ = self.model(beh_seqs, lex_seqs, mode=mode)
                probs = F.softmax(step_logits, dim=-1)

                all_step_probs.append(probs.cpu().numpy())
                all_labels.append(labels.numpy().astype(int))

        step_probs_all = np.concatenate(all_step_probs, axis=0)  # (N, max_seq_len, 2)
        labels_all = np.concatenate(all_labels, axis=0)          # (N,)

        horizon_reports = {}
        max_available_k = step_probs_all.shape[1]

        for k in horizons:
            if k > max_available_k:
                continue
            k_idx = k - 1
            probs_at_k = step_probs_all[:, k_idx, 1]
            preds_at_k = (probs_at_k >= 0.5).astype(int)

            rep = compute_classification_metrics(
                y_true=labels_all,
                y_pred=preds_at_k,
                y_prob=probs_at_k,
                model_name=f"MultiView_{mode.upper()}_K{k}",
                dataset_split=split_name,
            )
            horizon_reports[k] = rep

        return horizon_reports

    def save(self, file_path: Union[str, Path]) -> None:
        """Saves weights and configuration to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": {
                    "beh_input_dim": self.beh_input_dim,
                    "beh_hidden_dim": self.beh_hidden_dim,
                    "lex_vocab_size": self.lex_vocab_size,
                    "lex_embedding_dim": self.lex_embedding_dim,
                    "lex_output_dim": self.lex_output_dim,
                    "fuse_dim": self.fuse_dim,
                    "num_classes": self.num_classes,
                    "dropout_rate": self.dropout_rate,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "batch_size": self.batch_size,
                    "random_seed": self.random_seed,
                },
                "is_fitted": self.is_fitted,
            },
            path,
        )

    @classmethod
    def load(cls, file_path: Union[str, Path], device: Optional[str] = None) -> "DeepDNSMultiViewClassifier":
        """Loads serialized Multi-View Network from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: '{file_path}'")

        data = torch.load(path, map_location=device or "cpu", weights_only=False)
        cfg = data["config"]
        classifier = cls(
            beh_input_dim=cfg["beh_input_dim"],
            beh_hidden_dim=cfg["beh_hidden_dim"],
            lex_vocab_size=cfg["lex_vocab_size"],
            lex_embedding_dim=cfg["lex_embedding_dim"],
            lex_output_dim=cfg["lex_output_dim"],
            fuse_dim=cfg["fuse_dim"],
            num_classes=cfg["num_classes"],
            dropout_rate=cfg["dropout_rate"],
            learning_rate=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            batch_size=cfg["batch_size"],
            random_seed=cfg["random_seed"],
            device=device,
        )
        classifier.model.load_state_dict(data["state_dict"])
        classifier.is_fitted = data["is_fitted"]
        return classifier
