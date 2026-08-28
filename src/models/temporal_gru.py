"""
Temporal GRU Architecture and Streaming Sequence Classifier for DeepDNS.

Implements causal recurrent temporal modeling over sequential DNS observation windows:
X = [x_1, x_2, ..., x_K] where each x_t in R^12.

Guarantees strict forward causality: prediction at step k depends exclusively on
observations x_1 ... x_k and is invariant to future tokens x_{k+1} ... x_K.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.data.schema import CAUSAL_FEATURE_NAMES, FORBIDDEN_MODEL_COLUMNS
from src.data.sequences import StreamingSequenceDataset, DEFAULT_MAX_SEQ_LEN, sequence_collate_fn
from src.evaluation.metrics import compute_classification_metrics, EvaluationReport

logger = logging.getLogger("TemporalGRU")


class TemporalGRUNetwork(nn.Module):
    """
    PyTorch Neural Network module implementing causal GRU recurrence over DNS streams.
    
    Exposes both step-by-step sequential representations (h_1 ... h_K) and 
    single-step streaming updates for real-time online inference.
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        num_layers: int = 1,
        num_classes: int = 2,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Causal Recurrent Backbone
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0.0,
        )

        # Classification Head applied to hidden state h_t
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(32, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass over sequence batch.

        Args:
            x: Input tensor of shape (Batch_Size, Seq_Len, input_dim).
            seq_lens: Optional 1D tensor of shape (Batch_Size,) with actual sequence lengths.

        Returns:
            step_logits: Tensor of shape (Batch_Size, Seq_Len, num_classes) with predictions at every step t.
            final_logits: Tensor of shape (Batch_Size, num_classes) with prediction at the valid sequence horizon.
            h_all: Tensor of shape (Batch_Size, Seq_Len, hidden_dim) with recurrent hidden states.
        """
        # gru_out shape: (Batch_Size, Seq_Len, hidden_dim)
        gru_out, h_n = self.gru(x)

        # Compute predictions across all steps: (Batch_Size, Seq_Len, num_classes)
        step_logits = self.classifier(gru_out)

        # Extract logits at actual sequence length
        if seq_lens is not None:
            batch_size = x.size(0)
            seq_lens_clamped = torch.clamp(seq_lens.long(), min=1, max=x.size(1))
            batch_indices = torch.arange(batch_size, device=x.device)
            final_logits = step_logits[batch_indices, seq_lens_clamped - 1]
        else:
            final_logits = step_logits[:, -1, :]

        return step_logits, final_logits, gru_out

    def forward_step(
        self,
        x_t: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step streaming execution for online deployment:
        x_t -> GRU(h_{t-1}, x_t) -> h_t -> prediction p_t

        Args:
            x_t: Tensor of shape (Batch_Size, input_dim) representing the current query.
            h_prev: Optional hidden state tensor of shape (num_layers, Batch_Size, hidden_dim).

        Returns:
            logits_t: Tensor of shape (Batch_Size, num_classes).
            h_t: Next hidden state tensor of shape (num_layers, Batch_Size, hidden_dim).
        """
        if x_t.dim() == 2:
            x_step = x_t.unsqueeze(1)  # (B, 1, input_dim)
        else:
            x_step = x_t

        out, h_next = self.gru(x_step, h_prev)
        logits_t = self.classifier(out.squeeze(1))
        return logits_t, h_next


class TemporalGRUClassifier:
    """
    High-level interface for training, evaluating, and running inference with Temporal GRU.
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 64,
        num_layers: int = 1,
        num_classes: int = 2,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 128,
        random_seed: int = 42,
        device: Optional[str] = None,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.feature_names = CAUSAL_FEATURE_NAMES

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._set_seed(self.random_seed)

        self.model = TemporalGRUNetwork(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

        self.is_fitted = False

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def _validate_tensor_input(self, X: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Validates 3D tensor shape (B, K, 12)."""
        if isinstance(X, np.ndarray):
            tensor = torch.tensor(X, dtype=torch.float32)
        elif isinstance(X, torch.Tensor):
            tensor = X.float()
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)

        if tensor.dim() != 3 or tensor.size(2) != len(self.feature_names):
            raise ValueError(
                f"Expected 3D input shape (Batch, Seq_Len, {len(self.feature_names)}), got {tensor.shape}"
            )
        return tensor

    def fit(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        epochs: int = 10,
        verbose: bool = True,
    ) -> "TemporalGRUClassifier":
        """
        Trains the Temporal GRU using mini-batch AdamW optimizer.
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=sequence_collate_fn,
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
                seqs, lens, labels = batch[0], batch[1], batch[2]
                seqs = seqs.to(self.device, non_blocking=True)
                lens = lens.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).long()

                optimizer.zero_grad()
                _, final_logits, _ = self.model(seqs, lens)
                loss = criterion(final_logits, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(1, num_batches)
            if verbose and (epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1):
                logger.info(f"  [GRU Epoch {epoch+1:02d}/{epochs:02d}] Train Loss: {avg_loss:.4f}")

        self.is_fitted = True
        return self

    def predict_step_proba(
        self,
        X: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Computes prediction probabilities for all sequence steps t=1..K.
        
        Args:
            X: Input tensor of shape (Batch_Size, Seq_Len, 12).

        Returns:
            step_probs: Array of shape (Batch_Size, Seq_Len, num_classes).
        """
        X_t = self._validate_tensor_input(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            step_logits, _, _ = self.model(X_t)
            step_probs = F.softmax(step_logits, dim=-1)
        return step_probs.cpu().numpy()

    def predict_proba(
        self,
        X: Union[np.ndarray, torch.Tensor],
        seq_lens: Optional[Union[np.ndarray, torch.Tensor, List[int]]] = None,
    ) -> np.ndarray:
        """
        Computes final prediction probabilities at actual sequence length.
        """
        X_t = self._validate_tensor_input(X).to(self.device)
        lens_t = None
        if seq_lens is not None:
            lens_t = torch.tensor(seq_lens, device=self.device, dtype=torch.long)

        self.model.eval()
        with torch.no_grad():
            _, final_logits, _ = self.model(X_t, lens_t)
            probs = F.softmax(final_logits, dim=-1)
        return probs.cpu().numpy()

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        seq_lens: Optional[Union[np.ndarray, torch.Tensor, List[int]]] = None,
    ) -> np.ndarray:
        """Predicts binary class labels (0 or 1)."""
        probs = self.predict_proba(X, seq_lens)
        return np.argmax(probs, axis=-1)

    def evaluate_at_horizons(
        self,
        dataset: StreamingSequenceDataset,
        horizons: List[int] = [5, 10, 15, 20, 25, 30],
        split_name: str = "test",
    ) -> Dict[int, EvaluationReport]:
        """
        Evaluates model metrics specifically at multiple evidence horizons K.
        """
        self.model.eval()
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=sequence_collate_fn,
        )

        all_step_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                seqs, lens, labels = batch[0], batch[1], batch[2]
                seqs = seqs.to(self.device)
                step_logits, _, _ = self.model(seqs)
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
                model_name=f"Temporal_GRU_K{k}",
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
                    "input_dim": self.input_dim,
                    "hidden_dim": self.hidden_dim,
                    "num_layers": self.num_layers,
                    "num_classes": self.num_classes,
                    "dropout_rate": self.dropout_rate,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "batch_size": self.batch_size,
                    "random_seed": self.random_seed,
                },
                "is_fitted": self.is_fitted,
                "feature_names": self.feature_names,
            },
            path,
        )

    @classmethod
    def load(cls, file_path: Union[str, Path], device: Optional[str] = None) -> "TemporalGRUClassifier":
        """Loads serialized Temporal GRU from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: '{file_path}'")

        data = torch.load(path, map_location=device or "cpu", weights_only=False)
        cfg = data["config"]
        classifier = cls(
            input_dim=cfg["input_dim"],
            hidden_dim=cfg["hidden_dim"],
            num_layers=cfg["num_layers"],
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
        classifier.feature_names = data["feature_names"]
        return classifier
