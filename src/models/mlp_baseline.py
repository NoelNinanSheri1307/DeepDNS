"""
Lightweight Multi-Layer Perceptron (MLP) Baseline for DeepDNS.

Implements a 12-dimensional causal neural feature classifier in PyTorch:
Input (12) -> Linear(12, 64) -> LayerNorm -> ReLU -> Dropout -> Linear(64, 32) -> ReLU -> Linear(32, 2)
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.data.schema import CAUSAL_FEATURE_NAMES, FORBIDDEN_MODEL_COLUMNS


class MLPNetwork(nn.Module):
    """PyTorch Neural Network module for the MLP baseline."""

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dims: Tuple[int, int] = (64, 32),
        num_classes: int = 2,
        dropout_rate: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.use_layer_norm = use_layer_norm

        h1, h2 = hidden_dims

        # Layer 1
        self.fc1 = nn.Linear(input_dim, h1)
        self.ln1 = nn.LayerNorm(h1) if use_layer_norm else nn.Identity()
        self.act1 = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)

        # Layer 2
        self.fc2 = nn.Linear(h1, h2)
        self.act2 = nn.ReLU()

        # Output Layer
        self.fc_out = nn.Linear(h2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (Batch_Size, input_dim)
        Returns:
            Logits tensor of shape (Batch_Size, num_classes)
        """
        h = self.fc1(x)
        h = self.ln1(h)
        h = self.act1(h)
        h = self.dropout(h)

        h = self.fc2(h)
        h = self.act2(h)

        logits = self.fc_out(h)
        return logits


class MLPClassifierWrapper:
    """
    High-level scikit-learn compatible wrapper for the PyTorch MLP baseline.
    """

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dims: Tuple[int, int] = (64, 32),
        num_classes: int = 2,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        random_seed: int = 42,
        device: Optional[str] = None,
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.feature_names = CAUSAL_FEATURE_NAMES

        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._set_seed(self.random_seed)

        self.model = MLPNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            num_classes=self.num_classes,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

        self.is_fitted = False

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def _validate_input(self, X: Union[pd.DataFrame, np.ndarray, torch.Tensor]) -> torch.Tensor:
        """Validates input matrix and verifies absence of forbidden features."""
        if isinstance(X, pd.DataFrame):
            for col in FORBIDDEN_MODEL_COLUMNS:
                if col in X.columns:
                    raise ValueError(f"Data Leakage Breach: Forbidden column '{col}' passed to model!")
            arr = X[self.feature_names].values.astype(np.float32)
            tensor = torch.tensor(arr, dtype=torch.float32)
        elif isinstance(X, np.ndarray):
            arr = X.astype(np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] != len(self.feature_names):
                raise ValueError(f"Expected input shape (*, {len(self.feature_names)}), got {arr.shape}")
            tensor = torch.tensor(arr, dtype=torch.float32)
        elif isinstance(X, torch.Tensor):
            tensor = X.float()
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            if tensor.shape[1] != len(self.feature_names):
                raise ValueError(f"Expected input shape (*, {len(self.feature_names)}), got {tensor.shape}")
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

        return tensor

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray, torch.Tensor],
        y: Union[pd.Series, np.ndarray, torch.Tensor],
        epochs: int = 10,
        val_data: Optional[Tuple[Union[pd.DataFrame, np.ndarray], Union[pd.Series, np.ndarray]]] = None,
        verbose: bool = False,
    ) -> "MLPClassifierWrapper":
        """
        Trains the MLP network using cross-entropy loss and AdamW optimizer.
        """
        X_t = self._validate_input(X)
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_arr = y.values.astype(np.int64)
        elif isinstance(y, np.ndarray):
            y_arr = y.astype(np.int64)
        elif isinstance(y, torch.Tensor):
            y_arr = y.cpu().numpy().astype(np.int64)
        else:
            y_arr = np.asarray(y, dtype=np.int64)

        y_t = torch.tensor(y_arr, dtype=torch.long)

        dataset = TensorDataset(X_t, y_t)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

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
            for batch_x, batch_y in dataloader:
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(1, num_batches)
            if verbose or (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0 or epoch == epochs - 1:
                print(f"  [MLP Epoch {epoch+1:02d}/{epochs:02d}] Loss: {avg_loss:.4f}")

        self.is_fitted = True
        return self

    def predict_logits(self, X: Union[pd.DataFrame, np.ndarray, torch.Tensor]) -> np.ndarray:
        """Computes raw logits of shape (N, num_classes)."""
        X_t = self._validate_input(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
        return logits.cpu().numpy()

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray, torch.Tensor]) -> np.ndarray:
        """Computes softmax probabilities of shape (N, num_classes)."""
        logits_arr = self.predict_logits(X)
        logits_t = torch.tensor(logits_arr, dtype=torch.float32)
        probas = F.softmax(logits_t, dim=-1).numpy()
        return probas

    def predict(self, X: Union[pd.DataFrame, np.ndarray, torch.Tensor]) -> np.ndarray:
        """Predicts class indices (0 or 1)."""
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=-1)

    def save(self, file_path: Union[str, Path]) -> None:
        """Saves model weights and configuration to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": {
                    "input_dim": self.input_dim,
                    "hidden_dims": self.hidden_dims,
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
    def load(cls, file_path: Union[str, Path], device: Optional[str] = None) -> "MLPClassifierWrapper":
        """Loads serialized model from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: '{file_path}'")

        data = torch.load(path, map_location=device or "cpu", weights_only=False)
        cfg = data["config"]
        wrapper = cls(
            input_dim=cfg["input_dim"],
            hidden_dims=cfg["hidden_dims"],
            num_classes=cfg["num_classes"],
            dropout_rate=cfg["dropout_rate"],
            learning_rate=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"],
            batch_size=cfg["batch_size"],
            random_seed=cfg["random_seed"],
            device=device,
        )
        wrapper.model.load_state_dict(data["state_dict"])
        wrapper.is_fitted = data["is_fitted"]
        wrapper.feature_names = data["feature_names"]
        return wrapper
