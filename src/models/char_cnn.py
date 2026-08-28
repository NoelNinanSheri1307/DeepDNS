"""
Character-level Lexical Encoder (Character-CNN) for DeepDNS.

Extracts rich sub-word and orthographic representations z_lex in R^128 from raw
domain name query strings, capturing tunneling signatures such as high-entropy
subdomain randomization, chunking artifacts, and character distribution anomalies.
"""

import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from src.data.schema import FORBIDDEN_MODEL_COLUMNS

DEFAULT_MAX_CHAR_LEN = 128
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class CharacterTokenizer:
    """
    Deterministic character-level tokenizer for DNS domain names.
    
    Constructs a fixed, immutable vocabulary of printable ASCII characters.
    Index 0: <pad>
    Index 1: <unk>
    Index 2-44: Standard lowercase alphanumeric and valid DNS punctuation.
    """

    def __init__(self, max_length: int = DEFAULT_MAX_CHAR_LEN):
        self.max_length = max_length
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN
        self.pad_idx = 0
        self.unk_idx = 1

        # Fixed deterministic printable character set
        valid_chars = sorted(list(set(string.ascii_lowercase + string.digits + ".-_~+/=")))
        self.char_to_idx: Dict[str, int] = {
            self.pad_token: self.pad_idx,
            self.unk_token: self.unk_idx,
        }
        for idx, ch in enumerate(valid_chars, start=2):
            self.char_to_idx[ch] = idx

        self.idx_to_char: Dict[int, str] = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

    def encode(self, domain_str: str) -> np.ndarray:
        """
        Converts a single domain string into a fixed-length numpy array of token IDs.
        """
        if not isinstance(domain_str, str):
            domain_str = str(domain_str) if domain_str is not None else ""

        cleaned = domain_str.strip().lower()[: self.max_length]
        tokens = np.full(self.max_length, self.pad_idx, dtype=np.int64)
        for i, ch in enumerate(cleaned):
            tokens[i] = self.char_to_idx.get(ch, self.unk_idx)

        return tokens

    def batch_encode(self, domain_list: List[str]) -> np.ndarray:
        """
        Encodes a list of domain strings into a 2D numpy array of shape (N, max_length).
        """
        out = np.empty((len(domain_list), self.max_length), dtype=np.int64)
        for i, dom in enumerate(domain_list):
            out[i] = self.encode(dom)
        return out

    def decode(self, token_ids: Union[np.ndarray, torch.Tensor, List[int]]) -> str:
        """Decodes token array back to string."""
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids.cpu().numpy()
        else:
            ids = np.asarray(token_ids)

        chars = []
        for idx in ids:
            if idx == self.pad_idx:
                continue
            chars.append(self.idx_to_char.get(int(idx), self.unk_token))
        return "".join(chars)


class CharacterCNNEncoder(nn.Module):
    """
    Multi-Branch 1D Convolutional Neural Network for DNS Lexical Embedding.
    
    Architecture:
    Token Indices (B, L) -> Embedding(32) -> 3x Conv1D Branches (k=3, 5, 7) ->
    Global Max Pooling -> Concatenation (96) -> Linear Projection (128) ->
    LayerNorm -> ReLU -> Dropout -> z_lex in R^128.
    """

    def __init__(
        self,
        vocab_size: int = 45,
        embedding_dim: int = 32,
        num_filters: int = 32,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        output_dim: int = 128,
        dropout_rate: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.num_classes = num_classes

        # Character Embedding Layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0,
        )

        # Multi-Branch Parallel 1D Convolutions
        self.conv_branches = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2,
            )
            for k in kernel_sizes
        ])

        # Projection to 128-dimensional lexical embedding
        total_conv_channels = num_filters * len(kernel_sizes)  # 32 * 3 = 96
        self.projection = nn.Sequential(
            nn.Linear(total_conv_channels, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

        # Standalone Classification Head (used for lexical-only ablation)
        self.classifier = nn.Linear(output_dim, num_classes)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts lexical embedding z_lex in R^128 from token tensor.

        Args:
            x: Integer tensor of shape (Batch_Size, L) or (Batch_Size, K, L).

        Returns:
            z_lex: Tensor of shape (Batch_Size, 128) or (Batch_Size, K, 128).
        """
        if x.dim() == 3:
            # Multi-observation sequence: (B, K, L)
            batch_size, seq_len, char_len = x.shape
            x_flat = x.view(batch_size * seq_len, char_len)
            z_flat = self.extract_embedding(x_flat)
            return z_flat.view(batch_size, seq_len, self.output_dim)

        # 2D Single query: (B, L)
        # 1. Embedding: (B, L, E) -> Transpose for Conv1D: (B, E, L)
        embeds = self.embedding(x).transpose(1, 2)

        # 2. Multi-Branch Convolutions with Global Max Pooling
        conv_outputs = []
        for conv in self.conv_branches:
            c = F.relu(conv(embeds))               # (B, num_filters, L)
            pooled, _ = torch.max(c, dim=-1)       # Global max pool over L -> (B, num_filters)
            conv_outputs.append(pooled)

        # 3. Concatenate Filter Maps -> (B, 96)
        concat_features = torch.cat(conv_outputs, dim=-1)

        # 4. Project to 128D Representation -> (B, 128)
        z_lex = self.projection(concat_features)
        return z_lex

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass computing both lexical embedding z_lex and standalone logits.

        Args:
            x: Tensor of shape (B, L) or (B, K, L).

        Returns:
            logits: Standalone classification logits of shape (B, 2) or (B, K, 2).
            z_lex: Lexical embedding tensor of shape (B, 128) or (B, K, 128).
        """
        z_lex = self.extract_embedding(x)
        logits = self.classifier(z_lex)
        return logits, z_lex


class CharacterCNNClassifier:
    """
    High-level interface for training, evaluating, and running inference with Character-CNN.
    """

    def __init__(
        self,
        vocab_size: int = 45,
        embedding_dim: int = 32,
        num_filters: int = 32,
        kernel_sizes: Tuple[int, ...] = (3, 5, 7),
        output_dim: int = 128,
        dropout_rate: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        random_seed: int = 42,
        device: Optional[str] = None,
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.output_dim = output_dim
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

        self.model = CharacterCNNEncoder(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            num_filters=self.num_filters,
            kernel_sizes=self.kernel_sizes,
            output_dim=self.output_dim,
            dropout_rate=self.dropout_rate,
        ).to(self.device)

        self.is_fitted = False

    def _set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

    def _validate_input(self, X: Union[pd.DataFrame, np.ndarray, torch.Tensor, List[str]]) -> torch.Tensor:
        """Validates and tokenizes input into integer tensor."""
        if isinstance(X, pd.DataFrame):
            for col in FORBIDDEN_MODEL_COLUMNS:
                if col in X.columns and col != "sld":
                    raise ValueError(f"Data Leakage Breach: Forbidden column '{col}' passed to model!")
            if "sld" in X.columns:
                domains = X["sld"].fillna("").astype(str).tolist()
                tokens = self.tokenizer.batch_encode(domains)
                return torch.tensor(tokens, dtype=torch.long)
            elif "domain" in X.columns:
                domains = X["domain"].fillna("").astype(str).tolist()
                tokens = self.tokenizer.batch_encode(domains)
                return torch.tensor(tokens, dtype=torch.long)
            else:
                raise ValueError("DataFrame must contain 'sld' or 'domain' text column.")
        elif isinstance(X, list) and len(X) > 0 and isinstance(X[0], str):
            tokens = self.tokenizer.batch_encode(X)
            return torch.tensor(tokens, dtype=torch.long)
        elif isinstance(X, np.ndarray):
            return torch.tensor(X, dtype=torch.long)
        elif isinstance(X, torch.Tensor):
            return X.long()
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray, torch.Tensor, List[str]],
        y: Union[pd.Series, np.ndarray, torch.Tensor],
        epochs: int = 10,
        verbose: bool = False,
    ) -> "CharacterCNNClassifier":
        """Trains Character-CNN model."""
        X_t = self._validate_input(X)
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_arr = y.values.astype(np.int64)
        elif isinstance(y, torch.Tensor):
            y_arr = y.cpu().numpy().astype(np.int64)
        else:
            y_arr = np.asarray(y, dtype=np.int64)

        y_t = torch.tensor(y_arr, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

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
            for bx, by in loader:
                bx = bx.to(self.device, non_blocking=True)
                by = by.to(self.device, non_blocking=True)

                optimizer.zero_grad()
                logits, _ = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(1, num_batches)
            if verbose and (epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1):
                print(f"  [Char-CNN Epoch {epoch+1:02d}/{epochs:02d}] Loss: {avg_loss:.4f}")

        self.is_fitted = True
        return self

    def extract_features(self, X: Union[pd.DataFrame, np.ndarray, torch.Tensor, List[str]]) -> np.ndarray:
        """Extracts 128-dimensional lexical embedding array."""
        X_t = self._validate_input(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            z_lex = self.model.extract_embedding(X_t)
        return z_lex.cpu().numpy()

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray, torch.Tensor, List[str]]) -> np.ndarray:
        """Computes softmax probabilities."""
        X_t = self._validate_input(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits, _ = self.model(X_t)
            probs = F.softmax(logits, dim=-1)
        return probs.cpu().numpy()

    def predict(self, X: Union[pd.DataFrame, np.ndarray, torch.Tensor, List[str]]) -> np.ndarray:
        """Predicts binary class labels."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=-1)

    def save(self, file_path: Union[str, Path]) -> None:
        """Saves model weights and configuration to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "config": {
                    "vocab_size": self.vocab_size,
                    "embedding_dim": self.embedding_dim,
                    "num_filters": self.num_filters,
                    "kernel_sizes": self.kernel_sizes,
                    "output_dim": self.output_dim,
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
    def load(cls, file_path: Union[str, Path], device: Optional[str] = None) -> "CharacterCNNClassifier":
        """Loads serialized Character-CNN from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: '{file_path}'")

        data = torch.load(path, map_location=device or "cpu", weights_only=False)
        cfg = data["config"]
        classifier = cls(
            vocab_size=cfg["vocab_size"],
            embedding_dim=cfg["embedding_dim"],
            num_filters=cfg["num_filters"],
            kernel_sizes=tuple(cfg["kernel_sizes"]),
            output_dim=cfg["output_dim"],
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
