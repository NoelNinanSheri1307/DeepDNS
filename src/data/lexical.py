"""
Lexical character-level tokenization and dataset loader for DNS Threats.

Encodes raw domain name strings into fixed-length integer token tensors for Character-level CNNs.
"""

import gzip
import string
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


DEFAULT_MAX_DOMAIN_LEN = 128
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class DNSThreatsLexicalTokenizer:
    """
    Character-level tokenizer for DNS domain names.
    
    Builds a deterministic ASCII vocabulary mapping characters to indices [0 ... V-1].
    Index 0: <pad>
    Index 1: <unk>
    Index 2+: Valid ASCII characters (a-z, 0-9, '-', '.', '_', etc.)
    """

    def __init__(self, max_length: int = DEFAULT_MAX_DOMAIN_LEN):
        self.max_length = max_length
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN
        self.pad_idx = 0
        self.unk_idx = 1

        # Build standard printable character vocabulary
        valid_chars = sorted(list(set(string.ascii_lowercase + string.digits + ".-_~+/=")))
        self.char_to_idx: Dict[str, int] = {
            self.pad_token: self.pad_idx,
            self.unk_token: self.unk_idx,
        }
        for idx, ch in enumerate(valid_chars, start=2):
            self.char_to_idx[ch] = idx

        self.idx_to_char: Dict[int, str] = {v: k for k, v in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)

    def encode_domain(self, domain_str: str) -> np.ndarray:
        """
        Converts a single domain string into a fixed-length numpy array of token IDs.
        
        Args:
            domain_str: Raw DNS domain query string.

        Returns:
            np.ndarray of shape (max_length,) with integer token IDs.
        """
        if not isinstance(domain_str, str):
            domain_str = str(domain_str) if domain_str is not None else ""

        # Normalize: convert to lowercase and strip whitespace
        cleaned = domain_str.strip().lower()

        # Truncate if exceeds max_length
        cleaned = cleaned[: self.max_length]

        tokens = np.full(self.max_length, self.pad_idx, dtype=np.int64)
        for i, ch in enumerate(cleaned):
            tokens[i] = self.char_to_idx.get(ch, self.unk_idx)

        return tokens

    def decode_tokens(self, token_ids: Union[np.ndarray, torch.Tensor, List[int]]) -> str:
        """Decodes integer token sequence back into a domain string."""
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


class DNSThreatsDataset(Dataset):
    """
    Streaming / chunk-compatible PyTorch Dataset for DNS Threats multiclass domains.
    """

    def __init__(
        self,
        domains: List[str],
        labels: List[int],
        tokenizer: Optional[DNSThreatsLexicalTokenizer] = None,
        max_length: int = DEFAULT_MAX_DOMAIN_LEN,
    ):
        if len(domains) != len(labels):
            raise ValueError("Length of domains and labels must match.")

        self.domains = domains
        self.labels = np.asarray(labels, dtype=np.int64)
        self.tokenizer = tokenizer or DNSThreatsLexicalTokenizer(max_length=max_length)
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.domains)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        domain_str = self.domains[idx]
        tokens = self.tokenizer.encode_domain(domain_str)
        token_tensor = torch.tensor(tokens, dtype=torch.long)
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
        return token_tensor, label_tensor

    @classmethod
    def load_from_gz(
        cls,
        gz_path: Union[str, Path],
        max_rows: Optional[int] = None,
        tokenizer: Optional[DNSThreatsLexicalTokenizer] = None,
        max_length: int = DEFAULT_MAX_DOMAIN_LEN,
    ) -> "DNSThreatsDataset":
        """
        Loads domain strings and multiclass labels directly from a compressed .csv.gz file.
        """
        path = Path(gz_path)
        if not path.exists():
            raise FileNotFoundError(f"DNS Threats file not found: '{gz_path}'")

        df = pd.read_csv(path, compression="gzip", nrows=max_rows)
        if "domain" not in df.columns or "class" not in df.columns:
            raise ValueError("CSV must contain 'domain' and 'class' columns.")

        domains = df["domain"].fillna("").astype(str).tolist()
        labels = df["class"].astype(int).tolist()

        return cls(domains=domains, labels=labels, tokenizer=tokenizer, max_length=max_length)
