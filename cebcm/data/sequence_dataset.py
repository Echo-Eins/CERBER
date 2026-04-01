"""
Sequential SONAR Dataset for autoregressive training.

Loads pre-encoded sequences of SONAR vectors where each sequence
represents a coherent text unit (paragraph, dialogue, article).

Dataset format (.pt file):
{
    "sequences": list[Tensor],   # variable-length [L_i, 1024] per sequence
    "texts": list[list[str]],    # original sentences per sequence
    "source": str,               # dataset name
    "metadata": dict,            # encoding parameters, stats
}

Provides collated batches with padding, lengths, and optional type_ids.

Usage:
    dataset = SONARSequenceDataset("data/squad_sequences.pt")
    loader = DataLoader(dataset, batch_size=32, collate_fn=dataset.collate)
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


class SONARSequenceDataset(Dataset):
    """
    Dataset of SONAR vector sequences for autoregressive training.

    Each item is a sequence of SONAR vectors representing consecutive
    sentences from a coherent text (paragraph, article, or dialogue turn).
    """

    def __init__(
            self,
            path: str | Path,
            max_seq_len: int = 64,
            min_seq_len: int = 3,
    ):
        """
        Args:
            path: Path to .pt file with sequences
            max_seq_len: Truncate sequences longer than this
            min_seq_len: Skip sequences shorter than this
        """
        data = torch.load(path, weights_only=False)

        self.sequences: list[Tensor] = []
        self.texts: list[list[str]] = []
        self.max_seq_len = max_seq_len

        raw_seqs = data["sequences"]
        raw_texts = data.get("texts", [None] * len(raw_seqs))

        for seq, txt in zip(raw_seqs, raw_texts):
            if isinstance(seq, Tensor) and seq.dim() == 2 and seq.shape[0] >= min_seq_len:
                self.sequences.append(seq[:max_seq_len])
                if txt is not None:
                    self.texts.append(txt[:max_seq_len])
                else:
                    self.texts.append([])

        self.source = data.get("source", "unknown")
        self.metadata = data.get("metadata", {})
        self.d_model = self.sequences[0].shape[-1] if self.sequences else 1024

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with:
                "vectors": [L, D] SONAR vectors
                "length": int — actual sequence length
                "texts": list[str] — original sentences
        """
        seq = self.sequences[idx]
        return {
            "vectors": seq,
            "length": seq.shape[0],
            "texts": self.texts[idx],
        }

    @staticmethod
    def collate(batch: list[dict]) -> dict:
        """
        Collate variable-length sequences into a padded batch.

        Returns:
            dict with:
                "vectors": [B, max_L, D] padded SONAR vectors
                "lengths": [B] actual lengths
                "targets": [B, max_L-1, D] shifted targets for next-prediction
                "target_lengths": [B] target lengths (lengths - 1)
                "type_ids": [B, max_L] — all zeros (no type info yet)
        """
        vectors = [item["vectors"] for item in batch]
        lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)

        B = len(vectors)
        max_len = int(lengths.max().item())
        D = vectors[0].shape[-1]

        # Pad sequences
        padded = torch.zeros(B, max_len, D)
        for i, v in enumerate(vectors):
            padded[i, : v.shape[0]] = v

        # Targets: shifted by 1 (for next-vector prediction)
        targets = padded[:, 1:]  # [B, max_L-1, D]
        target_lengths = lengths - 1  # [B]

        # Type IDs: default all zeros (statements)
        type_ids = torch.zeros(B, max_len, dtype=torch.long)

        return {
            "vectors": padded,
            "lengths": lengths,
            "targets": targets,
            "target_lengths": target_lengths,
            "type_ids": type_ids,
        }

    def get_statistics(self) -> dict:
        """Compute dataset statistics."""
        lengths = [seq.shape[0] for seq in self.sequences]
        norms = [seq.norm(dim=-1).mean().item() for seq in self.sequences]
        return {
            "num_sequences": len(self.sequences),
            "mean_length": sum(lengths) / len(lengths) if lengths else 0,
            "max_length": max(lengths) if lengths else 0,
            "min_length": min(lengths) if lengths else 0,
            "mean_norm": sum(norms) / len(norms) if norms else 0,
            "source": self.source,
        }

    def split(
            self, train_ratio: float = 0.9, seed: int = 42
    ) -> tuple["SONARSequenceDataset", "SONARSequenceDataset"]:
        """Split into train/val datasets."""
        g = torch.Generator().manual_seed(seed)
        n = len(self.sequences)
        perm = torch.randperm(n, generator=g).tolist()
        split_idx = int(n * train_ratio)

        train_ds = SONARSequenceDataset.__new__(SONARSequenceDataset)
        train_ds.sequences = [self.sequences[i] for i in perm[:split_idx]]
        train_ds.texts = [self.texts[i] for i in perm[:split_idx]]
        train_ds.max_seq_len = self.max_seq_len
        train_ds.source = self.source
        train_ds.metadata = self.metadata
        train_ds.d_model = self.d_model

        val_ds = SONARSequenceDataset.__new__(SONARSequenceDataset)
        val_ds.sequences = [self.sequences[i] for i in perm[split_idx:]]
        val_ds.texts = [self.texts[i] for i in perm[split_idx:]]
        val_ds.max_seq_len = self.max_seq_len
        val_ds.source = self.source
        val_ds.metadata = self.metadata
        val_ds.d_model = self.d_model

        return train_ds, val_ds