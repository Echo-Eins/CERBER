"""
PyTorch Dataset for pre-encoded SONAR embeddings.

Loads .pt files produced by encode_dataset.py.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path


class SONARVectorDataset(Dataset):
    """
    Dataset of pre-encoded SONAR embeddings.

    Each item is a single embedding vector [1024].
    Texts are stored for evaluation/debugging.

    Args:
        path: Path to .pt file with {"embeddings": Tensor[N, 1024], "texts": list[str]}
    """

    def __init__(self, path: str | Path):
        data = torch.load(path, weights_only=False)
        self.embeddings: torch.Tensor = data["embeddings"]  # [N, 1024]
        self.texts: list[str] = data["texts"]
        self.norms: torch.Tensor = data.get(
            "norms", self.embeddings.norm(dim=-1)
        )

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.embeddings[idx]

    def get_text(self, idx: int) -> str:
        return self.texts[idx]

    def subset(self, start: int, end: int) -> "SONARVectorDataset":
        """Create a view of a contiguous subset (for train/test split)."""
        new = SONARVectorDataset.__new__(SONARVectorDataset)
        new.embeddings = self.embeddings[start:end]
        new.texts = self.texts[start:end]
        new.norms = self.norms[start:end]
        return new
