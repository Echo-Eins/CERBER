"""
Chain data generation for Stage 3 Chain Head training.

Generates positive reasoning chains and 4 types of negatives:
  1. Shuffled: same vectors in random order (tests order sensitivity)
  2. Truncated: missing a key reasoning step (tests completeness)
  3. Corrupted: one vector replaced with random noise (tests integrity)
  4. Wrong conclusion: correct steps → incorrect final answer (tests logic)

Positive chains come from sequential SONAR vectors in dialogues/paragraphs.
The order matters — V₁ → V₂ → ... → Vₙ represents a reasoning sequence.

Curriculum: easy negatives first (shuffled), then harder (wrong conclusion).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class ChainDataConfig:
    """Configuration for chain data generation."""
    min_chain_len: int = 5       # Minimum chain length
    max_chain_len: int = 15      # Maximum chain length (spec: 5-20)
    num_negatives: int = 7       # Negatives per positive (spec: up to 31)
    # Negative type distribution (must sum to 1.0)
    # Curriculum: adjust these during training
    neg_ratio_shuffled: float = 0.25
    neg_ratio_truncated: float = 0.25
    neg_ratio_corrupted: float = 0.25
    neg_ratio_wrong_conclusion: float = 0.25
    # Corruption parameters
    noise_std: float = 0.2       # Std for random vector replacement (~SONAR norm)
    target_norm: float = 0.2051  # SONAR embedding norm
    # Curriculum: adjust ratios based on epoch progress (0.0=start, 1.0=end)
    # Start: mostly shuffled (easy). End: mostly wrong_conclusion (hard).
    curriculum_enabled: bool = True


def apply_curriculum(cfg: ChainDataConfig, progress: float) -> ChainDataConfig:
    """
    Adjust negative ratios based on training progress.

    Curriculum: easy → medium → hard (from spec §5.5)
      - progress 0.0-0.2: 50% shuffled, 20% truncated, 20% corrupted, 10% wrong
      - progress 0.2-0.5: 25% each (balanced)
      - progress 0.5-1.0: 10% shuffled, 20% truncated, 25% corrupted, 45% wrong

    Args:
        cfg: Base config
        progress: Training progress in [0.0, 1.0]

    Returns:
        New config with adjusted ratios
    """
    if not cfg.curriculum_enabled:
        return cfg

    import copy
    new_cfg = copy.copy(cfg)

    if progress < 0.2:
        # Easy: mostly shuffled (obvious order violation)
        new_cfg.neg_ratio_shuffled = 0.50
        new_cfg.neg_ratio_truncated = 0.20
        new_cfg.neg_ratio_corrupted = 0.20
        new_cfg.neg_ratio_wrong_conclusion = 0.10
    elif progress < 0.5:
        # Medium: balanced
        new_cfg.neg_ratio_shuffled = 0.25
        new_cfg.neg_ratio_truncated = 0.25
        new_cfg.neg_ratio_corrupted = 0.25
        new_cfg.neg_ratio_wrong_conclusion = 0.25
    else:
        # Hard: mostly wrong conclusion (requires deep reasoning)
        new_cfg.neg_ratio_shuffled = 0.10
        new_cfg.neg_ratio_truncated = 0.20
        new_cfg.neg_ratio_corrupted = 0.25
        new_cfg.neg_ratio_wrong_conclusion = 0.45

    return new_cfg


def generate_negative_chains(
    positive: Tensor,
    all_vectors: Tensor,
    cfg: ChainDataConfig,
) -> tuple[Tensor, Tensor, list[str]]:
    """
    Generate negative chains from a positive chain.

    Args:
        positive: [L, D] positive chain vectors
        all_vectors: [pool_size, D] pool of vectors for wrong-conclusion sampling
        cfg: Chain data configuration

    Returns:
        negatives: [N, max_L, D] padded negative chains
        neg_lengths: [N] actual lengths of each negative
        neg_types: [N] string labels for each negative type
    """
    L, D = positive.shape
    N = cfg.num_negatives
    device = positive.device

    # Compute how many of each type
    n_shuffled = max(1, round(N * cfg.neg_ratio_shuffled))
    n_truncated = max(1, round(N * cfg.neg_ratio_truncated))
    n_corrupted = max(1, round(N * cfg.neg_ratio_corrupted))
    n_wrong = N - n_shuffled - n_truncated - n_corrupted
    n_wrong = max(0, n_wrong)
    # Adjust if total doesn't match
    total = n_shuffled + n_truncated + n_corrupted + n_wrong
    if total < N:
        n_shuffled += N - total

    negatives = []
    neg_lengths = []
    neg_types = []

    # 1. Shuffled: same vectors, random order
    for _ in range(n_shuffled):
        perm = torch.randperm(L, device=device)
        # Ensure it's actually shuffled (not identity permutation)
        attempts = 0
        while torch.equal(perm, torch.arange(L, device=device)) and attempts < 5:
            perm = torch.randperm(L, device=device)
            attempts += 1
        negatives.append(positive[perm])
        neg_lengths.append(L)
        neg_types.append("shuffled")

    # 2. Truncated: remove 1-3 key steps from the middle
    for _ in range(n_truncated):
        if L <= cfg.min_chain_len:
            # Can't truncate further — use random drop of 1
            drop_count = 1
        else:
            drop_count = random.randint(1, min(3, L - cfg.min_chain_len))
        # Remove from middle (not start/end — those are easier to detect)
        middle_start = max(1, L // 4)
        middle_end = min(L - 1, 3 * L // 4)
        if middle_end - middle_start < drop_count:
            drop_indices = random.sample(range(1, L - 1), min(drop_count, L - 2))
        else:
            drop_indices = random.sample(range(middle_start, middle_end), drop_count)
        keep_mask = torch.ones(L, dtype=torch.bool, device=device)
        keep_mask[drop_indices] = False
        truncated = positive[keep_mask]
        negatives.append(truncated)
        neg_lengths.append(truncated.shape[0])
        neg_types.append("truncated")

    # 3. Corrupted: replace one vector with random noise on SONAR sphere
    for _ in range(n_corrupted):
        corrupt_idx = random.randint(1, L - 2)  # Don't corrupt first/last
        corrupted = positive.clone()
        noise = torch.randn(D, device=device)
        noise = noise / noise.norm() * cfg.target_norm  # Project to SONAR sphere
        corrupted[corrupt_idx] = noise
        negatives.append(corrupted)
        neg_lengths.append(L)
        neg_types.append("corrupted")

    # 4. Wrong conclusion: correct prefix → wrong final vector
    for _ in range(n_wrong):
        wrong = positive.clone()
        # Replace last vector with a random vector from the pool
        # that is NOT similar to the correct conclusion
        pool_size = all_vectors.shape[0]
        random_idx = random.randint(0, pool_size - 1)
        candidate = all_vectors[random_idx]
        # Ensure it's actually different (cosine < 0.8)
        cos_sim = F.cosine_similarity(
            positive[-1].unsqueeze(0), candidate.unsqueeze(0)
        ).item()
        attempts = 0
        while cos_sim > 0.8 and attempts < 10:
            random_idx = random.randint(0, pool_size - 1)
            candidate = all_vectors[random_idx]
            cos_sim = F.cosine_similarity(
                positive[-1].unsqueeze(0), candidate.unsqueeze(0)
            ).item()
            attempts += 1
        wrong[-1] = candidate
        negatives.append(wrong)
        neg_lengths.append(L)
        neg_types.append("wrong_conclusion")

    # Pad all negatives to same length
    max_len = max(n.shape[0] for n in negatives)
    padded = torch.zeros(len(negatives), max_len, D, device=device)
    lengths_tensor = torch.zeros(len(negatives), dtype=torch.long, device=device)
    for i, neg in enumerate(negatives):
        padded[i, :neg.shape[0]] = neg
        lengths_tensor[i] = neg.shape[0]

    return padded, lengths_tensor, neg_types


class ChainDataset(Dataset):
    """
    Dataset that generates positive/negative chain pairs from SONAR sequences.

    Each item contains:
      - positive_chain: [L, D] sequential SONAR vectors
      - negative_chains: [N, max_L, D] negative chains
      - pos_length: int
      - neg_lengths: [N] int
      - neg_types: list[str]

    Source data: same SONAR sequence data used for Stage 2 training.
    Chains are contiguous subsequences of length min_chain_len..max_chain_len.
    """

    def __init__(
        self,
        sequences: list[Tensor],   # list of [L_i, D] variable-length SONAR sequences
        cfg: ChainDataConfig,
    ):
        super().__init__()
        self.cfg = cfg

        # Flatten all vectors into a pool for wrong-conclusion sampling
        self.all_vectors: list[Tensor] = []
        self.chains: list[Tensor] = []

        for seq in sequences:
            if not isinstance(seq, Tensor) or seq.dim() != 2:
                continue
            seq_len = seq.shape[0]
            if seq_len < cfg.min_chain_len:
                continue

            # Extract all valid chains from this sequence
            max_cl = min(cfg.max_chain_len, seq_len)
            min_cl = min(cfg.min_chain_len, seq_len)
            for start in range(seq_len - min_cl + 1):
                # Random chain length for diversity
                cl = random.randint(min_cl, min(max_cl, seq_len - start))
                chain = seq[start:start + cl]
                self.chains.append(chain)

            # Add all vectors to pool
            self.all_vectors.append(seq)

        # Build flat vector pool
        if self.all_vectors:
            self.vector_pool = torch.cat(self.all_vectors, dim=0)
        else:
            # Fallback: random vectors on SONAR sphere
            d = sequences[0].shape[-1] if sequences else 1024
            self.vector_pool = torch.randn(1000, d)
            self.vector_pool = F.normalize(self.vector_pool, dim=-1) * cfg.target_norm

    def set_curriculum_progress(self, progress: float) -> None:
        """Update negative ratios based on training progress [0.0, 1.0]."""
        self.cfg = apply_curriculum(self.cfg, progress)

    def __len__(self) -> int:
        return len(self.chains)

    def __getitem__(self, idx: int) -> dict:
        positive = self.chains[idx]
        L = positive.shape[0]

        negatives, neg_lengths, neg_types = generate_negative_chains(
            positive, self.vector_pool, self.cfg
        )

        return {
            "positive": positive,            # [L, D]
            "pos_length": L,
            "negatives": negatives,          # [N, max_neg_L, D]
            "neg_lengths": neg_lengths,      # [N]
            "neg_types": neg_types,          # list[str]
        }


def chain_collate_fn(batch: list[dict]) -> dict:
    """
    Collate chain data into padded batches.

    Pads positive chains and negative chains to uniform size within the batch.
    """
    B = len(batch)
    D = batch[0]["positive"].shape[-1]

    # Find max lengths
    max_pos_len = max(b["pos_length"] for b in batch)
    max_neg_len = max(b["negatives"].shape[1] for b in batch)
    N = batch[0]["negatives"].shape[0]  # Same for all (from config)

    # Allocate
    positives = torch.zeros(B, max_pos_len, D)
    pos_lengths = torch.zeros(B, dtype=torch.long)
    negatives = torch.zeros(B, N, max_neg_len, D)
    neg_lengths = torch.zeros(B, N, dtype=torch.long)

    for i, b in enumerate(batch):
        pl = b["pos_length"]
        positives[i, :pl] = b["positive"]
        pos_lengths[i] = pl

        nl = b["negatives"].shape[1]
        negatives[i, :, :nl] = b["negatives"]
        neg_lengths[i] = b["neg_lengths"]

    return {
        "positives": positives,        # [B, max_pos_L, D]
        "pos_lengths": pos_lengths,    # [B]
        "negatives": negatives,        # [B, N, max_neg_L, D]
        "neg_lengths": neg_lengths,    # [B, N]
    }
