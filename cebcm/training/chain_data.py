"""
Chain data generation for Stage 3 Chain Head training.

Hard negative strategy — negatives that defeat cosine-distance shortcuts:

  1. Adjacent-swap: swap 1-2 neighboring pairs (NOT full shuffle).
     Forces model to learn fine-grained order, not just "is it shuffled?"
  2. Truncated: remove 1 key step from the middle.
     Chain is shorter but otherwise coherent — tests completeness.
  3. Interpolated corruption: blend one vector toward a random direction.
     Stays on-manifold (cos~0.6-0.8 with neighbors), not trivially detectable.
  4. Same-document wrong conclusion: replace last vector with another
     vector FROM THE SAME SEQUENCE. cos to prefix stays ~0.7-0.9,
     eliminating the cross-document cosine shortcut entirely.

Positive chains come from sequential SONAR vectors in dialogues/paragraphs.
Curriculum: adjacent-swap (medium) → same-doc wrong conclusion (hardest).
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
    num_negatives: int = 15      # Increased: 7 was too few for meaningful ranking
    # Negative type distribution (must sum to 1.0)
    neg_ratio_adj_swap: float = 0.25       # Adjacent-swap (replaces full shuffle)
    neg_ratio_truncated: float = 0.20
    neg_ratio_corrupted: float = 0.25      # Interpolated corruption (on-manifold)
    neg_ratio_wrong_conclusion: float = 0.30  # Same-document wrong conclusion
    # Corruption parameters
    corruption_alpha_min: float = 0.3   # Min interpolation toward random vector
    corruption_alpha_max: float = 0.7   # Max interpolation toward random vector
    target_norm: float = 0.2051         # SONAR embedding norm
    # Curriculum
    curriculum_enabled: bool = True


def apply_curriculum(cfg: ChainDataConfig, progress: float) -> ChainDataConfig:
    """
    Adjust negative ratios based on training progress.

    Curriculum: medium → hard
      progress 0.0-0.3: 35% adj-swap, 25% truncated, 25% corrupted, 15% wrong-conclusion
      progress 0.3-0.6: 25% each (balanced)
      progress 0.6-1.0: 15% adj-swap, 15% truncated, 25% corrupted, 45% wrong-conclusion
    """
    if not cfg.curriculum_enabled:
        return cfg

    import copy
    new_cfg = copy.copy(cfg)

    if progress < 0.3:
        new_cfg.neg_ratio_adj_swap = 0.35
        new_cfg.neg_ratio_truncated = 0.25
        new_cfg.neg_ratio_corrupted = 0.25
        new_cfg.neg_ratio_wrong_conclusion = 0.15
    elif progress < 0.6:
        new_cfg.neg_ratio_adj_swap = 0.25
        new_cfg.neg_ratio_truncated = 0.20
        new_cfg.neg_ratio_corrupted = 0.25
        new_cfg.neg_ratio_wrong_conclusion = 0.30
    else:
        # Hard: emphasize same-doc wrong conclusion (hardest)
        new_cfg.neg_ratio_adj_swap = 0.15
        new_cfg.neg_ratio_truncated = 0.15
        new_cfg.neg_ratio_corrupted = 0.25
        new_cfg.neg_ratio_wrong_conclusion = 0.45

    return new_cfg


def generate_negative_chains(
    positive: Tensor,
    same_doc_vectors: Tensor,
    global_pool: Tensor,
    cfg: ChainDataConfig,
) -> tuple[Tensor, Tensor, list[str]]:
    """
    Generate hard negative chains from a positive chain.

    Args:
        positive: [L, D] positive chain vectors
        same_doc_vectors: [S, D] ALL vectors from the SAME source sequence
            (for same-document wrong conclusions — eliminates cross-doc shortcut)
        global_pool: [pool_size, D] global vector pool (fallback)
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
    n_swap = max(1, round(N * cfg.neg_ratio_adj_swap))
    n_truncated = max(1, round(N * cfg.neg_ratio_truncated))
    n_corrupted = max(1, round(N * cfg.neg_ratio_corrupted))
    n_wrong = N - n_swap - n_truncated - n_corrupted
    n_wrong = max(0, n_wrong)
    total = n_swap + n_truncated + n_corrupted + n_wrong
    if total < N:
        n_wrong += N - total

    negatives = []
    neg_lengths = []
    neg_types = []

    # ─── 1. Adjacent-swap: swap neighboring pairs ─────────────────────
    # Scale swap count with chain length: 1 swap in 5-vec chain is 20%,
    # but 1 swap in 15-vec chain is only 7% — too subtle for SONAR.
    # Use floor(L/4) as minimum to keep relative disruption ~25%.
    for _ in range(n_swap):
        swapped = positive.clone()
        min_swaps = max(1, L // 4)
        max_swaps = min(min_swaps + 1, L - 1)
        n_swaps = random.randint(min_swaps, max_swaps)
        swap_positions = random.sample(range(L - 1), n_swaps)
        for pos in swap_positions:
            swapped[pos], swapped[pos + 1] = swapped[pos + 1].clone(), swapped[pos].clone()
        negatives.append(swapped)
        neg_lengths.append(L)
        neg_types.append("adj_swap")

    # ─── 2. Truncated: remove 1 step from the middle ─────────────────
    for _ in range(n_truncated):
        if L <= cfg.min_chain_len:
            drop_count = 1
        else:
            drop_count = 1  # Only 1 — harder to detect than removing 3
        middle_start = max(1, L // 4)
        middle_end = min(L - 1, 3 * L // 4)
        if middle_end <= middle_start:
            drop_idx = random.randint(1, L - 2) if L > 2 else 0
            drop_indices = [drop_idx]
        else:
            drop_indices = random.sample(range(middle_start, middle_end), drop_count)
        keep_mask = torch.ones(L, dtype=torch.bool, device=device)
        keep_mask[drop_indices] = False
        truncated = positive[keep_mask]
        negatives.append(truncated)
        neg_lengths.append(truncated.shape[0])
        neg_types.append("truncated")

    # ─── 3. Interpolated corruption: blend toward random (on-manifold) ─
    # Instead of replacing with pure noise (cos~0 = trivially detectable),
    # interpolate: v_corrupt = (1-α)·v_original + α·v_random
    # This keeps cos(v_corrupt, neighbors) ~ 0.5-0.8 — on manifold.
    for _ in range(n_corrupted):
        corrupt_idx = random.randint(1, max(1, L - 2))
        corrupted = positive.clone()
        alpha = random.uniform(cfg.corruption_alpha_min, cfg.corruption_alpha_max)
        # Pick random direction from global pool (not pure randn)
        rand_idx = random.randint(0, global_pool.shape[0] - 1)
        rand_vec = global_pool[rand_idx]
        blended = (1 - alpha) * positive[corrupt_idx] + alpha * rand_vec
        # Project back to SONAR sphere
        blended = F.normalize(blended, dim=-1) * cfg.target_norm
        corrupted[corrupt_idx] = blended
        negatives.append(corrupted)
        neg_lengths.append(L)
        neg_types.append("corrupted")

    # ─── 4. Same-document wrong conclusion ────────────────────────────
    # Replace last vector with another vector from the SAME sequence.
    # This eliminates the cross-document cosine shortcut entirely —
    # all vectors have similar cos to the chain prefix (~0.7-0.9).
    # The model must learn that the SPECIFIC conclusion doesn't follow
    # from the SPECIFIC reasoning steps.
    for _ in range(n_wrong):
        wrong = positive.clone()
        correct_last = positive[-1]

        # Try to find a same-doc vector that's different from correct conclusion
        if same_doc_vectors.shape[0] > 1:
            # Compute cosine to correct conclusion
            cos_all = F.cosine_similarity(
                correct_last.unsqueeze(0), same_doc_vectors, dim=-1
            )  # [S]
            # Exclude vectors too similar to correct (cos > 0.95 = near-duplicate)
            # and the chain vectors themselves
            valid_mask = cos_all < 0.95
            # Also exclude vectors that ARE in the positive chain
            for chain_vec in positive:
                chain_cos = F.cosine_similarity(
                    chain_vec.unsqueeze(0), same_doc_vectors, dim=-1
                )
                valid_mask = valid_mask & (chain_cos < 0.99)

            valid_indices = valid_mask.nonzero(as_tuple=False).squeeze(-1)
            if valid_indices.numel() > 0:
                # Pick the HARDEST: highest cos to correct that isn't the correct one
                valid_cos = cos_all[valid_indices]
                # Top-k hardest candidates, pick one randomly for diversity
                k = min(5, valid_indices.numel())
                topk_local = valid_cos.topk(k).indices
                chosen_local = topk_local[random.randint(0, k - 1)]
                chosen_idx = valid_indices[chosen_local]
                wrong[-1] = same_doc_vectors[chosen_idx]
            else:
                # Fallback: use global pool
                rand_idx = random.randint(0, global_pool.shape[0] - 1)
                wrong[-1] = global_pool[rand_idx]
        else:
            # Single-vector sequence fallback
            rand_idx = random.randint(0, global_pool.shape[0] - 1)
            wrong[-1] = global_pool[rand_idx]

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

    Stores per-chain source sequence index so same-document wrong conclusions
    can be generated (defeating the cross-document cosine shortcut).
    """

    def __init__(
        self,
        sequences: list[Tensor],   # list of [L_i, D] variable-length SONAR sequences
        cfg: ChainDataConfig,
    ):
        super().__init__()
        self.cfg = cfg

        # Store source sequences for same-document negative generation
        self.source_sequences: list[Tensor] = []
        self.all_vectors: list[Tensor] = []

        for seq in sequences:
            if not isinstance(seq, Tensor) or seq.dim() != 2:
                continue
            seq_len = seq.shape[0]
            if seq_len < cfg.min_chain_len:
                continue
            self.source_sequences.append(seq)
            self.all_vectors.append(seq)

        # Build global vector pool (for fallback)
        if self.all_vectors:
            self.global_pool = torch.cat(self.all_vectors, dim=0)
        else:
            d = sequences[0].shape[-1] if sequences else 1024
            self.global_pool = torch.randn(1000, d)
            self.global_pool = F.normalize(self.global_pool, dim=-1) * cfg.target_norm

    def set_curriculum_progress(self, progress: float) -> None:
        """Update negative ratios based on training progress [0.0, 1.0]."""
        self.cfg = apply_curriculum(self.cfg, progress)

    def __len__(self) -> int:
        # Each source sequence yields one chain per epoch access.
        # With shuffle=True in DataLoader, each epoch sees different
        # random chains from each sequence — much more diversity than
        # pre-extracted overlapping sliding windows.
        return len(self.source_sequences)

    def __getitem__(self, idx: int) -> dict:
        # Extract a RANDOM chain from the source sequence each time.
        # This gives different chains each epoch, preventing overfitting
        # on fixed pre-extracted overlapping chains.
        seq = self.source_sequences[idx]
        seq_len = seq.shape[0]

        max_cl = min(self.cfg.max_chain_len, seq_len)
        min_cl = min(self.cfg.min_chain_len, seq_len)
        cl = random.randint(min_cl, max_cl)
        start = random.randint(0, seq_len - cl)
        positive = seq[start:start + cl]
        L = positive.shape[0]

        negatives, neg_lengths, neg_types = generate_negative_chains(
            positive, seq, self.global_pool, self.cfg
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
    Returns neg_types flattened for per-type metric tracking.
    """
    B = len(batch)
    D = batch[0]["positive"].shape[-1]

    max_pos_len = max(b["pos_length"] for b in batch)
    max_neg_len = max(b["negatives"].shape[1] for b in batch)
    N = batch[0]["negatives"].shape[0]

    positives = torch.zeros(B, max_pos_len, D)
    pos_lengths = torch.zeros(B, dtype=torch.long)
    negatives = torch.zeros(B, N, max_neg_len, D)
    neg_lengths = torch.zeros(B, N, dtype=torch.long)
    all_neg_types: list[list[str]] = []

    for i, b in enumerate(batch):
        pl = b["pos_length"]
        positives[i, :pl] = b["positive"]
        pos_lengths[i] = pl

        nl = b["negatives"].shape[1]
        negatives[i, :, :nl] = b["negatives"]
        neg_lengths[i] = b["neg_lengths"]
        all_neg_types.append(b["neg_types"])

    return {
        "positives": positives,        # [B, max_pos_L, D]
        "pos_lengths": pos_lengths,    # [B]
        "negatives": negatives,        # [B, N, max_neg_L, D]
        "neg_lengths": neg_lengths,    # [B, N]
        "neg_types": all_neg_types,    # [B][N] str labels
    }
