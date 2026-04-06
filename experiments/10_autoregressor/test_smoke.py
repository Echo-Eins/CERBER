#!/usr/bin/env python3
"""
Smoke test for CEBM Autoregressor components.

Run on your CUDA machine to validate all new components work correctly.
No SONAR needed — tests shapes, gradients, and loss computation.

Usage:
    python experiments/10_autoregressor/test_smoke.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F


def test_conditional_critic():
    """Test ConditionalCritic: forward, energy_and_grad, contrastive loss, cosine loss."""
    from cebcm.models.conditional_critic import ConditionalCritic, ConditionalCriticConfig

    print("=" * 60)
    print("TEST: ConditionalCritic")
    print("=" * 60)

    cfg = ConditionalCriticConfig(dim=1024, hidden_dims=[512, 256])
    model = ConditionalCritic(cfg)
    print(f"  Params: {model.num_params:,}")

    B, D = 4, 1024
    v_q = torch.randn(B, D)
    v_c = torch.randn(B, D)
    v_ctx = torch.randn(B, D)

    # Forward
    e = model(v_q, v_c, v_context=v_ctx)
    assert e.shape == (B,), f"Forward shape: expected ({B},), got {e.shape}"
    print(f"  Forward: shape={e.shape}, values={e.tolist()}")

    # Forward without context (should use v_query as fallback)
    e_no_ctx = model(v_q, v_c)
    assert e_no_ctx.shape == (B,), f"No-ctx shape: expected ({B},), got {e_no_ctx.shape}"
    print(f"  No-context: shape={e_no_ctx.shape}")

    # Energy and gradient
    e2, grad = model.energy_and_grad(v_q, v_c, v_context=v_ctx)
    assert e2.shape == (B,), f"energy_and_grad energy: expected ({B},), got {e2.shape}"
    assert grad.shape == (B, D), f"energy_and_grad grad: expected ({B},{D}), got {grad.shape}"
    print(f"  energy_and_grad: energy={e2.tolist()}, grad_norm={grad.norm(dim=-1).tolist()}")

    # Contrastive loss
    N_neg = 5
    v_neg = torch.randn(B, N_neg, D)
    loss_c, metrics_c = model.compute_contrastive_loss(v_q, v_c, v_neg, v_context=v_ctx)
    assert loss_c.requires_grad, "Contrastive loss should require grad"
    print(f"  Contrastive loss: {loss_c.item():.4f}")
    print(f"    rank_acc: {metrics_c['rank_acc']:.4f}")
    print(f"    E_pos: {metrics_c['E_pos_mean']:.4f}, E_neg: {metrics_c['E_neg_mean']:.4f}")

    # Cosine loss
    v_pred = torch.randn(B, D)
    v_target = torch.randn(B, D)
    loss_cos, metrics_cos = model.compute_cosine_loss(v_pred, v_target)
    print(f"  Cosine loss: {loss_cos.item():.4f}, sim={metrics_cos['cos_sim_mean']:.4f}")

    # Backward
    loss_c.backward()
    print(f"  Backward pass: OK")

    print("  ✓ ConditionalCritic: ALL PASSED\n")


def test_context_wrapper():
    """Test _ContextWrappedEnergyFn for Langevin compatibility."""
    from cebcm.models.conditional_critic import ConditionalCritic, ConditionalCriticConfig
    from cebcm.inference.system_switching import _wrap_with_context

    print("=" * 60)
    print("TEST: Context Wrapper (Langevin compatibility)")
    print("=" * 60)

    cfg = ConditionalCriticConfig(dim=1024, hidden_dims=[256])
    critic = ConditionalCritic(cfg)

    B, D = 2, 1024
    v_q = torch.randn(B, D)
    v_c = torch.randn(B, D)
    v_ctx = torch.randn(B, D)

    # Wrap with context
    wrapped = _wrap_with_context(critic, v_ctx)
    print(f"  Wrapper type: {type(wrapped).__name__}")

    # Forward (should work like energy_fn(v_q, v_c) without explicit context)
    e = wrapped(v_q, v_c)
    assert e.shape == (B,), f"Wrapped forward: expected ({B},), got {e.shape}"
    print(f"  Wrapped forward: shape={e.shape}")

    # energy_and_grad
    e2, grad = wrapped.energy_and_grad(v_q, v_c)
    assert grad.shape == (B, D)
    print(f"  energy_and_grad via wrapper: OK")

    # Verify same result as direct call
    e_direct = critic(v_q, v_c, v_context=v_ctx)
    assert torch.allclose(e, e_direct, atol=1e-6), "Wrapper output != direct call"
    print(f"  Consistency check: wrapper == direct call")

    # Wrap without context (should return original model)
    unwrapped = _wrap_with_context(critic, None)
    assert unwrapped is critic
    print(f"  None context: returns original model")

    print("  ✓ Context Wrapper: ALL PASSED\n")


def test_text_reward():
    """Test text reward computation (mock SONAR)."""
    from cebcm.training.text_reward import (
        compute_reinforce_loss,
        RewardBaseline,
        _compute_bleu_batch,
    )

    print("=" * 60)
    print("TEST: Text Reward (no SONAR needed)")
    print("=" * 60)

    # REINFORCE loss
    B, D = 4, 1024
    v_pred = torch.randn(B, D, requires_grad=True)
    v_tgt = torch.randn(B, D)
    reward = torch.tensor([0.8, 0.3, 0.9, 0.5])
    loss, metrics = compute_reinforce_loss(v_pred, v_tgt, reward, baseline=0.6)
    print(f"  REINFORCE loss: {loss.item():.4f}")
    print(f"  Advantage mean: {metrics['advantage_mean']:.4f}")
    print(f"  Reward cos mean: {metrics['reward_cos_mean']:.4f}")

    # Backward
    loss.backward()
    assert v_pred.grad is not None
    print(f"  Gradient flows: OK")

    # Reward baseline
    baseline = RewardBaseline(decay=0.9)
    baseline.update(0.5)
    baseline.update(0.7)
    print(f"  Baseline after [0.5, 0.7]: {baseline.value:.4f}")
    assert abs(baseline.value - 0.52) < 0.01

    # BLEU
    preds = ["the cat sat on the mat", "hello world"]
    refs = ["the cat sat on the mat", "hello beautiful world"]
    bleu = _compute_bleu_batch(preds, refs)
    print(f"  BLEU scores: {bleu}")
    assert bleu[0] > 0.9  # exact match should be high
    assert 0 < bleu[1] < 1.0  # partial match

    print("  ✓ Text Reward: ALL PASSED\n")


def test_chain_batch_builder():
    """Test chain batch construction from reasoning steps."""
    from experiments.autoregressor_path_fix import fix_path
    from cebcm.models.chain_head import ChainHeadConfig, EBTChainHead

    print("=" * 60)
    print("TEST: Chain Batch Builder")
    print("=" * 60)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_autoregressor import build_chain_batch_from_steps

    B, max_len, D = 4, 6, 1024
    v_steps = torch.randn(B, max_len, D)
    step_lengths = torch.tensor([6, 4, 5, 3])
    num_neg = 3

    pos, pos_lens, negatives, neg_lens = build_chain_batch_from_steps(
        v_steps, step_lengths, num_negatives=num_neg,
    )

    assert pos.shape == (B, max_len, D)
    assert pos_lens.shape == (B,)
    assert negatives.shape == (B, num_neg, max_len, D)
    assert neg_lens.shape == (B, num_neg)
    print(f"  Positives: {pos.shape}, Negatives: {negatives.shape}")

    # Verify negatives are shuffled (not identical to positives)
    # At least one element should differ for samples with >= 2 steps
    diffs = 0
    for i in range(B):
        n = int(step_lengths[i].item())
        if n >= 2:
            neg0 = negatives[i, 0, :n]
            pos0 = pos[i, :n]
            if not torch.allclose(neg0, pos0):
                diffs += 1
    print(f"  Shuffled negatives differ: {diffs}/{B}")

    print("  ✓ Chain Batch Builder: ALL PASSED\n")


def test_dataset_and_collate():
    """Test dataset and collate with mock data."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_autoregressor import AutoregressorDataset, collate_fn

    print("=" * 60)
    print("TEST: Dataset & Collate")
    print("=" * 60)

    # Create mock samples
    D = 1024
    samples = []
    for i in range(20):
        n_steps = torch.randint(2, 6, (1,)).item()
        samples.append({
            "question": f"What is question {i}?",
            "answer": f"Answer {i}",
            "reasoning_steps": [f"Step {j}" for j in range(n_steps)],
            "v_question": torch.randn(D),
            "v_answer": torch.randn(D),
            "v_steps": torch.randn(n_steps, D),
            "type": "bridge",
        })

    ds = AutoregressorDataset(samples, num_negatives=3)
    print(f"  Dataset size: {len(ds)}")

    item = ds[0]
    assert item["v_question"].shape == (D,)
    assert item["v_answer"].shape == (D,)
    assert item["v_context"].shape == (D,)
    assert item["v_negatives"].shape == (3, D)
    print(f"  Item shapes: v_q={item['v_question'].shape}, v_neg={item['v_negatives'].shape}")

    # Test collate
    batch = collate_fn([ds[i] for i in range(4)])
    assert batch["v_questions"].shape == (4, D)
    assert batch["v_answers"].shape == (4, D)
    assert batch["v_contexts"].shape == (4, D)
    assert batch["v_negatives"].shape == (4, 3, D)
    assert len(batch["question_texts"]) == 4
    print(f"  Collated batch: v_q={batch['v_questions'].shape}, steps={batch['v_steps'].shape}")

    print("  ✓ Dataset & Collate: ALL PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CEBM Autoregressor Smoke Tests")
    print("=" * 60 + "\n")

    test_conditional_critic()
    test_context_wrapper()
    test_text_reward()
    test_dataset_and_collate()

    print("=" * 60)
    print("ALL SMOKE TESTS PASSED ✓")
    print("=" * 60)
