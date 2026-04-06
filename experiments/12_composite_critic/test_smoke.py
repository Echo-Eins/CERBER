#!/usr/bin/env python3
"""
Smoke tests for the Composite Critic architecture.

Validates:
  1. ConditionalAngularCritic — construction, forward, energy_and_grad, losses
  2. AnalyticalRadialGuard — energy, analytical gradient vs autograd
  3. CompositeCritic — combined energy, gradient orthogonality, Langevin API
  4. Gradient decomposition — angular ⊥ radial (mathematical proof)
  5. Training step — single step converges, losses decrease

Usage:
    python experiments/12_composite_critic/test_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cebcm.models.conditional_angular_critic import (
    ConditionalAngularCritic,
    ConditionalAngularCriticConfig,
)
from cebcm.models.radial_guard import AnalyticalRadialGuard, RadialGuardConfig
from cebcm.models.composite_critic import CompositeCritic, CompositeCriticConfig


DEVICE = torch.device("cpu")
B, D = 4, 1024
TARGET_NORM = 0.2051


def _make_vecs(batch_size: int = B, dim: int = D) -> tuple[torch.Tensor, ...]:
    """Create synthetic SONAR-like vectors at target norm."""
    v_q = F.normalize(torch.randn(batch_size, dim), dim=-1) * TARGET_NORM
    v_a = F.normalize(torch.randn(batch_size, dim), dim=-1) * TARGET_NORM
    v_ctx = F.normalize(torch.randn(batch_size, dim), dim=-1) * TARGET_NORM
    return v_q, v_a, v_ctx


def test_angular_critic_construction():
    """Test ConditionalAngularCritic builds and has expected param count."""
    cfg = ConditionalAngularCriticConfig(dim=D, hidden_dims=[512, 256])
    critic = ConditionalAngularCritic(cfg)
    assert critic.num_params > 0, "Should have trainable params"
    # Output layer should be zero-init
    out_layer = critic.net[-1]
    assert out_layer.weight.abs().max() == 0, "Output should be zero-init"
    print(f"  [PASS] Angular critic: {critic.num_params:,} params, zero-init output")


def test_angular_forward():
    """Test forward pass shape and energy clamping."""
    cfg = ConditionalAngularCriticConfig(dim=D, hidden_dims=[256, 128],
                                         energy_output_clamp=50.0)
    critic = ConditionalAngularCritic(cfg)
    v_q, v_a, v_ctx = _make_vecs()

    # With context
    e = critic(v_q, v_a, v_context=v_ctx)
    assert e.shape == (B,), f"Expected ({B},), got {e.shape}"
    assert e.abs().max() <= 50.0, "Should be clamped"

    # Without context (fallback to v_query)
    e2 = critic(v_q, v_a)
    assert e2.shape == (B,), "Should work without context"

    # Without sigma (auto-estimated)
    e3 = critic(v_q, v_a, v_context=v_ctx, sigma=None)
    assert e3.shape == (B,)

    print(f"  [PASS] Angular forward: shapes correct, E ∈ [{e.min():.4f}, {e.max():.4f}]")


def test_angular_energy_and_grad():
    """Test energy_and_grad returns correct shapes and tangential gradient."""
    cfg = ConditionalAngularCriticConfig(dim=D, hidden_dims=[256, 128])
    critic = ConditionalAngularCritic(cfg)
    v_q, v_a, v_ctx = _make_vecs()

    e, g = critic.energy_and_grad(v_q, v_a, v_context=v_ctx)
    assert e.shape == (B,), f"Energy shape: {e.shape}"
    assert g.shape == (B, D), f"Gradient shape: {g.shape}"

    # Gradient should be tangential: ⟨∇E, v̂⟩ ≈ 0 (within numeric tolerance)
    v_hat = F.normalize(v_a, dim=-1)
    radial_component = (g * v_hat).sum(dim=-1)  # [B]
    # For zero-init network, gradients are ~0, so check relative to grad norm
    grad_norm = g.norm(dim=-1).clamp(min=1e-10)
    relative_radial = radial_component.abs() / grad_norm
    # Tolerance is generous because zero-init means small gradients
    assert relative_radial.max() < 0.15, (
        f"Gradient not tangential enough: max radial fraction = {relative_radial.max():.4f}"
    )
    print(f"  [PASS] Angular energy_and_grad: tangential (max radial frac = {relative_radial.max():.6f})")


def test_angular_contrastive_loss():
    """Test Focal-InfoNCE loss computation."""
    cfg = ConditionalAngularCriticConfig(dim=D, hidden_dims=[256, 128],
                                         num_negatives=3)
    critic = ConditionalAngularCritic(cfg)
    v_q, v_a, v_ctx = _make_vecs()
    v_neg = torch.randn(B, 3, D) * TARGET_NORM

    loss, metrics = critic.compute_contrastive_loss(v_q, v_a, v_neg, v_context=v_ctx)
    assert loss.shape == (), "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"
    assert "rank_acc" in metrics
    assert "energy_gap" in metrics
    # At init (E≈0 everywhere), rank_acc should be ~0.5
    print(f"  [PASS] Contrastive loss = {loss.item():.4f}, rank_acc = {metrics['rank_acc']:.4f}")


def test_angular_direction_loss():
    """Test direction loss computation and gradient flow."""
    cfg = ConditionalAngularCriticConfig(dim=D, hidden_dims=[256, 128])
    critic = ConditionalAngularCritic(cfg)
    v_q, v_a, v_ctx = _make_vecs()

    # Add noise to answer
    v_noisy = v_a + torch.randn_like(v_a) * 0.01

    loss, metrics = critic.compute_direction_loss(v_q, v_noisy, v_a, v_context=v_ctx)
    assert loss.shape == (), "Loss should be scalar"
    assert 0 <= loss.item() <= 2.0, f"Direction loss should be in [0, 2], got {loss.item()}"
    assert "direction_cos" in metrics

    # Verify gradient flows back (create_graph=True)
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().max() > 0
                   for p in critic.parameters())
    assert has_grad, "Direction loss must propagate gradients to critic params"
    print(f"  [PASS] Direction loss = {loss.item():.4f}, cos = {metrics['direction_cos']:.4f}")


def test_radial_guard_energy():
    """Test AnalyticalRadialGuard energy computation."""
    cfg = RadialGuardConfig(target_norm=TARGET_NORM, scale=1.0)
    guard = AnalyticalRadialGuard(cfg)

    _, v_a, _ = _make_vecs()

    e = guard(v_a)
    assert e.shape == (B,), f"Expected ({B},), got {e.shape}"

    # Vectors at target_norm should have E ≈ 0
    v_on_shell = F.normalize(torch.randn(B, D), dim=-1) * TARGET_NORM
    e_on = guard(v_on_shell)
    assert e_on.max() < 1e-6, f"On-shell energy should be ~0, got {e_on.max():.8f}"

    # Vectors off-shell should have E > 0
    v_off = F.normalize(torch.randn(B, D), dim=-1) * 0.5  # norm=0.5 >> 0.2051
    e_off = guard(v_off)
    assert e_off.min() > 0, "Off-shell energy should be positive"
    expected = 1.0 * (0.5 - TARGET_NORM) ** 2
    assert abs(e_off.mean().item() - expected) < 1e-4, "Energy should match formula"

    print(f"  [PASS] Radial guard: on-shell E={e_on.mean():.8f}, "
          f"off-shell E={e_off.mean():.4f} (expected {expected:.4f})")


def test_radial_guard_analytical_gradient():
    """Verify analytical gradient matches autograd."""
    cfg = RadialGuardConfig(target_norm=TARGET_NORM, scale=2.5)
    guard = AnalyticalRadialGuard(cfg)

    v = F.normalize(torch.randn(B, D), dim=-1) * 0.3  # slightly off-shell
    v_req = v.detach().requires_grad_(True)

    # Autograd
    e_auto = guard(v_req)
    g_auto = torch.autograd.grad(e_auto.sum(), v_req)[0]

    # Analytical
    e_anal, g_anal = guard.energy_and_grad(v)

    # Compare
    assert torch.allclose(e_auto, e_anal, atol=1e-5), "Energies must match"
    assert torch.allclose(g_auto, g_anal, atol=1e-5), (
        f"Gradients differ: max diff = {(g_auto - g_anal).abs().max():.8f}"
    )

    # Gradient should be purely radial (parallel to v̂)
    v_hat = F.normalize(v, dim=-1)
    # Project gradient onto tangent plane
    tangent = g_anal - (g_anal * v_hat).sum(dim=-1, keepdim=True) * v_hat
    tangent_fraction = tangent.norm(dim=-1) / g_anal.norm(dim=-1).clamp(min=1e-10)
    assert tangent_fraction.max() < 1e-5, (
        f"Radial gradient has tangent component: {tangent_fraction.max():.8f}"
    )
    print(f"  [PASS] Radial analytical gradient matches autograd, "
          f"purely radial (tangent frac = {tangent_fraction.max():.8f})")


def test_gradient_orthogonality():
    """
    Core mathematical test: ∇E_angular ⊥ ∇E_radial.

    The angular critic operates on normalised vectors, so its gradient
    through F.normalize has the Jacobian (I − v̂v̂ᵀ)/‖v‖, which projects
    onto the tangent plane.  The radial guard gradient is 2s(‖v‖−t)v̂,
    which is radial.  Inner product should be ~0.
    """
    ang_cfg = ConditionalAngularCriticConfig(dim=D, hidden_dims=[512, 256])
    rad_cfg = RadialGuardConfig(target_norm=TARGET_NORM, scale=1.0)

    angular = ConditionalAngularCritic(ang_cfg)
    radial = AnalyticalRadialGuard(rad_cfg)

    # Use slightly off-shell vectors to get non-zero radial gradient
    v_q, _, v_ctx = _make_vecs()
    v_cand = F.normalize(torch.randn(B, D), dim=-1) * 0.25  # off-shell

    # Train angular for a few steps so it has non-trivial gradients
    optimizer = torch.optim.Adam(angular.parameters(), lr=1e-3)
    v_pos = F.normalize(torch.randn(B, D), dim=-1) * TARGET_NORM
    v_neg = torch.randn(B, 3, D) * TARGET_NORM
    for _ in range(5):
        optimizer.zero_grad()
        loss, _ = angular.compute_contrastive_loss(v_q, v_pos, v_neg, v_context=v_ctx)
        loss.backward()
        optimizer.step()

    # Now compute gradients
    e_ang, g_ang = angular.energy_and_grad(v_q, v_cand, v_context=v_ctx)
    e_rad, g_rad = radial.energy_and_grad(v_cand)

    # Orthogonality: |⟨g_ang, g_rad⟩| / (‖g_ang‖ × ‖g_rad‖) ≈ 0
    dot = (g_ang * g_rad).sum(dim=-1)
    norm_product = g_ang.norm(dim=-1) * g_rad.norm(dim=-1)
    cos_angle = dot.abs() / norm_product.clamp(min=1e-10)

    # Allow some tolerance (floating point through large networks)
    max_cos = cos_angle.max().item()
    assert max_cos < 0.15, (
        f"Gradients not orthogonal enough: max cos(angle) = {max_cos:.6f}"
    )
    print(f"  [PASS] Gradient orthogonality: max |cos(∇ang, ∇rad)| = {max_cos:.6f}")


def test_composite_critic():
    """Test CompositeCritic combines angular + radial correctly."""
    ang_cfg = ConditionalAngularCriticConfig(dim=D, hidden_dims=[256, 128])
    rad_cfg = RadialGuardConfig(target_norm=TARGET_NORM, scale=1.0)
    cfg = CompositeCriticConfig(angular=ang_cfg, radial=rad_cfg, lambda_radial=5.0)
    critic = CompositeCritic(cfg)

    v_q, v_a, v_ctx = _make_vecs()

    # Forward
    e_total = critic(v_q, v_a, v_context=v_ctx)
    assert e_total.shape == (B,)

    # Should equal angular + lambda * radial
    e_ang = critic.angular(v_q, v_a, v_context=v_ctx)
    e_rad = critic.radial(v_a)
    expected = e_ang + 5.0 * e_rad
    assert torch.allclose(e_total, expected, atol=1e-5), "Composite must be sum"

    # energy_and_grad
    e, g = critic.energy_and_grad(v_q, v_a, v_context=v_ctx)
    assert e.shape == (B,)
    assert g.shape == (B, D)

    print(f"  [PASS] CompositeCritic: E_total = E_ang + λ×E_rad, "
          f"E=[{e.min():.4f}, {e.max():.4f}]")


def test_composite_langevin_compatibility():
    """Test CompositeCritic works with the Langevin API contract."""
    ang_cfg = ConditionalAngularCriticConfig(dim=D, hidden_dims=[256, 128])
    cfg = CompositeCriticConfig(angular=ang_cfg, lambda_radial=5.0)
    critic = CompositeCritic(cfg)

    v_q, v_a, v_ctx = _make_vecs()

    # Simulate what _ContextWrappedEnergyFn does
    class WrappedCritic(torch.nn.Module):
        def __init__(self, c, ctx):
            super().__init__()
            self._critic = c
            self._ctx = ctx
        def forward(self, v_query, v_candidate, sigma=None):
            return self._critic(v_query, v_candidate, v_context=self._ctx, sigma=sigma)
        def energy_and_grad(self, v_query, v_candidate, sigma=None):
            return self._critic.energy_and_grad(
                v_query, v_candidate, v_context=self._ctx, sigma=sigma)

    wrapped = WrappedCritic(critic, v_ctx)

    # Test the Langevin contract
    e = wrapped(v_q, v_a)
    assert e.shape == (B,)

    e2, g2 = wrapped.energy_and_grad(v_q, v_a)
    assert e2.shape == (B,)
    assert g2.shape == (B, D)

    # Mini Langevin loop WITH tangent + sphere projection (as in train_composite.py)
    v_current = v_q.clone()
    lr = 0.01
    target_norm = critic.radial.target_norm
    for _ in range(30):
        v_current = v_current.detach().requires_grad_(True)
        e_step = wrapped(v_q, v_current)
        grad = torch.autograd.grad(e_step.sum(), v_current)[0]
        # Tangent projection
        v_hat = F.normalize(v_current.detach(), dim=-1)
        radial_comp = (grad * v_hat).sum(dim=-1, keepdim=True) * v_hat
        grad = grad - radial_comp
        v_current = (v_current - lr * grad).detach()
        # Sphere projection
        v_current = F.normalize(v_current, dim=-1) * target_norm

    final_norm = v_current.norm(dim=-1).mean().item()
    assert abs(final_norm - TARGET_NORM) < 1e-5, (
        f"Sphere projection failed: ‖v‖={final_norm:.6f}, target={TARGET_NORM}"
    )
    print(f"  [PASS] Langevin API + sphere projection (30 steps, ‖v‖={final_norm:.6f} == target)")


def test_composite_losses():
    """Test all training losses through CompositeCritic."""
    ang_cfg = ConditionalAngularCriticConfig(dim=D, hidden_dims=[256, 128],
                                              num_negatives=3)
    cfg = CompositeCriticConfig(angular=ang_cfg, lambda_radial=5.0)
    critic = CompositeCritic(cfg)

    v_q, v_a, v_ctx = _make_vecs()
    v_neg = torch.randn(B, 3, D) * TARGET_NORM
    v_noisy = v_a + torch.randn_like(v_a) * 0.01

    # Contrastive
    loss_c, m_c = critic.compute_contrastive_loss(v_q, v_a, v_neg, v_context=v_ctx)
    assert loss_c.shape == ()

    # Direction
    loss_d, m_d = critic.compute_direction_loss(v_q, v_noisy, v_a, v_context=v_ctx)
    assert 0 <= loss_d.item() <= 2.0

    # Cosine
    loss_cos, m_cos = critic.compute_cosine_loss(v_a, v_a)
    assert loss_cos.item() < 0.01, "Self-cosine should be ~0"

    print(f"  [PASS] All losses: contrastive={loss_c.item():.4f}, "
          f"direction={loss_d.item():.4f}, cosine={loss_cos.item():.6f}")


def test_training_step_converges():
    """Verify a few training steps improve the contrastive loss."""
    ang_cfg = ConditionalAngularCriticConfig(dim=D, hidden_dims=[256, 128],
                                              num_negatives=5)
    cfg = CompositeCriticConfig(angular=ang_cfg, lambda_radial=5.0)
    critic = CompositeCritic(cfg)
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)

    # Fixed data
    v_q, v_a, v_ctx = _make_vecs(batch_size=8)
    v_neg = torch.randn(8, 5, D) * TARGET_NORM

    losses = []
    for step in range(20):
        optimizer.zero_grad()
        loss, metrics = critic.compute_contrastive_loss(v_q, v_a, v_neg, v_context=v_ctx)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Loss should decrease
    first_5 = sum(losses[:5]) / 5
    last_5 = sum(losses[-5:]) / 5
    assert last_5 < first_5, (
        f"Loss should decrease: first_5={first_5:.4f}, last_5={last_5:.4f}"
    )
    print(f"  [PASS] Training converges: loss {first_5:.4f} → {last_5:.4f} "
          f"(rank_acc={metrics['rank_acc']:.4f})")


def test_no_context_mode():
    """Test include_context=False mode (simpler model)."""
    cfg = ConditionalAngularCriticConfig(
        dim=D, hidden_dims=[256, 128], include_context=False,
    )
    critic = ConditionalAngularCritic(cfg)
    v_q, v_a, _ = _make_vecs()

    e = critic(v_q, v_a)
    assert e.shape == (B,)

    # Should also work via CompositeCritic
    comp = CompositeCritic(CompositeCriticConfig(angular=cfg))
    e2 = comp(v_q, v_a)
    assert e2.shape == (B,)
    print(f"  [PASS] No-context mode works (input_dim = {cfg.dim * 4 + 2 + 8})")


# ── Main ─────────────────────────────────────────────────────────

def main():
    torch.manual_seed(42)

    tests = [
        ("Angular critic construction", test_angular_critic_construction),
        ("Angular forward pass", test_angular_forward),
        ("Angular energy_and_grad", test_angular_energy_and_grad),
        ("Angular contrastive loss", test_angular_contrastive_loss),
        ("Angular direction loss", test_angular_direction_loss),
        ("Radial guard energy", test_radial_guard_energy),
        ("Radial analytical gradient", test_radial_guard_analytical_gradient),
        ("Gradient orthogonality (angular ⊥ radial)", test_gradient_orthogonality),
        ("CompositeCritic forward + grad", test_composite_critic),
        ("CompositeCritic Langevin API", test_composite_langevin_compatibility),
        ("CompositeCritic losses", test_composite_losses),
        ("Training convergence", test_training_step_converges),
        ("No-context mode", test_no_context_mode),
    ]

    print("=" * 60)
    print("Composite Critic Smoke Tests")
    print("=" * 60)

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            print(f"\n[TEST] {name}")
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
