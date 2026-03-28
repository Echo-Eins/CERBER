"""
Honest training test: 2 epochs on synthetic SONAR-like data.
Uses the REAL Stage 1.5 architecture with Cayley parametrization + adaptive sigma.
No mocks — actual forward/backward, actual Langevin inference.
"""

import sys
import os
import math
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from cebcm.models.energy import SimpleEnergy
from cebcm.models.actor import LatentDenoiseActor
from cebcm.inference.langevin import run_langevin
from cebcm.inference.sigma_schedule import AdaptiveSigmaEnergyWrapper, SigmaScheduleConfig


def generate_sonar_like_data(n_samples: int, dim: int = 1024, target_norm: float = 0.2051):
    """Generate synthetic data on the SONAR sphere."""
    vecs = torch.randn(n_samples, dim)
    vecs = F.normalize(vecs, dim=-1) * target_norm
    return vecs


def main():
    device = "cpu"
    dim = 128  # Reduced dim for speed (real: 1024)
    hidden = [256, 128, 64]
    target_norm = 0.2051
    batch_size = 16
    n_samples = 128
    n_epochs = 2
    lr_critic = 1e-4
    lr_actor = 5e-5
    sigma_min, sigma_max = 0.001, 1.0
    langevin_steps = 10
    langevin_lr = 0.001
    noise_scale = 0.0002

    print("=" * 60)
    print("CERBER Stage 1.5 Training Test (Cayley + Adaptive Sigma)")
    print("=" * 60)

    # 1. Create models with Cayley parametrization
    print("\n[1] Creating models...")
    critic = SimpleEnergy(
        dim=dim, hidden_dims=hidden, norm_mode="orthonorm",
        activation="groupsort", ortho_n_iters=0,
    ).to(device)
    actor = LatentDenoiseActor(
        dim=dim, hidden_dims=hidden, norm_mode="spectral_norm",
        activation="silu",
    ).to(device)

    # Verify Cayley parametrization is active
    has_cayley = any("parametrizations" in name for name, _ in critic.named_parameters())
    print(f"  Critic: {sum(p.numel() for p in critic.parameters())} params, Cayley={has_cayley}")
    print(f"  Actor:  {sum(p.numel() for p in actor.parameters())} params")

    # Check orthogonality of first layer
    first_layer = list(critic.net.children())[0]
    W = first_layer.weight
    min_d = min(W.shape)
    if W.shape[0] <= W.shape[1]:
        ortho_err = (W @ W.t() - torch.eye(W.shape[0])).norm().item()
    else:
        ortho_err = (W.t() @ W - torch.eye(W.shape[1])).norm().item()
    print(f"  Critic L1 orthogonality error: {ortho_err:.2e}")

    # 2. Generate synthetic data
    print("\n[2] Generating synthetic SONAR data...")
    data = generate_sonar_like_data(n_samples, dim, target_norm)
    print(f"  {n_samples} vectors, dim={dim}, norm={data.norm(dim=-1).mean():.4f}")

    # 3. Setup optimizers
    opt_c = torch.optim.AdamW(critic.parameters(), lr=lr_critic, weight_decay=0.01)
    opt_a = torch.optim.AdamW(actor.parameters(), lr=lr_actor, weight_decay=0.01)

    # 4. Training loop
    print("\n[3] Training...")
    sigma_sched_cfg = SigmaScheduleConfig(
        enabled=True, mode="hybrid", sigma_max=0.3, sigma_min=0.01, adaptive_blend=0.5,
    )

    for epoch in range(n_epochs):
        critic.train(); actor.train()
        epoch_loss_c = 0.0; epoch_loss_a = 0.0
        n_batches = 0

        perm = torch.randperm(n_samples)
        for i in range(0, n_samples - batch_size + 1, batch_size):
            idx = perm[i:i + batch_size]
            clean = data[idx].to(device)

            # Sample query (retrieval positive = shifted clean for synthetic)
            query = clean + torch.randn_like(clean) * 0.02
            query = F.normalize(query, dim=-1) * target_norm

            # Add noise to create noisy candidates
            sigma_val = torch.exp(
                torch.empty(batch_size, 1).uniform_(math.log(sigma_min), math.log(sigma_max))
            )
            noise = torch.randn_like(clean) * sigma_val * clean.norm(dim=-1, keepdim=True)
            noisy = clean + noise
            noisy = F.normalize(noisy, dim=-1) * target_norm

            # --- Critic loss: MDSM (simplified) ---
            noisy_req = noisy.detach().requires_grad_(True)
            e_noisy = critic(query, noisy_req, sigma=sigma_val)
            grad_e = torch.autograd.grad(
                e_noisy.sum(), noisy_req, create_graph=True
            )[0]

            # DSM target: direction from noisy → clean
            target_dir = F.normalize(clean - noisy, dim=-1)
            grad_dir = F.normalize(grad_e, dim=-1)

            # Directional loss: cosine between -grad_E and (clean - noisy)
            loss_dir = (1.0 - F.cosine_similarity(-grad_dir, target_dir, dim=-1)).mean()

            # Ranking: E(clean) < E(noisy)
            e_clean = critic(query, clean, sigma=sigma_val)
            e_noisy_val = critic(query, noisy, sigma=sigma_val)
            rank_loss = F.relu(e_clean - e_noisy_val + 0.5).mean()

            loss_c = loss_dir + 0.25 * rank_loss

            # Energy regularization
            l_energy_reg = (e_clean ** 2).mean()
            loss_c = loss_c + 0.01 * l_energy_reg

            opt_c.zero_grad()
            loss_c.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            opt_c.step()

            # --- Actor loss ---
            with torch.no_grad():
                v_actor, _ = actor.predict_step(
                    query, noisy, sigma=sigma_val,
                    step_size=1.0, target_norm=target_norm,
                    tangent_projection=True,
                )

            e_actor = critic(query, v_actor, sigma=sigma_val)
            loss_a = e_actor.mean()  # minimize energy of actor output

            opt_a.zero_grad()
            loss_a.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            opt_a.step()

            epoch_loss_c += loss_c.item()
            epoch_loss_a += loss_a.item()
            n_batches += 1

        avg_lc = epoch_loss_c / max(n_batches, 1)
        avg_la = epoch_loss_a / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/{n_epochs}: critic_loss={avg_lc:.4f} actor_loss={avg_la:.4f}")

    # 5. Evaluate with Langevin (adaptive sigma)
    print("\n[4] Evaluating with Langevin + Adaptive Sigma...")
    critic.eval(); actor.eval()

    eval_batch = 8
    eval_idx = torch.randperm(n_samples)[:eval_batch]
    clean_eval = data[eval_idx].to(device)
    query_eval = clean_eval + torch.randn_like(clean_eval) * 0.02
    query_eval = F.normalize(query_eval, dim=-1) * target_norm

    # Add noise
    sigma_eval = torch.full((eval_batch, 1), 0.2)
    noise_eval = torch.randn_like(clean_eval) * sigma_eval * clean_eval.norm(dim=-1, keepdim=True)
    noisy_eval = F.normalize(clean_eval + noise_eval, dim=-1) * target_norm

    cos_before = F.cosine_similarity(noisy_eval, clean_eval, dim=-1).mean().item()
    print(f"  Cosine before: {cos_before:.4f}")

    # Actor step
    with torch.no_grad():
        v_after_actor, _ = actor.predict_step(
            query_eval, noisy_eval, sigma=sigma_eval,
            step_size=1.0, target_norm=target_norm, tangent_projection=True,
        )
    cos_actor = F.cosine_similarity(v_after_actor, clean_eval, dim=-1).mean().item()
    print(f"  Cosine after actor: {cos_actor:.4f} (delta={cos_actor - cos_before:+.4f})")

    # Langevin with adaptive sigma
    wrapper = AdaptiveSigmaEnergyWrapper(critic, sigma_sched_cfg, max_steps=langevin_steps)
    result = run_langevin(
        method="pid",
        energy_fn=wrapper,
        v_query=query_eval,
        v_init=v_after_actor,
        lr=langevin_lr,
        noise_scale=noise_scale,
        max_steps=langevin_steps,
        target_norm=target_norm,
        tangent_noise=True,
        plateau_patience=langevin_steps + 1,
        kp=0.1, ki=0.01, kd=0.05,
    )

    cos_after = F.cosine_similarity(result.v_final, clean_eval, dim=-1).mean().item()
    print(f"  Cosine after Langevin ({result.num_steps} steps): {cos_after:.4f} (delta={cos_after - cos_before:+.4f})")

    # Energy trajectory
    if result.trajectory:
        print(f"  Energy: {result.trajectory[0]:.4f} → {result.trajectory[-1]:.4f}")

    # 6. Final orthogonality check after training
    print("\n[5] Post-training orthogonality check...")
    first_layer = list(critic.net.children())[0]
    W = first_layer.weight
    if W.shape[0] <= W.shape[1]:
        ortho_err = (W @ W.t() - torch.eye(W.shape[0])).norm().item()
    else:
        ortho_err = (W.t() @ W - torch.eye(W.shape[1])).norm().item()
    print(f"  Critic L1 orthogonality error after training: {ortho_err:.2e}")

    # Verify all weights are finite
    all_finite = all(torch.isfinite(p).all() for p in critic.parameters())
    print(f"  All critic weights finite: {all_finite}")
    all_finite_a = all(torch.isfinite(p).all() for p in actor.parameters())
    print(f"  All actor weights finite: {all_finite_a}")

    print("\n" + "=" * 60)
    print("TRAINING TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
