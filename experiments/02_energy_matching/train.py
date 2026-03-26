"""
Stage 2: Energy Matching â€” Train UnconditionalEnergy on SONAR embeddings.

Pipeline:
    SONAR encoder (frozen) â†’ pre-encoded .pt â†’ [UnconditionalEnergy] â†’ EM training
                                                  ^^^TRAINS^^^

Three training modes:
    1. energy_matching  â€” Pure EM loss from scratch
    2. nce_warmstart_em â€” NCE warmstart (10 epochs) â†’ EM fine-tune
    3. cosine_em        â€” Cosine direction EM (for 1-Lipschitz networks)

Mathematical basis:
    Energy Matching (Balcerak et al., NeurIPS 2025) trains E_Î¸(x) such that
    -âˆ‡_x E_Î¸(x_t) â‰ˆ u_t, where u_t = xâ‚ - xâ‚€ is the OT velocity field.
    x_t = (1-t)Â·xâ‚€ + tÂ·xâ‚ interpolates between prior (xâ‚€) and data (xâ‚).

    OT loss itself is simulation-free, but this script may run auxiliary
    sampling in NCE warmstart / negative-buffer refresh modes.
    The energy landscape is trained to encode transport + equilibrium behavior.

Usage:
    # NCE warmstart â†’ Energy Matching (recommended):
    python experiments/02_energy_matching/train.py \\
        --data data/wikitext_sonar_10k.pt --mode nce_warmstart_em

    # Pure Energy Matching:
    python experiments/02_energy_matching/train.py \\
        --data data/wikitext_sonar_10k.pt --mode energy_matching

    # Cosine EM (for 1-Lipschitz with orthonorm):
    python experiments/02_energy_matching/train.py \\
        --data data/wikitext_sonar_10k.pt --mode cosine_em

    # Ablation: spectral norm instead of orthonorm:
    python experiments/02_energy_matching/train.py \\
        --data data/wikitext_sonar_10k.pt --norm spectral_norm

Spec reference: Â§10.5, tasks/todo.md
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from configs.energy_matching import EnergyMatchingConfig
from cebcm.models.energy_unconditional import UnconditionalEnergy
from cebcm.training.energy_matching import (
    energy_matching_loss,
    energy_matching_cosine,
    energy_matching_weighted,
    generate_samples_ode,
)
from cebcm.training.negative_buffer import (
    NegativeBuffer,
    nce_loss,
    nce_loss_simple,
)
from cebcm.training.kill_criteria import summarize_unconditional_eval
from cebcm.data.dataset import SONARVectorDataset
from cerber_gui.sota_eval import (
    SOTAEvalConfig,
    compute_distribution_suite,
    compute_manifold_knn_metrics,
)


# ============================================================
# Training epoch functions
# ============================================================

def train_epoch_em(
    model: UnconditionalEnergy,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: EnergyMatchingConfig,
    device: torch.device,
    epoch: int,
    global_step: int,
    loss_fn: str = "cosine_em",
) -> tuple[dict, int]:
    """Train one epoch with Energy Matching loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, v_clean in enumerate(dataloader):
        v_clean = v_clean.to(device)

        # Warmup
        if global_step < config.warmup_steps:
            warmup_factor = max(global_step / config.warmup_steps, 1e-6)
            for pg in optimizer.param_groups:
                pg["lr"] = config.lr * warmup_factor

        # Select loss function
        if loss_fn == "energy_matching":
            loss = energy_matching_loss(
                energy_fn=model,
                x_data=v_clean,
                prior_std=config.prior_std,
                t_min=config.t_min,
                t_max=config.t_max,
            )
        elif loss_fn == "cosine_em":
            loss = energy_matching_cosine(
                energy_fn=model,
                x_data=v_clean,
                prior_std=config.prior_std,
                t_min=config.t_min,
                t_max=config.t_max,
                magnitude_weight=config.magnitude_weight,
            )
        elif loss_fn == "weighted_em":
            loss = energy_matching_weighted(
                energy_fn=model,
                x_data=v_clean,
                prior_std=config.prior_std,
                t_min=config.t_min,
                t_max=config.t_max,
                near_data_weight=config.near_data_weight,
            )
        else:
            raise ValueError(f"Unknown loss: {loss_fn}")

        optimizer.zero_grad()
        loss.backward()

        for p in model.parameters():
            if p.grad is not None:
                torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0, out=p.grad)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Skip non-finite OR outlier spikes (finite but huge losses that
        # push the model in wrong directions despite gradient clipping)
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm) or loss.item() > 100.0:
            optimizer.zero_grad(set_to_none=True)
            if batch_idx < 5 or (config.log_every > 0 and (batch_idx + 1) % config.log_every == 0):
                print(f"  [WARN] bad EM loss={loss.item():.4g} at batch {batch_idx + 1}; step skipped")
            global_step += 1
            continue

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        if config.log_every > 0 and (batch_idx + 1) % config.log_every == 0:
            e_scale = model.log_energy_scale.exp().item()
            print(
                f"  [{batch_idx + 1}/{len(dataloader)}] "
                f"loss={loss.item():.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f} "
                f"E_scale={e_scale:.2f}"
            )

    return {
        "loss": total_loss / max(num_batches, 1),
    }, global_step


def train_epoch_nce(
    model: UnconditionalEnergy,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    config: EnergyMatchingConfig,
    neg_buffer: NegativeBuffer,
    device: torch.device,
    epoch: int,
    global_step: int,
) -> tuple[dict, int]:
    """Train one epoch with NCE loss using negative buffer."""
    model.train()
    total_loss = 0.0
    total_e_data = 0.0
    total_e_noise = 0.0
    num_batches = 0

    for batch_idx, v_clean in enumerate(dataloader):
        v_clean = v_clean.to(device)
        B = v_clean.shape[0]

        # Warmup
        if global_step < config.warmup_steps:
            warmup_factor = max(global_step / config.warmup_steps, 1e-6)
            for pg in optimizer.param_groups:
                pg["lr"] = config.lr * warmup_factor

        # Sample negatives from buffer
        v_noise = neg_buffer.sample(B, device)

        # NCE loss
        if config.nce_loss_type == "full":
            loss = nce_loss(model, v_clean, v_noise, prior_std=config.prior_std)
        else:
            loss = nce_loss_simple(model, v_clean, v_noise)

        optimizer.zero_grad()
        loss.backward()

        # Sanitize gradients: replace inf/nan with 0 so a single
        # exploding element doesn't corrupt model parameters forever.
        for p in model.parameters():
            if p.grad is not None:
                torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0, out=p.grad)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Skip non-finite OR outlier spikes
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm) or loss.item() > 100.0:
            optimizer.zero_grad(set_to_none=True)
            if batch_idx < 5 or (config.log_every > 0 and (batch_idx + 1) % config.log_every == 0):
                print(f"  [WARN] bad NCE loss={loss.item():.4g} at batch {batch_idx + 1}; step skipped")
            global_step += 1
            continue

        optimizer.step()

        # Track energies
        with torch.no_grad():
            e_data = model(v_clean).mean().item()
            e_noise = model(v_noise).mean().item()

        total_loss += loss.item()
        total_e_data += e_data
        total_e_noise += e_noise
        num_batches += 1
        global_step += 1

        if config.log_every > 0 and (batch_idx + 1) % config.log_every == 0:
            print(
                f"  [{batch_idx + 1}/{len(dataloader)}] "
                f"NCE_loss={loss.item():.4f} "
                f"E_data={e_data:.4f} E_noise={e_noise:.4f} "
                f"gap={e_noise - e_data:.4f}"
            )

    # Refresh buffer after each epoch
    neg_buffer.refresh(model, device)

    return {
        "loss": total_loss / max(num_batches, 1),
        "e_data_mean": total_e_data / max(num_batches, 1),
        "e_noise_mean": total_e_noise / max(num_batches, 1),
        "energy_gap": (total_e_noise - total_e_data) / max(num_batches, 1),
    }, global_step


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate_energy_matching(
    model: UnconditionalEnergy,
    dataset: SONARVectorDataset,
    config: EnergyMatchingConfig,
    device: torch.device,
    num_samples: int = 100,
) -> dict:
    """
    Strict SOTA-aligned evaluation for unconditional energy models.

    Primary:
    - energy descent under perturbation,
    - distribution/manifold quality.

    Secondary diagnostics:
    - paired cosine/geodesic movement relative to clean.
    """
    model.eval()
    results: dict[str, dict | float] = {}

    # --- 1) Energy statistics on real data ---
    indices = torch.randperm(len(dataset))[:num_samples]
    v_clean = torch.stack([dataset[i.item()] for i in indices]).to(device)

    energies = model(v_clean)
    results["energy_data_mean"] = energies.mean().item()
    results["energy_data_std"] = energies.std().item()
    results["energy_data_min"] = energies.min().item()
    results["energy_data_max"] = energies.max().item()

    noisy_bank: list[torch.Tensor] = []
    refined_bank: list[torch.Tensor] = []

    # --- 2) Local refinement diagnostics by noise scale ---
    for noise_scale in config.eval_noise_scales:
        norms = v_clean.norm(dim=-1, keepdim=True)
        v_noisy = v_clean + torch.randn_like(v_clean) * noise_scale * norms

        cos_before = F.cosine_similarity(v_clean, v_noisy, dim=-1)
        geo_before = torch.acos(cos_before.clamp(min=-1.0, max=1.0))
        l2_before = torch.norm(v_clean - v_noisy, dim=-1)
        energy_before = model(v_noisy).detach()

        # Deterministic gradient following for local field diagnostics.
        v_current = v_noisy.clone()
        for _ in range(50):
            v_grad = v_current.detach().requires_grad_(True)
            with torch.enable_grad():
                energy = model(v_grad)
                grad = torch.autograd.grad(energy.sum(), v_grad)[0]
            v_current = v_current - 0.01 * grad
            if config.target_norm is not None:
                v_current = F.normalize(v_current, dim=-1) * config.target_norm

        cos_after = F.cosine_similarity(v_clean, v_current, dim=-1)
        geo_after = torch.acos(cos_after.clamp(min=-1.0, max=1.0))
        l2_after = torch.norm(v_clean - v_current, dim=-1)
        energy_after = model(v_current).detach()
        step_norm = torch.norm(v_current - v_noisy, dim=-1)

        noisy_bank.append(v_noisy.detach())
        refined_bank.append(v_current.detach())

        results[f"noise_{noise_scale}"] = {
            "cos_before_mean": cos_before.mean().item(),
            "cos_after_mean": cos_after.mean().item(),
            "improvement": (cos_after - cos_before).mean().item(),
            "success_rate": (cos_after > cos_before).float().mean().item(),
            "geodesic_before_mean": geo_before.mean().item(),
            "geodesic_after_mean": geo_after.mean().item(),
            "geodesic_improvement": (geo_before - geo_after).mean().item(),
            "l2_before_mean": l2_before.mean().item(),
            "l2_after_mean": l2_after.mean().item(),
            "l2_improvement": (l2_before - l2_after).mean().item(),
            "energy_before_mean": energy_before.mean().item(),
            "energy_after_mean": energy_after.mean().item(),
            "energy_improvement": (energy_before - energy_after).mean().item(),
            "energy_success_rate": (energy_after < energy_before).float().mean().item(),
            "step_norm_mean": step_norm.mean().item(),
        }

    # --- 3) Distribution / manifold quality ---
    n_gen = min(config.eval_generate_samples, 512)
    samples = generate_samples_ode(
        energy_fn=model,
        num_samples=n_gen,
        dim=config.energy_dim,
        num_steps=config.eval_ode_steps,
        prior_std=config.prior_std,
        target_norm=config.target_norm,
        device=device,
    )

    sample_norms = samples.norm(dim=-1)
    data_norms = v_clean.norm(dim=-1)
    samples_norm = F.normalize(samples, dim=-1)
    pairwise_cos = (samples_norm @ samples_norm.T).fill_diagonal_(0)
    n = samples_norm.shape[0]

    data_norm = F.normalize(v_clean, dim=-1)
    data_pairwise_cos = (data_norm @ data_norm.T).fill_diagonal_(0)
    results["samples"] = {
        "norm_mean": sample_norms.mean().item(),
        "norm_std": sample_norms.std().item(),
        "data_norm_mean": data_norms.mean().item(),
        "pairwise_cos_mean": pairwise_cos.sum().item() / max(1, n * (n - 1)),
        "data_pairwise_cos_mean": data_pairwise_cos.sum().item() / max(1, num_samples * (num_samples - 1)),
        "energy_mean": model(samples).mean().item(),
        "energy_std": model(samples).std().item(),
    }

    ref_bank_size = min(len(dataset), max(512, num_samples * 4))
    ref_indices = torch.randperm(len(dataset))[:ref_bank_size]
    ref_bank = torch.stack([dataset[i.item()] for i in ref_indices]).to(device)

    real = ref_bank[: min(ref_bank.shape[0], samples.shape[0])]
    fake = samples[: real.shape[0]]

    dist_cfg = SOTAEvalConfig()
    distribution = compute_distribution_suite(real, fake, dist_cfg)

    if noisy_bank and refined_bank:
        noisy_diag = torch.cat(noisy_bank, dim=0)
        refined_diag = torch.cat(refined_bank, dim=0)
        knn = compute_manifold_knn_metrics(
            ref_bank=ref_bank,
            noisy=noisy_diag,
            denoised=refined_diag,
            k=dist_cfg.manifold_k,
        )
    else:
        knn = {}

    distribution.update(knn)
    results["distribution"] = distribution

    noise_keys = [k for k in results.keys() if k.startswith("noise_")]
    if noise_keys:
        denom = float(len(noise_keys))
        results["summary"] = {
            "mean_cos_improvement": sum(results[k]["improvement"] for k in noise_keys) / denom,
            "mean_cos_success_rate": sum(results[k]["success_rate"] for k in noise_keys) / denom,
            "mean_geodesic_improvement": sum(results[k]["geodesic_improvement"] for k in noise_keys) / denom,
            "mean_l2_improvement": sum(results[k]["l2_improvement"] for k in noise_keys) / denom,
            "mean_energy_improvement": sum(results[k]["energy_improvement"] for k in noise_keys) / denom,
            "mean_energy_success_rate": sum(results[k]["energy_success_rate"] for k in noise_keys) / denom,
            "mean_step_norm": sum(results[k]["step_norm_mean"] for k in noise_keys) / denom,
        }

    return results

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Energy Matching training")
    parser.add_argument("--data", type=str, required=True, help="Path to .pt dataset")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--mode", type=str, default=None,
                        choices=["energy_matching", "nce_warmstart_em", "cosine_em", "weighted_em"],
                        help="Training mode")
    parser.add_argument("--norm", type=str, default=None,
                        choices=["orthonorm", "spectral_norm", "none"])
    parser.add_argument("--activation", type=str, default=None,
                        choices=["groupsort", "lipschitz_spline", "relu"])
    parser.add_argument("--no_nce_warmstart", action="store_true",
                        help="Disable NCE warmstart even in nce_warmstart_em mode")
    parser.add_argument("--ed", type=str, default=None,
                        help="Eval denoising frequency: 'none' to disable, or integer N for every N epochs (default: config.eval_every_epoch)")
    args = parser.parse_args()

    config = EnergyMatchingConfig()
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.mode is not None:
        if args.mode == "nce_warmstart_em":
            config.nce_warmstart = True
            config.loss_type = "cosine_em"
        elif args.mode in ("energy_matching", "cosine_em", "weighted_em"):
            config.loss_type = args.mode
            config.nce_warmstart = False
        else:
            config.loss_type = args.mode
    if args.norm is not None:
        config.norm_mode = args.norm
    if args.activation is not None:
        config.activation = args.activation
    if args.no_nce_warmstart:
        config.nce_warmstart = False
    config.use_wandb = args.wandb

    # --ed flag: eval denoising frequency
    if args.ed is not None:
        if args.ed.lower() == "none":
            config.eval_every_epoch = 0  # 0 = disabled
        else:
            config.eval_every_epoch = int(args.ed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # â”€â”€ Data â”€â”€
    print(f"Loading dataset from {args.data}...")
    full_dataset = SONARVectorDataset(args.data)
    print(f"  Total vectors: {len(full_dataset)}, dim: {full_dataset.embeddings.shape[1]}")

    n_test = min(config.num_test_sentences, len(full_dataset) // 10)
    n_train = len(full_dataset) - n_test
    train_dataset = full_dataset.subset(0, n_train)
    test_dataset = full_dataset.subset(n_train, len(full_dataset))
    print(f"  Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True
    )

    # â”€â”€ Model â”€â”€
    model = UnconditionalEnergy(
        dim=config.energy_dim,
        hidden_dims=config.energy_hidden_dims,
        norm_mode=config.norm_mode,
        activation=config.activation,
        ortho_n_iters=config.ortho_n_iters,
        groupsort_size=config.groupsort_size,
        spline_num_knots=config.spline_num_knots,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"UnconditionalEnergy: {num_params:,} parameters")
    print(f"  Architecture: norm={config.norm_mode}, activation={config.activation}")
    print(f"  Hidden dims: {config.energy_hidden_dims}")
    print(f"  Loss: {config.loss_type}")
    print(f"  NCE warmstart: {config.nce_warmstart} ({config.nce_epochs} epochs)")
    print(f"  Prior: N(0, {config.prior_std:.5f}Â²I), target_norm={config.target_norm}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=config.lr * 0.1
    )

    start_epoch = 0
    global_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}...")
        ckpt = torch.load(args.resume, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt.get("global_step", 0)
        print(f"  Resumed at epoch {start_epoch}")

    # â”€â”€ Negative Buffer â”€â”€
    neg_buffer = NegativeBuffer(
        buffer_size=config.buffer_size,
        dim=config.energy_dim,
        init_std=config.prior_std,
        refresh_fraction=config.buffer_refresh_fraction,
        langevin_steps=config.buffer_langevin_steps,
        langevin_lr=config.buffer_langevin_lr,
        langevin_noise=config.buffer_langevin_noise,
        target_norm=config.target_norm,
    )
    # Seed buffer with noisy data (much better than random init)
    neg_buffer.seed_from_data(train_dataset.embeddings, noise_scale=config.buffer_seed_noise)
    print(f"  Negative buffer: {config.buffer_size} vectors, seeded from data")

    # â”€â”€ WandB â”€â”€
    wandb_run = None
    if config.use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=config.wandb_project,
            config={
                "lr": config.lr,
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "loss_type": config.loss_type,
                "norm_mode": config.norm_mode,
                "activation": config.activation,
                "nce_warmstart": config.nce_warmstart,
                "nce_epochs": config.nce_epochs,
                "prior_std": config.prior_std,
                "target_norm": config.target_norm,
                "num_params": num_params,
            },
        )

    # â”€â”€ Output dirs â”€â”€
    output_dir = Path(config.output_dir)
    checkpoint_dir = Path(config.checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # â”€â”€ Training loop â”€â”€
    print(f"\n{'='*60}")
    print(f"Energy Matching Training â€” {config.num_epochs} epochs")
    if config.nce_warmstart:
        print(f"  Phase 1: NCE warmstart ({config.nce_epochs} epochs)")
        print(f"  Phase 2: {config.loss_type} ({config.num_epochs - config.nce_epochs} epochs)")
    else:
        print(f"  Loss: {config.loss_type}")
    print(f"  Batch size: {config.batch_size}, LR: {config.lr}")
    print(f"  t range: [{config.t_min}, {config.t_max}]")
    print(f"{'='*60}\n")

    all_metrics: list[dict] = []
    best_eval_score = -float("inf")
    start_time = time.time()

    for epoch in range(start_epoch, config.num_epochs):
        epoch_start = time.time()

        # Determine training mode for this epoch
        in_nce_phase = config.nce_warmstart and epoch < config.nce_epochs

        if in_nce_phase:
            phase_str = f"NCE [{epoch + 1}/{config.nce_epochs}]"
            print(f"Epoch {epoch + 1}/{config.num_epochs} ({phase_str})")
            train_metrics, global_step = train_epoch_nce(
                model, train_loader, optimizer, config,
                neg_buffer, device, epoch, global_step,
            )
        else:
            em_epoch = epoch - config.nce_epochs if config.nce_warmstart else epoch
            phase_str = f"EM [{em_epoch + 1}]"
            print(f"Epoch {epoch + 1}/{config.num_epochs} ({phase_str})")
            train_metrics, global_step = train_epoch_em(
                model, train_loader, optimizer, config,
                device, epoch, global_step,
                loss_fn=config.loss_type,
            )

        # Step scheduler
        if global_step >= config.warmup_steps:
            scheduler.step()

        epoch_time = time.time() - epoch_start
        metrics_str = " ".join(f"{k}={v:.4f}" for k, v in train_metrics.items())
        print(f"  {metrics_str} lr={optimizer.param_groups[0]['lr']:.6f} ({epoch_time:.1f}s)")

        # â”€â”€ Evaluate â”€â”€
        eval_metrics = None
        if config.eval_every_epoch > 0 and (
            (epoch + 1) % config.eval_every_epoch == 0 or epoch == config.num_epochs - 1
        ):
            print("  Evaluating...")
            eval_metrics = evaluate_energy_matching(
                model, test_dataset, config, device,
                num_samples=config.eval_num_samples,
            )

            # Print local diagnostics
            for key, val in eval_metrics.items():
                if key.startswith("noise_"):
                    print(
                        f"    {key}: "
                        f"energy_improvement={val['energy_improvement']:+.4f}, "
                        f"energy_success={val['energy_success_rate']:.0%}, "
                        f"d_cos={val['improvement']:+.4f}, "
                        f"d_geo={val['geodesic_improvement']:+.4f}, "
                        f"d_l2={val['l2_improvement']:+.4f}, "
                        f"step={val['step_norm_mean']:.6f}"
                    )

            # Print sample quality
            if "samples" in eval_metrics:
                s = eval_metrics["samples"]
                print(
                    f"    samples: norm={s['norm_mean']:.4f}Â±{s['norm_std']:.4f} "
                    f"(data={s['data_norm_mean']:.4f}), "
                    f"pairwise_cos={s['pairwise_cos_mean']:.4f} "
                    f"(data={s['data_pairwise_cos_mean']:.4f})"
                )
            if "distribution" in eval_metrics:
                d = eval_metrics["distribution"]
                print(
                    f"    distribution: mmd={d.get('mmd_rbf', float('nan')):.6f}, "
                    f"c2st={d.get('c2st_acc', float('nan')):.4f}, "
                    f"prdc(P/C)={d.get('prdc_precision', float('nan')):.4f}/"
                    f"{d.get('prdc_coverage', float('nan')):.4f}, "
                    f"knn_l2_impr={d.get('knn_l2_improvement', float('nan')):+.4e}"
                )

            kill_report = summarize_unconditional_eval(eval_metrics)
            eval_metrics["kill_criteria"] = kill_report
            print(
                f"    composite_score={kill_report['score']:+.6f}, "
                f"noise_pass_rate={kill_report['aggregate'].get('noise_pass_rate', float('nan')):.0%}, "
                f"strict_pass={kill_report['passed']}"
            )

            if kill_report["score"] > best_eval_score:
                best_eval_score = kill_report["score"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "epoch": epoch,
                        "global_step": global_step,
                        "metrics": eval_metrics,
                        "config": {
                            "energy_dim": config.energy_dim,
                            "energy_hidden_dims": config.energy_hidden_dims,
                            "norm_mode": config.norm_mode,
                            "activation": config.activation,
                            "loss_type": config.loss_type,
                        },
                    },
                    checkpoint_dir / "best.pt",
                )
                print(f"    * New best (composite score: {best_eval_score:+.6f})")

        # Log
        epoch_record = {"epoch": epoch + 1, "train": train_metrics, "phase": "nce" if in_nce_phase else "em"}
        if eval_metrics is not None:
            epoch_record["eval"] = eval_metrics
        all_metrics.append(epoch_record)

        if wandb_run is not None:
            log_dict = {f"train/{k}": v for k, v in train_metrics.items()}
            log_dict["train/lr"] = optimizer.param_groups[0]["lr"]
            log_dict["train/phase"] = 0 if in_nce_phase else 1
            if eval_metrics is not None:
                for key, val in eval_metrics.items():
                    if isinstance(val, dict):
                        for mk, mv in val.items():
                            if isinstance(mv, (int, float, bool)):
                                log_dict[f"eval/{key}/{mk}"] = float(mv)
                    else:
                        log_dict[f"eval/{key}"] = val
            wandb_run.log(log_dict, step=epoch + 1)

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                },
                checkpoint_dir / f"epoch_{epoch + 1:03d}.pt",
            )

    # â”€â”€ Final evaluation â”€â”€
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")

    print("\nFinal evaluation...")
    model.eval()
    final_eval = evaluate_energy_matching(
        model, test_dataset, config, device,
        num_samples=min(200, len(test_dataset)),
    )
    final_kill_report = summarize_unconditional_eval(final_eval)
    final_eval["kill_criteria"] = final_kill_report

    for key, val in final_eval.items():
        if key.startswith("noise_"):
            print(
                f"  {key}: "
                f"energy_improvement={val['energy_improvement']:+.4f}, "
                f"energy_success={val['energy_success_rate']:.0%}, "
                f"d_cos={val['improvement']:+.4f}, "
                f"d_geo={val['geodesic_improvement']:+.4f}, "
                f"d_l2={val['l2_improvement']:+.4f}, "
                f"step={val['step_norm_mean']:.6f}"
            )

    # Save final
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "epoch": config.num_epochs - 1,
            "global_step": global_step,
            "final_eval": final_eval,
            "config": {
                "energy_dim": config.energy_dim,
                "energy_hidden_dims": config.energy_hidden_dims,
                "norm_mode": config.norm_mode,
                "activation": config.activation,
                "loss_type": config.loss_type,
            },
        },
        checkpoint_dir / "final.pt",
    )

    # Save metrics
    summary = {
        "total_time_seconds": total_time,
        "num_epochs": config.num_epochs,
        "best_eval_score": best_eval_score,
        "best_improvement": best_eval_score,
        "final_eval": final_eval,
        "config": {
            "lr": config.lr,
            "batch_size": config.batch_size,
            "loss_type": config.loss_type,
            "norm_mode": config.norm_mode,
            "activation": config.activation,
            "nce_warmstart": config.nce_warmstart,
            "nce_epochs": config.nce_epochs,
            "prior_std": config.prior_std,
            "target_norm": config.target_norm,
        },
        "epoch_metrics": all_metrics,
    }
    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nMetrics saved to {output_dir / 'training_metrics.json'}")

    # â”€â”€ Strict kill criterion â”€â”€
    print(f"\n{'='*60}")
    print("STRICT KILL CRITERION CHECK (SOTA)")
    print(f"{'='*60}")
    for key, block in final_kill_report["per_noise"].items():
        status = "PASS" if block["passed"] else "FAIL"
        m = block["metrics"]
        print(
            f"  {key}: "
            f"energy_improvement={m['energy_improvement']:+.4f}, "
            f"energy_success={m['energy_success_rate']:.0%}, "
            f"step={m['step_norm_mean']:.6f}, "
            f"cos_diag={m['cos_improvement']:+.4f} -> {status}"
        )

    print("  Global gates:")
    for gate_name, gate_ok in final_kill_report["global_gates"].items():
        print(f"    - {gate_name}: {'PASS' if gate_ok else 'FAIL'}")
    print("  Distribution gates:")
    for gate_name, gate_ok in final_kill_report["distribution_gates"].items():
        print(f"    - {gate_name}: {'PASS' if gate_ok else 'FAIL'}")
    print(
        f"  Composite score: {final_kill_report['score']:+.6f} "
        f"(noise_pass_rate={final_kill_report['aggregate'].get('noise_pass_rate', float('nan')):.0%})"
    )

    if final_kill_report["passed"]:
        print("\n  VERDICT: Energy Matching PASSED (strict) - unconditional quality gates satisfied")
    else:
        print("\n  VERDICT: Energy Matching FAILED (strict) - review objective/regularization/sampler")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()


