import torch
from cebcm.models.chain_generator import ChainGenerator
from cebcm.models.composite_critic import CompositeCritic
from cebcm.models.chain_head import ChainHead
from cerber_gui.chain_generator_diagnostics import (
    load_chain_generator_from_checkpoint,
    load_composite_critic_from_checkpoint,
    _state,
    run_generation
)

def test_inference():
    print("Testing Active Inference...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check if checkpoints exist, otherwise we can't test
    gen_path = "experiments/13_chain_generator/output/checkpoints/best.pt"
    critic_path = "experiments/12_composite_critic/output/checkpoints/best.pt"
    
    import os
    if not os.path.exists(gen_path):
        print(f"Skipping test: Generator not found at {gen_path}")
        return
        
    print("Loading generator...")
    gen, info_gen = load_chain_generator_from_checkpoint(gen_path, device)
    _state.generator = gen
    _state.device = device
    
    if os.path.exists(critic_path):
        print("Loading critic...")
        crit, info_crit = load_composite_critic_from_checkpoint(critic_path, device)
        _state.critic = crit
    else:
        print(f"No critic at {critic_path}, skipping critic-dependent tests.")
        return

    v_q = torch.randn(1024, device=device)
    v_target = torch.randn(1024, device=device)
    v_q = v_q / v_q.norm()
    v_target = v_target / v_target.norm()
    
    print("\n--- Testing Greedy (Beam 1, Cands 1) ---")
    res1 = run_generation(v_q, num_steps=3, v_target=v_target, beam_width=1, num_candidates=1)
    print(f"Greedy steps: {res1.num_steps}, chain shape: {res1.v_chain.shape}")
    
    print("\n--- Testing Best-of-N Rejection Sampling (Beam 1, Cands 4) ---")
    res2 = run_generation(v_q, num_steps=3, v_target=v_target, beam_width=1, num_candidates=4, noise_std=0.02)
    print(f"Rejection steps: {res2.num_steps}, chain shape: {res2.v_chain.shape}")
    
    print("\n--- Testing Energy-Guided Beam Search (Beam 4, Cands 4) ---")
    res3 = run_generation(v_q, num_steps=3, v_target=v_target, beam_width=4, num_candidates=4, noise_std=0.02)
    print(f"Beam steps: {res3.num_steps}, chain shape: {res3.v_chain.shape}")
    
    print("\nAll tests passed!")

if __name__ == "__main__":
    test_inference()
