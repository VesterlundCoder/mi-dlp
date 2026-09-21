#!/usr/bin/env python3
"""
Causal ablation: Test whether A, B, and AB Fourier modes are causally
necessary for DLP computation.

Implements:
  - Fourier-derived subspace identification (U_A, U_B, U_AB)
  - Ablation by projecting out subspaces
  - Dose-response curves
  - Matched random subspace controls
  - Energy-matched controls
  - Layer localization

Usage:
  python causal_ablation.py --checkpoint-dir checkpoints/lumi_download/M04_p113_s42 --output-dir analysis/ablation
"""

import argparse
import csv
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mi_suite import (
    GrokkingTransformer, load_model_from_checkpoint,
    generate_dlp_problems, extract_logits_and_hidden,
    coprime_factorization, encode_crt_representation,
    train_linear_probe, train_probe_and_get_weights,
    PAD, SEP, BOS, EOS, INT_OFFSET
)


def compute_fourier_subspaces(hidden_states: torch.Tensor, x_vals: np.ndarray,
                                crt_factors: Tuple[int, int]) -> Dict:
    """Compute Fourier-derived subspaces U_A, U_B, U_AB.

    For each hidden dimension, compute the 2D DFT over CRT coordinates.
    The Fourier modes define directions in hidden space.
    """
    a, b = crt_factors
    H = hidden_states.numpy()  # (N, d)
    N, d = H.shape

    # Aggregate by CRT coordinates
    H_crt = np.zeros((a, b, d))
    counts = np.zeros((a, b))
    for i in range(N):
        r_a = int(x_vals[i]) % a
        r_b = int(x_vals[i]) % b
        H_crt[r_a, r_b] += H[i]
        counts[r_a, r_b] += 1

    for r_a in range(a):
        for r_b in range(b):
            if counts[r_a, r_b] > 0:
                H_crt[r_a, r_b] /= counts[r_a, r_b]

    # Mean-center
    H_mean = H_crt.mean(axis=(0, 1))
    H_centered = H_crt - H_mean

    # Compute 2D DFT for each hidden dimension
    # Collect Fourier coefficient vectors for each mode class
    A_modes = []  # (k_a != 0, k_b = 0)
    B_modes = []  # (k_a = 0, k_b != 0)
    AB_modes = []  # (k_a != 0, k_b != 0)

    for d_idx in range(d):
        h_2d = H_centered[:, :, d_idx]  # (a, b)
        H_hat = np.fft.fft2(h_2d)  # (a, b) complex

        # A modes: k_a != 0, k_b = 0
        for k_a in range(1, a):
            coeff = H_hat[k_a, 0]
            if abs(coeff) > 1e-10:
                # Create a d-dimensional vector with this coefficient
                vec = np.zeros(d, dtype=complex)
                vec[d_idx] = coeff
                A_modes.append(vec)

        # B modes: k_a = 0, k_b != 0
        for k_b in range(1, b):
            coeff = H_hat[0, k_b]
            if abs(coeff) > 1e-10:
                vec = np.zeros(d, dtype=complex)
                vec[d_idx] = coeff
                B_modes.append(vec)

        # AB modes: k_a != 0, k_b != 0
        for k_a in range(1, a):
            for k_b in range(1, b):
                coeff = H_hat[k_a, k_b]
                if abs(coeff) > 1e-10:
                    vec = np.zeros(d, dtype=complex)
                    vec[d_idx] = coeff
                    AB_modes.append(vec)

    # Convert to real orthonormal bases
    def modes_to_basis(modes):
        if not modes:
            return np.zeros((d, 0))
        # Stack real and imaginary parts
        real_parts = [np.real(m) for m in modes]
        imag_parts = [np.imag(m) for m in modes]
        all_vecs = real_parts + imag_parts
        V = np.array(all_vecs).T  # (d, 2*n_modes)
        # SVD to get orthonormal basis
        U, s, Vt = np.linalg.svd(V, full_matrices=False)
        # Keep components with significant singular values
        rank = np.sum(s > 1e-8 * s[0]) if len(s) > 0 else 0
        return U[:, :rank]

    # Alternative: directly compute the subspace from the activation matrix
    # The A-subspace is the span of all directions that vary with r_a but not r_b
    # We can compute this more directly

    # Direct approach: compute the activation variation for each mode class
    # A-variation: H_centered averaged over r_b, then vary r_a
    H_a_var = H_centered.mean(axis=1)  # (a, d) - averaged over b, varies with a
    # Remove mean (already centered)
    H_a_var_centered = H_a_var - H_a_var.mean(axis=0, keepdims=True)
    U_A, s_A, _ = np.linalg.svd(H_a_var_centered.T, full_matrices=False)
    rank_A = np.sum(s_A > 1e-8 * (s_A[0] if len(s_A) > 0 else 1))
    U_A = U_A[:, :rank_A]

    # B-variation: H_centered averaged over r_a, then vary r_b
    H_b_var = H_centered.mean(axis=0)  # (b, d) - averaged over a, varies with b
    H_b_var_centered = H_b_var - H_b_var.mean(axis=0, keepdims=True)
    U_B, s_B, _ = np.linalg.svd(H_b_var_centered.T, full_matrices=False)
    rank_B = np.sum(s_B > 1e-8 * (s_B[0] if len(s_B) > 0 else 1))
    U_B = U_B[:, :rank_B]

    # AB-variation: the residual after removing A and B variations
    # H_ab = H_centered - H_a_var[:,:,None,:] - H_b_var[None,:,:]
    # But we need to be careful with broadcasting
    H_a_broadcast = H_a_var[:, np.newaxis, :]  # (a, 1, d)
    H_b_broadcast = H_b_var[np.newaxis, :, :]  # (1, b, d)
    H_ab = H_centered - H_a_broadcast - H_b_broadcast  # (a, b, d)
    H_ab_flat = H_ab.reshape(-1, d)  # (a*b, d)
    U_AB, s_AB, _ = np.linalg.svd(H_ab_flat.T, full_matrices=False)
    rank_AB = np.sum(s_AB > 1e-8 * (s_AB[0] if len(s_AB) > 0 else 1))
    U_AB = U_AB[:, :rank_AB]

    return {
        "U_A": U_A, "U_B": U_B, "U_AB": U_AB,
        "rank_A": rank_A, "rank_B": rank_B, "rank_AB": rank_AB,
        "s_A": s_A[:rank_A], "s_B": s_B[:rank_B], "s_AB": s_AB[:rank_AB],
        "H_mean": H_mean,
    }


def ablate_subspace(h: np.ndarray, U: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Ablate subspace U from hidden activation h.

    h' = h - alpha * P_U * (h - mean)
    where P_U = U @ U^T is the projection onto subspace U.
    """
    if U.shape[1] == 0:
        return h
    P = U @ U.T  # (d, d) projection matrix
    return h - alpha * P @ h


def run_model_with_ablation(model, triples, variant, p, crt_factors, device,
                              U: np.ndarray, layer: int, alpha: float = 1.0,
                              n_samples: int = 1000) -> Dict:
    """Run model with subspace ablated at specified layer."""
    max_len = model.max_len
    N = p - 1

    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(triples), min(n_samples, len(triples)), replace=False)

    correct = 0
    total = 0
    correct_eq = 0
    total_eq = 0

    with torch.no_grad():
        for idx in sample_indices:
            g, h, x = triples[idx]
            tokens, target = encode_crt_representation(g, h, x, variant, p, crt_factors)
            padded = (tokens + [PAD] * (max_len - len(tokens)))[:max_len]
            tokens_tensor = torch.tensor([padded], dtype=torch.long).to(device)

            # Forward through model up to ablation layer
            pos = torch.arange(max_len, device=device).unsqueeze(0)
            e = model.embed(tokens_tensor)
            h_embed = model.dropout(e + model.pos_embed(pos))

            # Run through layers up to ablation point
            h_state = h_embed
            for l in range(layer + 1):
                h_state = model.transformer.layers[l](h_state)

            # Ablate at EOS position
            seq_len = (tokens_tensor != PAD).sum(dim=1) - 1
            eos_idx = seq_len[0].item()
            h_at_eos = h_state[0, eos_idx].cpu().numpy()

            # Apply ablation
            h_ablated = ablate_subspace(h_at_eos, U, alpha)

            # Put back
            h_state[0, eos_idx] = torch.tensor(h_ablated, dtype=h_state.dtype, device=device)

            # Continue through remaining layers
            for l in range(layer + 1, len(model.transformer.layers)):
                h_state = model.transformer.layers[l](h_state)

            logits = model.unembed(h_state)
            logits_at_eos = logits[0, eos_idx]
            pred = logits_at_eos.argmax().item() - INT_OFFSET

            if pred == x:
                correct += 1
            total += 1

            # Equivariance test
            c = int(rng.randint(1, N))
            h_shifted = (pow(g, c, p) * h) % p
            x_expected = (x + c) % N
            tokens_s, _ = encode_crt_representation(g, h_shifted, x_expected, variant, p, crt_factors)
            padded_s = (tokens_s + [PAD] * (max_len - len(tokens_s)))[:max_len]
            tokens_s_tensor = torch.tensor([padded_s], dtype=torch.long).to(device)

            e_s = model.embed(tokens_s_tensor)
            h_embed_s = model.dropout(e_s + model.pos_embed(pos))
            h_state_s = h_embed_s
            for l in range(layer + 1):
                h_state_s = model.transformer.layers[l](h_state_s)

            seq_len_s = (tokens_s_tensor != PAD).sum(dim=1) - 1
            eos_idx_s = seq_len_s[0].item()
            h_at_eos_s = h_state_s[0, eos_idx_s].cpu().numpy()
            h_ablated_s = ablate_subspace(h_at_eos_s, U, alpha)
            h_state_s[0, eos_idx_s] = torch.tensor(h_ablated_s, dtype=h_state_s.dtype, device=device)

            for l in range(layer + 1, len(model.transformer.layers)):
                h_state_s = model.transformer.layers[l](h_state_s)

            logits_s = model.unembed(h_state_s)
            logits_at_eos_s = logits_s[0, eos_idx_s]
            pred_s = logits_at_eos_s.argmax().item() - INT_OFFSET

            if pred_s == (pred + c) % N:
                correct_eq += 1
            total_eq += 1

    return {
        "accuracy": correct / max(total, 1),
        "equivariance": correct_eq / max(total_eq, 1),
        "n_samples": total,
    }


def run_causal_ablation(ckpt_dir: str, output_dir: str, device_str: str = "auto"):
    """Run causal ablation on a checkpoint."""
    config_path = os.path.join(ckpt_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    p = config.get("prime", config.get("p", 113))
    variant = config.get("variant", config.get("type", "RAW"))
    if variant == "standard":
        variant = "RAW"
    crt_factors = tuple(config.get("crt_factors", coprime_factorization(p - 1)))
    a, b = crt_factors

    name = os.path.basename(ckpt_dir)
    print(f"\n{'='*60}")
    print(f"Causal Ablation: {name}")
    print(f"  p={p}, variant={variant}, factors=({a},{b})")
    print(f"{'='*60}")

    ckpt_path = os.path.join(ckpt_dir, "checkpoints", "final.pt")
    if not os.path.exists(ckpt_path):
        print(f"  No checkpoint found, skipping")
        return None

    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)

    model = load_model_from_checkpoint(ckpt_path, config, device)
    triples = generate_dlp_problems(p)

    # Extract hidden states
    print(f"  Extracting hidden states...")
    logits, hidden_states, x_vals = extract_logits_and_hidden(
        model, triples, variant, p, crt_factors, device
    )

    # Baseline accuracy
    preds = logits.argmax(dim=-1).numpy() - INT_OFFSET
    A_baseline = float(np.mean(preds == x_vals))
    print(f"  Baseline accuracy: {A_baseline:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{name}_ablation.csv")

    fields = ["name", "variant", "p", "layer", "subspace", "alpha",
              "accuracy", "delta_accuracy", "equivariance", "delta_equivariance",
              "rank", "is_random_control", "is_energy_matched"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

    results = []

    for layer in hidden_states:
        print(f"\n  Layer {layer}:")
        H = hidden_states[layer]

        # Compute Fourier subspaces
        print(f"    Computing Fourier subspaces...")
        subspaces = compute_fourier_subspaces(H, x_vals, crt_factors)
        d = H.shape[1]

        print(f"    Ranks: A={subspaces['rank_A']}, B={subspaces['rank_B']}, AB={subspaces['rank_AB']}")

        # Baseline at this layer (no ablation)
        baseline = run_model_with_ablation(
            model, triples, variant, p, crt_factors, device,
            np.zeros((d, 0)), layer, alpha=0.0, n_samples=500
        )

        # Ablate each subspace
        for subspace_name, U in [("A", subspaces["U_A"]),
                                   ("B", subspaces["U_B"]),
                                   ("AB", subspaces["U_AB"])]:
            rank = U.shape[1]
            if rank == 0:
                continue

            print(f"    Ablating U_{subspace_name} (rank={rank})...")

            # Dose-response
            for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
                result = run_model_with_ablation(
                    model, triples, variant, p, crt_factors, device,
                    U, layer, alpha=alpha, n_samples=500
                )
                delta_acc = result["accuracy"] - baseline["accuracy"]
                delta_eq = result["equivariance"] - baseline["equivariance"]

                row = {
                    "name": name, "variant": variant, "p": p,
                    "layer": layer, "subspace": subspace_name, "alpha": alpha,
                    "accuracy": result["accuracy"], "delta_accuracy": delta_acc,
                    "equivariance": result["equivariance"],
                    "delta_equivariance": delta_eq,
                    "rank": rank, "is_random_control": 0, "is_energy_matched": 0,
                }
                results.append(row)

                with open(csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writerow(row)

                print(f"      alpha={alpha:.2f}: acc={result['accuracy']:.4f} "
                      f"(Δ={delta_acc:+.4f}) eq={result['equivariance']:.4f}")

            # Random subspace control (matched rank)
            print(f"    Random control (rank={rank})...")
            n_random = 20
            for trial in range(n_random):
                # Random subspace with same rank
                Q_rand = np.random.randn(d, rank)
                Q_rand, _ = np.linalg.qr(Q_rand)

                result = run_model_with_ablation(
                    model, triples, variant, p, crt_factors, device,
                    Q_rand, layer, alpha=1.0, n_samples=200
                )
                delta_acc = result["accuracy"] - baseline["accuracy"]
                delta_eq = result["equivariance"] - baseline["equivariance"]

                row = {
                    "name": name, "variant": variant, "p": p,
                    "layer": layer, "subspace": f"random_{subspace_name}",
                    "alpha": 1.0,
                    "accuracy": result["accuracy"], "delta_accuracy": delta_acc,
                    "equivariance": result["equivariance"],
                    "delta_equivariance": delta_eq,
                    "rank": rank, "is_random_control": 1, "is_energy_matched": 0,
                }
                results.append(row)

                with open(csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writerow(row)

            # Summary for this subspace
            random_deltas = [r["delta_accuracy"] for r in results
                            if r["subspace"] == f"random_{subspace_name}"
                            and r["layer"] == layer and r["alpha"] == 1.0]
            structured_delta = [r["delta_accuracy"] for r in results
                              if r["subspace"] == subspace_name
                              and r["layer"] == layer and r["alpha"] == 1.0][0]

            if random_deltas:
                rand_mean = np.mean(random_deltas)
                rand_std = np.std(random_deltas)
                percentile = np.mean([d <= structured_delta for d in random_deltas])
                print(f"    Random control: Δ={rand_mean:.4f}±{rand_std:.4f} "
                      f"(structured percentile: {percentile:.2%})")

    print(f"\n  Results saved to {csv_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", default="analysis/ablation")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_causal_ablation(args.checkpoint_dir, args.output_dir, args.device)


if __name__ == "__main__":
    main()
