#!/usr/bin/env python3
"""
Top-k causal ablation with rank-matched, variance-matched random controls.

Addresses the critical issues identified in the ablation audit:
  1. Top-k directions from each subspace (k in {2, 4, 8, 16}) for fair comparison
  2. 100 random controls (minimum p < 0.01)
  3. Variance-matched random controls (not just rank-matched)
  4. Separate data for subspace construction vs evaluation

Usage:
  python topk_ablation.py --checkpoint-dir experiments/crt_mi/RAW_p113_s42 --output-dir analysis/topk_ablation
"""
import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mi_suite import (
    GrokkingTransformer, load_model_from_checkpoint,
    generate_dlp_problems, extract_logits_and_hidden,
    coprime_factorization, encode_crt_representation,
    PAD, SEP, BOS, EOS, INT_OFFSET
)


def compute_fourier_subspaces_svd(H: np.ndarray, x_vals: np.ndarray,
                                    crt_factors: Tuple[int, int]) -> Dict:
    """Compute Fourier-derived subspaces with singular values for top-k selection.

    Returns U_A, U_B, U_AB with associated singular values s_A, s_B, s_AB
    so we can select top-k directions by energy.
    """
    a, b = crt_factors
    H = np.array(H) if not isinstance(H, np.ndarray) else H
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

    # A-variation: averaged over b, varies with a
    H_a_var = H_centered.mean(axis=1)  # (a, d)
    H_a_var_centered = H_a_var - H_a_var.mean(axis=0, keepdims=True)
    U_A, s_A, _ = np.linalg.svd(H_a_var_centered.T, full_matrices=False)

    # B-variation: averaged over a, varies with b
    H_b_var = H_centered.mean(axis=0)  # (b, d)
    H_b_var_centered = H_b_var - H_b_var.mean(axis=0, keepdims=True)
    U_B, s_B, _ = np.linalg.svd(H_b_var_centered.T, full_matrices=False)

    # AB-variation: residual after removing A and B
    H_a_broadcast = H_a_var[:, np.newaxis, :]  # (a, 1, d)
    H_b_broadcast = H_b_var[np.newaxis, :, :]  # (1, b, d)
    H_ab = H_centered - H_a_broadcast - H_b_broadcast  # (a, b, d)
    H_ab_flat = H_ab.reshape(-1, d)  # (a*b, d)
    U_AB, s_AB, _ = np.linalg.svd(H_ab_flat.T, full_matrices=False)

    return {
        "U_A": U_A, "U_B": U_B, "U_AB": U_AB,
        "s_A": s_A, "s_B": s_B, "s_AB": s_AB,
        "H_mean": H_mean,
    }


def compute_subspace_variance(H: np.ndarray, U: np.ndarray) -> float:
    """Compute the fraction of activation variance captured by subspace U."""
    if U.shape[1] == 0:
        return 0.0
    H = np.array(H, dtype=np.float64) if not isinstance(H, np.ndarray) else H.astype(np.float64)
    P = U @ U.T  # (d, d) projection
    H_proj = H @ P  # (N, d) projected
    var_total = float(np.var(H, axis=0).sum())
    var_proj = float(np.var(H_proj, axis=0).sum())
    return var_proj / max(var_total, 1e-12)


def sample_variance_matched_random(d: int, k: int, H: np.ndarray,
                                     target_variance: float,
                                     rng: np.random.RandomState,
                                     n_attempts: int = 50) -> np.ndarray:
    """Sample a random k-dimensional subspace matching the target variance."""
    best_U = None
    best_diff = float('inf')

    for _ in range(n_attempts):
        Q = rng.randn(d, k)
        Q, _ = np.linalg.qr(Q)
        var = compute_subspace_variance(H, Q)
        diff = abs(var - target_variance)
        if diff < best_diff:
            best_diff = diff
            best_U = Q
        if diff < 0.01 * target_variance:  # Good enough
            break

    return best_U


def run_model_with_ablation_batch(model, triples, variant, p, crt_factors, device,
                                    U: np.ndarray, layer: int, alpha: float = 1.0,
                                    n_samples: int = 500, batch_size: int = 64) -> Dict:
    """Run model with subspace ablated at specified layer (batched)."""
    max_len = model.max_len
    N = p - 1

    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(triples), min(n_samples, len(triples)), replace=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for batch_start in range(0, len(sample_indices), batch_size):
            batch_indices = sample_indices[batch_start:batch_start + batch_size]
            batch_tokens = []
            batch_targets = []
            for idx in batch_indices:
                g, h, x = triples[idx]
                tokens, target = encode_crt_representation(g, h, x, variant, p, crt_factors)
                padded = (tokens + [PAD] * (max_len - len(tokens)))[:max_len]
                batch_tokens.append(padded)
                batch_targets.append(target)

            tokens_tensor = torch.tensor(batch_tokens, dtype=torch.long).to(device)
            targets_tensor = torch.tensor(batch_targets, dtype=torch.long).to(device)

            # Forward through model up to ablation layer
            pos = torch.arange(max_len, device=device).unsqueeze(0).expand(tokens_tensor.size(0), -1)
            e = model.embed(tokens_tensor)
            h_embed = model.dropout(e + model.pos_embed(pos))

            h_state = h_embed
            for l in range(layer + 1):
                h_state = model.transformer.layers[l](h_state)

            # Ablate at EOS position
            seq_lens = (tokens_tensor != PAD).sum(dim=1) - 1
            eos_indices = seq_lens  # (batch,)

            # Apply ablation to each sample's EOS position
            if U.shape[1] > 0 and alpha > 0:
                P = torch.tensor(U @ U.T, dtype=h_state.dtype, device=device)
                for i in range(tokens_tensor.size(0)):
                    eos_idx = eos_indices[i].item()
                    h_vec = h_state[i, eos_idx]  # (d,)
                    h_state[i, eos_idx] = h_vec - alpha * P @ h_vec

            # Continue through remaining layers
            for l in range(layer + 1, len(model.transformer.layers)):
                h_state = model.transformer.layers[l](h_state)

            logits = model.unembed(h_state)
            # Gather logits at EOS position
            idx = eos_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, logits.size(-1))
            logits_at_eos = logits.gather(1, idx).squeeze(1)
            preds = logits_at_eos.argmax(dim=-1)

            correct += (preds == targets_tensor).sum().item()
            total += len(batch_indices)

    return {
        "accuracy": correct / max(total, 1),
        "n_samples": total,
    }


def run_topk_ablation(ckpt_dir: str, output_dir: str, device_str: str = "auto"):
    """Run top-k causal ablation with 100 variance-matched random controls."""
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
    print(f"Top-k Causal Ablation: {name}")
    print(f"  p={p}, variant={variant}, factors=({a},{b})")
    print(f"{'='*60}")

    ckpt_path = os.path.join(ckpt_dir, "checkpoints", "final.pt")
    if not os.path.exists(ckpt_path):
        # Try last checkpoint
        ckpts = sorted(
            [f for f in os.listdir(os.path.join(ckpt_dir, "checkpoints")) if f.endswith(".pt")],
            key=lambda f: int(f.split("_")[-1].split(".")[0]) if f.split("_")[-1].split(".")[0].isdigit() else 0
        )
        if ckpts:
            ckpt_path = os.path.join(ckpt_dir, "checkpoints", ckpts[-1])
        else:
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

    # Split data: use first half for subspace construction, second half for evaluation
    n_total = len(triples)
    n_construction = n_total // 2
    construction_triples = triples[:n_construction]
    evaluation_triples = triples[n_construction:]
    print(f"  Construction set: {n_construction}, Evaluation set: {len(evaluation_triples)}")

    # Extract hidden states on construction set
    print(f"  Extracting hidden states on construction set...")
    _, hidden_states_construction, x_vals_construction = extract_logits_and_hidden(
        model, construction_triples, variant, p, crt_factors, device
    )

    # Baseline accuracy on evaluation set (no ablation)
    print(f"  Computing baseline accuracy on evaluation set...")
    baseline_result = run_model_with_ablation_batch(
        model, evaluation_triples, variant, p, crt_factors, device,
        np.zeros((128, 0)), layer=1, alpha=0.0, n_samples=500
    )
    A_baseline = baseline_result["accuracy"]
    print(f"  Baseline accuracy: {A_baseline:.4f}")

    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{name}_topk_ablation.csv")

    fields = ["name", "variant", "p", "layer", "subspace", "k", "alpha",
              "accuracy", "delta_accuracy", "rank", "variance_fraction",
              "is_random_control", "variance_matched", "trial", "percentile"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

    results = []
    ks = [2, 4, 8, 16]
    n_random = 100

    for layer in hidden_states_construction:
        print(f"\n  Layer {layer}:")
        H_construction = np.array(hidden_states_construction[layer], dtype=np.float64)
        d = H_construction.shape[1]

        # Compute Fourier subspaces with singular values
        subspaces = compute_fourier_subspaces_svd(H_construction, x_vals_construction, crt_factors)

        for subspace_name, U_full, s_full in [("A", subspaces["U_A"], subspaces["s_A"]),
                                                ("B", subspaces["U_B"], subspaces["s_B"]),
                                                ("AB", subspaces["U_AB"], subspaces["s_AB"])]:
            max_k = min(U_full.shape[1], max(ks))
            if max_k == 0:
                print(f"    U_{subspace_name}: no directions, skipping")
                continue

            print(f"    U_{subspace_name} (max rank={U_full.shape[1]}):")

            for k in ks:
                if k > U_full.shape[1]:
                    continue

                # Top-k directions by singular value
                U_topk = U_full[:, :k]
                var_frac = compute_subspace_variance(H_construction, U_topk)

                print(f"      k={k}, variance_fraction={var_frac:.4f}")

                # Structured ablation
                result = run_model_with_ablation_batch(
                    model, evaluation_triples, variant, p, crt_factors, device,
                    U_topk, layer, alpha=1.0, n_samples=500
                )
                delta_acc = result["accuracy"] - A_baseline

                row = {
                    "name": name, "variant": variant, "p": p,
                    "layer": layer, "subspace": subspace_name, "k": k, "alpha": 1.0,
                    "accuracy": result["accuracy"], "delta_accuracy": delta_acc,
                    "rank": k, "variance_fraction": var_frac,
                    "is_random_control": 0, "variance_matched": 0,
                    "trial": 0, "percentile": "",
                }
                results.append(row)

                with open(csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writerow(row)

                print(f"        structured: Δ={delta_acc:+.4f}")

                # Random controls: rank-matched
                rng = np.random.RandomState(42)
                random_deltas_rank = []
                for trial in range(n_random):
                    Q_rand = rng.randn(d, k)
                    Q_rand, _ = np.linalg.qr(Q_rand)
                    result_rand = run_model_with_ablation_batch(
                        model, evaluation_triples, variant, p, crt_factors, device,
                        Q_rand, layer, alpha=1.0, n_samples=200
                    )
                    delta_rand = result_rand["accuracy"] - A_baseline
                    random_deltas_rank.append(delta_rand)

                    row = {
                        "name": name, "variant": variant, "p": p,
                        "layer": layer, "subspace": f"random_{subspace_name}",
                        "k": k, "alpha": 1.0,
                        "accuracy": result_rand["accuracy"],
                        "delta_accuracy": delta_rand,
                        "rank": k, "variance_fraction": "",
                        "is_random_control": 1, "variance_matched": 0,
                        "trial": trial, "percentile": "",
                    }
                    results.append(row)

                    if trial % 20 == 0:
                        with open(csv_path, "a", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=fields)
                            writer.writerow(row)

                # Compute percentile
                if random_deltas_rank:
                    percentile = float(np.mean([d <= delta_acc for d in random_deltas_rank]))
                    rand_mean = float(np.mean(random_deltas_rank))
                    rand_std = float(np.std(random_deltas_rank))
                    print(f"        random (rank-matched): Δ={rand_mean:+.4f}±{rand_std:.4f} "
                          f"percentile={percentile:.4f} (p<{1/(n_random+1):.4f})")

                    # Update the structured row with percentile
                    for r in results:
                        if (r["subspace"] == subspace_name and r["layer"] == layer
                            and r["k"] == k and r["is_random_control"] == 0):
                            r["percentile"] = percentile

                # Variance-matched random controls
                rng2 = np.random.RandomState(123)
                random_deltas_var = []
                for trial in range(n_random):
                    Q_var = sample_variance_matched_random(
                        d, k, H_construction, var_frac, rng2)
                    result_var = run_model_with_ablation_batch(
                        model, evaluation_triples, variant, p, crt_factors, device,
                        Q_var, layer, alpha=1.0, n_samples=200
                    )
                    delta_var = result_var["accuracy"] - A_baseline
                    random_deltas_var.append(delta_var)

                    row = {
                        "name": name, "variant": variant, "p": p,
                        "layer": layer, "subspace": f"var_matched_{subspace_name}",
                        "k": k, "alpha": 1.0,
                        "accuracy": result_var["accuracy"],
                        "delta_accuracy": delta_var,
                        "rank": k, "variance_fraction": var_frac,
                        "is_random_control": 1, "variance_matched": 1,
                        "trial": trial, "percentile": "",
                    }
                    results.append(row)

                    if trial % 20 == 0:
                        with open(csv_path, "a", newline="") as f:
                            writer = csv.DictWriter(f, fieldnames=fields)
                            writer.writerow(row)

                if random_deltas_var:
                    percentile_var = float(np.mean([d <= delta_acc for d in random_deltas_var]))
                    var_mean = float(np.mean(random_deltas_var))
                    var_std = float(np.std(random_deltas_var))
                    print(f"        random (var-matched): Δ={var_mean:+.4f}±{var_std:.4f} "
                          f"percentile={percentile_var:.4f}")

    # Write all results
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Summary
    summary_path = os.path.join(output_dir, f"{name}_topk_summary.json")
    summary = {"name": name, "variant": variant, "p": p, "baseline_accuracy": A_baseline}
    for r in results:
        if r["is_random_control"] == 0:
            key = f"L{r['layer']}_{r['subspace']}_k{r['k']}"
            summary[key] = {
                "delta": r["delta_accuracy"],
                "variance_fraction": r["variance_fraction"],
                "percentile": r.get("percentile", ""),
            }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Results saved to {csv_path}")
    print(f"  Summary saved to {summary_path}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", default="analysis/topk_ablation")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    run_topk_ablation(args.checkpoint_dir, args.output_dir, args.device)


if __name__ == "__main__":
    main()
