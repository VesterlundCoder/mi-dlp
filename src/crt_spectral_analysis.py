#!/usr/bin/env python3
"""
2D CRT Fourier spectral analysis of hidden representations.

Re-indexes hidden activations by CRT coordinates (r_a, r_b) and computes
the 2D DFT. Partitions spectral energy into:
  E_0:  constant mode
  E_A:  A-only modes (k_a != 0, k_b = 0)
  E_B:  B-only modes (k_a = 0, k_b != 0)
  E_AB: cross-component modes (k_a != 0, k_b != 0)

Also computes spectral entropy, Gini coefficient, and participation ratio.

Usage:
  python crt_spectral_analysis.py --run-dir experiments/crt_mi/RAW_p113_s42 --output-dir analysis/spectrum
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch


def compute_2d_crt_spectrum(H: np.ndarray, a: int, b: int) -> Dict:
    """
    Given hidden activations H of shape (N_samples, d_model) where each sample
    corresponds to a known x value, re-index by CRT coordinates and compute
    the 2D DFT.

    Returns spectral energy partition and complexity metrics.
    """
    N = a * b
    d_model = H.shape[1]

    # We need to aggregate H by CRT coordinate (r_a, r_b)
    # Each sample has x, and r_a = x % a, r_b = x % b
    # Average H over all samples with the same (r_a, r_b)

    # H_crt[r_a, r_b, :] = mean of H[x] for all x where x%a=r_a and x%b=r_b
    # But we need x_vals for this. This function assumes H is already
    # organized or we pass x_vals separately.

    raise NotImplementedError("Use compute_spectrum_from_hidden instead")


def compute_spectrum_from_hidden(H: np.ndarray, x_vals: np.ndarray,
                                  a: int, b: int) -> Dict:
    """
    Compute 2D CRT Fourier spectrum for a single hidden dimension.

    H: (N_samples, d_model) hidden activations
    x_vals: (N_samples,) discrete log values
    a, b: CRT factors

    Returns per-dimension spectral metrics averaged over all dimensions.
    """
    N = a * b
    d_model = H.shape[1]

    # Aggregate by CRT coordinate
    H_crt = np.zeros((a, b, d_model))
    counts = np.zeros((a, b))

    for i in range(len(x_vals)):
        r_a = int(x_vals[i]) % a
        r_b = int(x_vals[i]) % b
        H_crt[r_a, r_b] += H[i]
        counts[r_a, r_b] += 1

    # Average
    for r_a in range(a):
        for r_b in range(b):
            if counts[r_a, r_b] > 0:
                H_crt[r_a, r_b] /= counts[r_a, r_b]

    # For each hidden dimension, compute 2D DFT and spectral energy
    all_E0 = []
    all_EA = []
    all_EB = []
    all_EAB = []
    all_entropy = []
    all_gini = []
    all_pr = []

    for d in range(d_model):
        h_2d = H_crt[:, :, d]  # (a, b)

        # 2D DFT
        H_hat = np.fft.fft2(h_2d)  # (a, b)
        power = np.abs(H_hat) ** 2

        total_power = power.sum()
        if total_power == 0:
            continue

        # Partition energy
        E0 = power[0, 0]
        EA = sum(power[ka, 0] for ka in range(1, a))
        EB = sum(power[0, kb] for kb in range(1, b))
        EAB = total_power - E0 - EA - EB

        all_E0.append(E0 / total_power)
        all_EA.append(EA / total_power)
        all_EB.append(EB / total_power)
        all_EAB.append(EAB / total_power)

        # Spectral entropy
        p = power.flatten() / total_power
        p = p[p > 0]
        entropy = -np.sum(p * np.log(p))
        all_entropy.append(entropy)

        # Gini coefficient
        sorted_p = np.sort(power.flatten())
        n = len(sorted_p)
        gini = (2 * np.sum((np.arange(1, n+1)) * sorted_p) / (n * sorted_p.sum()) - (n+1)/n)
        all_gini.append(gini)

        # Participation ratio
        lambdas = power.flatten()
        pr = (lambdas.sum() ** 2) / (lambdas ** 2).sum() if (lambdas ** 2).sum() > 0 else 0
        all_pr.append(pr)

    return {
        "E0": float(np.mean(all_E0)) if all_E0 else 0,
        "EA": float(np.mean(all_EA)) if all_EA else 0,
        "EB": float(np.mean(all_EB)) if all_EB else 0,
        "EAB": float(np.mean(all_EAB)) if all_EAB else 0,
        "spectral_entropy": float(np.mean(all_entropy)) if all_entropy else 0,
        "gini": float(np.mean(all_gini)) if all_gini else 0,
        "participation_ratio": float(np.mean(all_pr)) if all_pr else 0,
        "n_dims_analyzed": len(all_E0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="analysis/spectrum")
    parser.add_argument("--n-layers", type=int, default=2)
    args = parser.parse_args()

    config_path = os.path.join(args.run_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    crt_factors = tuple(config["crt_factors"])
    n_layers = config.get("n_layers", args.n_layers)
    a, b = crt_factors

    hidden_dir = os.path.join(args.run_dir, "hidden_states")
    if not os.path.exists(hidden_dir):
        print(f"Error: hidden_states directory not found: {hidden_dir}")
        sys.exit(1)

    x_val_files = [f for f in os.listdir(hidden_dir) if f.startswith("x_vals_epoch")]
    epochs = sorted([int(f.replace("x_vals_epoch", "").replace(".pt", "")) for f in x_val_files])
    print(f"Found {len(epochs)} checkpoints")
    print(f"CRT factors: ({a}, {b})")

    os.makedirs(args.output_dir, exist_ok=True)
    run_name = os.path.basename(args.run_dir)
    output_path = os.path.join(args.output_dir, f"{run_name}_spectrum.json")

    all_results = []
    for epoch in epochs:
        print(f"\nAnalyzing epoch {epoch}...")
        x_vals = torch.load(os.path.join(hidden_dir, f"x_vals_epoch{epoch}.pt")).numpy()

        epoch_result = {"epoch": epoch, "layers": {}}
        for l in range(n_layers):
            H = torch.load(os.path.join(hidden_dir, f"layer{l}_epoch{epoch}.pt")).numpy()
            spectrum = compute_spectrum_from_hidden(H, x_vals, a, b)
            epoch_result["layers"][l] = spectrum
            print(f"  Layer {l}: E_A={spectrum['EA']:.4f} E_B={spectrum['EB']:.4f} "
                  f"E_AB={spectrum['EAB']:.4f} entropy={spectrum['spectral_entropy']:.4f} "
                  f"PR={spectrum['participation_ratio']:.2f}")

        all_results.append(epoch_result)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSpectral analysis saved to {output_path}")


if __name__ == "__main__":
    main()
