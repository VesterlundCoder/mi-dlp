#!/usr/bin/env python3
"""
CKA (Centered Kernel Alignment) representation similarity between RAW and CRT models.

Compares hidden representations across checkpoints to test whether:
  R_RAW(late) ≈ R_CRT(early)

Usage:
  python cka_analysis.py --raw-dir experiments/crt_mi/RAW_p113_s42 --crt-dir experiments/crt_mi/RAW+CRT-BOTH_p113_s42 --output-dir analysis/cka
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute linear CKA between two representation matrices.
    X: (N, d1), Y: (N, d2). Returns CKA in [0, 1]."""
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)

    X_norm = np.linalg.norm(X, 'fro')
    Y_norm = np.linalg.norm(Y, 'fro')

    if X_norm == 0 or Y_norm == 0:
        return 0.0

    # ||X^T Y||_F^2 / (||X||_F^2 * ||Y||_F^2)
    cross = np.linalg.norm(X.T @ Y, 'fro') ** 2
    return float(cross / (X_norm ** 2 * Y_norm ** 2))


def rbf_cka(X: np.ndarray, Y: np.ndarray, sigma: float = None) -> float:
    """Compute RBF CKA."""
    from sklearn.metrics.pairwise import rbf_kernel
    if sigma is None:
        sigma = np.median(np.linalg.norm(X - X[:, None], axis=-1))

    K_X = rbf_kernel(X, gamma=1.0 / (2 * sigma ** 2))
    K_Y = rbf_kernel(Y, gamma=1.0 / (2 * sigma ** 2))

    # Center
    H = np.eye(len(X)) - np.ones((len(X), len(X))) / len(X)
    K_Xc = H @ K_X @ H
    K_Yc = H @ K_Y @ H

    num = np.sum(K_Xc * K_Yc)
    den = np.sqrt(np.sum(K_Xc * K_Xc) * np.sum(K_Yc * K_Yc))
    return float(num / den) if den > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--crt-dir", required=True)
    parser.add_argument("--output-dir", default="analysis/cka")
    parser.add_argument("--n-layers", type=int, default=2)
    args = parser.parse_args()

    # Load configs
    with open(os.path.join(args.raw_dir, "config.json")) as f:
        raw_config = json.load(f)
    with open(os.path.join(args.crt_dir, "config.json")) as f:
        crt_config = json.load(f)

    n_layers = raw_config.get("n_layers", args.n_layers)

    raw_hidden_dir = os.path.join(args.raw_dir, "hidden_states")
    crt_hidden_dir = os.path.join(args.crt_dir, "hidden_states")

    if not os.path.exists(raw_hidden_dir) or not os.path.exists(crt_hidden_dir):
        print("Error: hidden_states directories not found")
        sys.exit(1)

    # Find common epochs
    raw_epochs = set()
    for f in os.listdir(raw_hidden_dir):
        if f.startswith("x_vals_epoch"):
            raw_epochs.add(int(f.replace("x_vals_epoch", "").replace(".pt", "")))

    crt_epochs = set()
    for f in os.listdir(crt_hidden_dir):
        if f.startswith("x_vals_epoch"):
            crt_epochs.add(int(f.replace("x_vals_epoch", "").replace(".pt", "")))

    raw_ep_list = sorted(raw_epochs)
    crt_ep_list = sorted(crt_epochs)
    print(f"RAW epochs: {len(raw_ep_list)}, CRT epochs: {len(crt_ep_list)}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Compute CKA matrix for each layer
    for l in range(n_layers):
        cka_matrix = np.zeros((len(raw_ep_list), len(crt_ep_list)))

        for i, raw_ep in enumerate(raw_ep_list):
            H_raw = torch.load(os.path.join(raw_hidden_dir, f"layer{l}_epoch{raw_ep}.pt")).numpy()
            for j, crt_ep in enumerate(crt_ep_list):
                H_crt = torch.load(os.path.join(crt_hidden_dir, f"layer{l}_epoch{crt_ep}.pt")).numpy()

                # Need same samples - use x_vals to align
                x_raw = torch.load(os.path.join(raw_hidden_dir, f"x_vals_epoch{raw_ep}.pt")).numpy()
                x_crt = torch.load(os.path.join(crt_hidden_dir, f"x_vals_epoch{crt_ep}.pt")).numpy()

                # Align by x value
                raw_idx = {v: i for i, v in enumerate(x_raw)}
                crt_idx = {v: i for i, v in enumerate(x_crt)}
                common = sorted(set(x_raw) & set(x_crt))

                if len(common) < 10:
                    continue

                raw_indices = [raw_idx[v] for v in common]
                crt_indices = [crt_idx[v] for v in common]

                H_raw_aligned = H_raw[raw_indices]
                H_crt_aligned = H_crt[crt_indices]

                cka_matrix[i, j] = linear_cka(H_raw_aligned, H_crt_aligned)

        # Save
        np.save(os.path.join(args.output_dir, f"cka_layer{l}.npy"), cka_matrix)

        # Save metadata
        meta = {
            "raw_epochs": raw_ep_list,
            "crt_epochs": crt_ep_list,
            "layer": l,
        }
        with open(os.path.join(args.output_dir, f"cka_layer{l}_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        # Find the ridge: for each RAW epoch, find the CRT epoch with max CKA
        best_matches = []
        for i, raw_ep in enumerate(raw_ep_list):
            j = np.argmax(cka_matrix[i])
            best_matches.append({
                "raw_epoch": raw_ep,
                "best_crt_epoch": crt_ep_list[j],
                "cka": float(cka_matrix[i, j]),
            })

        print(f"\nLayer {l} - Best CRT matches for RAW checkpoints:")
        for m in best_matches[::max(1, len(best_matches)//10)]:
            print(f"  RAW ep {m['raw_epoch']:6d} -> CRT ep {m['best_crt_epoch']:6d} (CKA={m['cka']:.4f})")

    print(f"\nCKA matrices saved to {args.output_dir}")


if __name__ == "__main__":
    main()
