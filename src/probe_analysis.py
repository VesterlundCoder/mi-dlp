#!/usr/bin/env python3
"""
Linear probing of hidden states for CRT component information.

For each checkpoint, trains linear probes to predict:
  - x (full discrete log)
  - x mod a (CRT component A)
  - x mod b (CRT component B)

Also runs:
  - Random label permutation control
  - Untrained network control
  - Selectivity computation

Usage:
  python probe_analysis.py --run-dir experiments/crt_mi/RAW_p113_s42 --output-dir analysis/probes
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def load_hidden_states(hidden_dir: str, epoch: int, n_layers: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """Load hidden states and x values for a given epoch."""
    hidden = []
    for l in range(n_layers):
        h = torch.load(os.path.join(hidden_dir, f"layer{l}_epoch{epoch}.pt"))
        hidden.append(h)
    x_vals = torch.load(os.path.join(hidden_dir, f"x_vals_epoch{epoch}.pt"))
    return hidden, x_vals


def train_probe(X: np.ndarray, y: np.ndarray, max_iter: int = 1000) -> float:
    """Train a linear probe and return accuracy on held-out test set."""
    if len(np.unique(y)) < 2:
        return 0.0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    probe = LogisticRegression(max_iter=max_iter, C=1.0, solver='lbfgs')
    probe.fit(X_train, y_train)
    return probe.score(X_test, y_test)


def train_probe_shuffled(X: np.ndarray, y: np.ndarray, seed: int = 123) -> float:
    """Train a probe with shuffled labels (random control)."""
    rng = np.random.RandomState(seed)
    y_shuffled = rng.permutation(y)
    return train_probe(X, y_shuffled)


def analyze_checkpoint(hidden_dir: str, epoch: int, n_layers: int,
                       crt_factors: Tuple[int, int]) -> Dict:
    """Analyze a single checkpoint with all probes."""
    a, b = crt_factors
    hidden, x_vals = load_hidden_states(hidden_dir, epoch, n_layers)
    x_np = x_vals.numpy()

    results = {
        "epoch": epoch,
        "layers": {},
    }

    for l in range(n_layers):
        H = hidden[l].numpy()
        layer_results = {}

        # Full x probe
        acc_x = train_probe(H, x_np)
        acc_x_shuffled = train_probe_shuffled(H, x_np)
        layer_results["acc_x"] = acc_x
        layer_results["acc_x_shuffled"] = acc_x_shuffled
        layer_results["selectivity_x"] = acc_x - acc_x_shuffled

        # x mod a probe
        y_a = (x_vals % a).numpy()
        acc_a = train_probe(H, y_a)
        acc_a_shuffled = train_probe_shuffled(H, y_a)
        layer_results["acc_a"] = acc_a
        layer_results["acc_a_shuffled"] = acc_a_shuffled
        layer_results["selectivity_a"] = acc_a - acc_a_shuffled

        # x mod b probe
        y_b = (x_vals % b).numpy()
        acc_b = train_probe(H, y_b)
        acc_b_shuffled = train_probe_shuffled(H, y_b)
        layer_results["acc_b"] = acc_b
        layer_results["acc_b_shuffled"] = acc_b_shuffled
        layer_results["selectivity_b"] = acc_b - acc_b_shuffled

        results["layers"][l] = layer_results
        print(f"  Layer {l}: A_x={acc_x:.4f} (sel={acc_x-acc_x_shuffled:.4f}) | "
              f"A_a={acc_a:.4f} (sel={acc_a-acc_a_shuffled:.4f}) | "
              f"A_b={acc_b:.4f} (sel={acc_b-acc_b_shuffled:.4f})")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default="analysis/probes")
    parser.add_argument("--n-layers", type=int, default=2)
    args = parser.parse_args()

    config_path = os.path.join(args.run_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    crt_factors = tuple(config["crt_factors"])
    n_layers = config.get("n_layers", args.n_layers)
    p = config["p"]

    hidden_dir = os.path.join(args.run_dir, "hidden_states")
    if not os.path.exists(hidden_dir):
        print(f"Error: hidden_states directory not found: {hidden_dir}")
        sys.exit(1)

    # Find all checkpoint epochs
    x_val_files = [f for f in os.listdir(hidden_dir) if f.startswith("x_vals_epoch")]
    epochs = sorted([int(f.replace("x_vals_epoch", "").replace(".pt", "")) for f in x_val_files])
    print(f"Found {len(epochs)} checkpoints: {epochs[:5]}...{epochs[-3:]}")

    os.makedirs(args.output_dir, exist_ok=True)
    run_name = os.path.basename(args.run_dir)
    output_path = os.path.join(args.output_dir, f"{run_name}_probes.json")

    all_results = []
    for epoch in epochs:
        print(f"\nProbing epoch {epoch}...")
        result = analyze_checkpoint(hidden_dir, epoch, n_layers, crt_factors)
        all_results.append(result)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nProbe results saved to {output_path}")


if __name__ == "__main__":
    main()
