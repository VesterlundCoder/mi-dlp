#!/usr/bin/env python3
"""
Unified probing pipeline for LUMI DLP grokking checkpoints.

Loads checkpoints from LUMI download, reconstructs the model, extracts hidden
representations, and trains linear probes for:
  - x (full discrete log)
  - x mod a (CRT component A)
  - x mod b (CRT component B)

Works with both M04 (d_model=128) and M12 (d_model=768) checkpoints.

Usage:
  python lumi_probe_pipeline.py --checkpoint-dir checkpoints/lumi_download/M04_p113_s42 --output-dir analysis/lumi_probes
  python lumi_probe_pipeline.py --scaling-ladder --output-dir analysis/lumi_probes
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ============================================================================
# Constants
# ============================================================================
PAD, SEP, BOS, EOS = 0, 1, 2, 3
INT_OFFSET = 4


# ============================================================================
# CRT Math
# ============================================================================

def coprime_factorization(q: int) -> Tuple[int, int]:
    n = q
    unique_primes = []
    prime_powers = {}
    d = 2
    while d * d <= n:
        if n % d == 0:
            unique_primes.append(d)
            e = 0
            while n % d == 0:
                e += 1
                n //= d
            prime_powers[d] = e
        d += 1
    if n > 1:
        unique_primes.append(n)
        prime_powers[n] = 1
    if len(unique_primes) < 2:
        return 1, q
    best = None
    from itertools import product as iproduct
    for assignment in iproduct([0, 1], repeat=len(unique_primes)):
        a, b = 1, 1
        for i, pr in enumerate(unique_primes):
            if assignment[i] == 0:
                a *= pr ** prime_powers[pr]
            else:
                b *= pr ** prime_powers[pr]
        if a == 1 or b == 1:
            continue
        ratio = max(a, b) / min(a, b)
        if best is None or ratio < best[0]:
            best = (ratio, min(a, b), max(a, b))
    return best[1], best[2]


def project_factor(y: int, p: int, factor: int) -> int:
    return pow(y, (p - 1) // factor, p)


def _is_primitive_root(g: int, p: int) -> bool:
    if math.gcd(g, p) != 1:
        return False
    order = p - 1
    factors = set()
    n = order
    d = 2
    while d * d <= n:
        if n % d == 0:
            factors.add(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        factors.add(n)
    for q in factors:
        if pow(g, order // q, p) == 1:
            return False
    return True


def generate_dlp_problems(p: int) -> List[Tuple[int, int, int]]:
    triples = []
    for g in range(2, p):
        if _is_primitive_root(g, p):
            for x in range(p - 1):
                h = pow(g, x, p)
                triples.append((g, h, x))
    return triples


# ============================================================================
# Encoding (must match the LUMI trainer)
# ============================================================================

def encode_crt_representation(g: int, h: int, x: int, variant: str,
                              p: int = 113, crt_factors: Tuple[int, int] = None) -> Tuple[List[int], int]:
    if crt_factors is None:
        a, b = coprime_factorization(p - 1)
    else:
        a, b = crt_factors

    g_a = project_factor(g, p, a)
    g_b = project_factor(g, p, b)
    h_a = project_factor(h, p, a)
    h_b = project_factor(h, p, b)

    def tok(val: int) -> int:
        return val + INT_OFFSET

    target = tok(x)

    if variant in ("RAW", "standard"):
        tokens = [BOS, tok(g), SEP, tok(h), EOS]
    elif variant == "CRT-BOTH":
        tokens = [BOS, tok(g_b), SEP, tok(g_a), SEP, tok(h_b), SEP, tok(h_a), EOS]
    elif variant == "CRT-16":
        tokens = [BOS, tok(g_b), SEP, tok(h_b), EOS]
    elif variant == "CRT-7":
        tokens = [BOS, tok(g_a), SEP, tok(h_a), EOS]
    elif variant == "RAW+CRT":
        tokens = [BOS, tok(g), SEP, tok(h), SEP,
                  tok(g_b), SEP, tok(g_a), SEP,
                  tok(h_b), SEP, tok(h_a), EOS]
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return tokens, target


# ============================================================================
# Model (must match the LUMI trainer)
# ============================================================================

class GrokkingTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, max_len: int = 5, dropout: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_len = max_len

        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.unembed = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor, return_hidden: bool = False) -> torch.Tensor:
        B, T = tokens.size()
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        h = self.dropout(self.embed(tokens) + self.pos_embed(pos))

        hidden_states = []
        for layer in self.transformer.layers:
            h = layer(h)
            if return_hidden:
                hidden_states.append(h)

        logits = self.unembed(h)
        if return_hidden:
            return logits, hidden_states
        return logits


# ============================================================================
# Load checkpoint and extract hidden states
# ============================================================================

def load_model_from_checkpoint(ckpt_path: str, config: Dict, device: torch.device) -> GrokkingTransformer:
    """Load a model from a checkpoint."""
    d_model = config.get("d_model", 128)
    n_heads = config.get("n_heads", 4)
    n_layers = config.get("n_layers", 2)
    vocab_size = config.get("vocab_size", 117)
    max_len = config.get("max_len", 5)

    model = GrokkingTransformer(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, max_len=max_len, dropout=0.0
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model


def extract_hidden_states(model: GrokkingTransformer, triples: List[Tuple[int, int, int]],
                          variant: str, p: int, crt_factors: Tuple[int, int],
                          device: torch.device, batch_size: int = 256) -> Tuple[List[torch.Tensor], np.ndarray]:
    """Extract hidden states at the EOS position for all triples."""
    max_len = model.max_len
    all_hidden = {l: [] for l in range(model.n_layers)}
    all_x = []

    with torch.no_grad():
        for i in range(0, len(triples), batch_size):
            batch = triples[i:i+batch_size]
            tokens_list = []
            x_list = []
            for g, h, x in batch:
                tokens, target = encode_crt_representation(g, h, x, variant, p, crt_factors)
                padded = tokens + [PAD] * (max_len - len(tokens))
                padded = padded[:max_len]
                tokens_list.append(padded)
                x_list.append(x)

            tokens_tensor = torch.tensor(tokens_list, dtype=torch.long).to(device)
            logits, hidden_states = model(tokens_tensor, return_hidden=True)

            seq_lens = (tokens_tensor != PAD).sum(dim=1) - 1
            for l, h in enumerate(hidden_states):
                idx = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, h.size(-1))
                h_at_pos = h.gather(1, idx).squeeze(1)
                all_hidden[l].append(h_at_pos.cpu().detach())

            all_x.extend(x_list)

    for l in all_hidden:
        all_hidden[l] = torch.cat(all_hidden[l], dim=0)

    return all_hidden, np.array(all_x)


# ============================================================================
# Probing
# ============================================================================

def train_probe(X: np.ndarray, y: np.ndarray, max_iter: int = 2000) -> float:
    if len(np.unique(y)) < 2:
        return 0.0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    probe = LogisticRegression(max_iter=max_iter, C=1.0, solver='lbfgs')
    probe.fit(X_train, y_train)
    return probe.score(X_test, y_test)


def train_probe_shuffled(X: np.ndarray, y: np.ndarray, seed: int = 123) -> float:
    rng = np.random.RandomState(seed)
    y_shuffled = rng.permutation(y)
    return train_probe(X, y_shuffled)


def probe_checkpoint(ckpt_dir: str, output_dir: str, device_str: str = "auto"):
    """Probe a single checkpoint directory."""
    config_path = os.path.join(ckpt_dir, "config.json")
    summary_path = os.path.join(ckpt_dir, "summary.json")

    with open(config_path) as f:
        config = json.load(f)

    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    p = config.get("prime", config.get("p", 113))
    variant = config.get("variant", config.get("type", "RAW"))
    crt_factors = tuple(config.get("crt_factors", coprime_factorization(p - 1)))
    a, b = crt_factors

    name = os.path.basename(ckpt_dir)
    print(f"\n{'='*60}")
    print(f"Probing: {name}")
    print(f"  p={p}, variant={variant}, factors=({a},{b})")
    print(f"  d_model={config.get('d_model',128)}, best_acc={summary.get('best_test_acc','?')}")
    print(f"{'='*60}")

    # Find checkpoint
    ckpt_path = os.path.join(ckpt_dir, "checkpoints", "final.pt")
    if not os.path.exists(ckpt_path):
        # Try partial download
        ckpt_dir_files = os.listdir(os.path.join(ckpt_dir, "checkpoints")) if os.path.exists(os.path.join(ckpt_dir, "checkpoints")) else []
        partial = [f for f in ckpt_dir_files if f.startswith(".final.pt")]
        if partial:
            ckpt_path = os.path.join(ckpt_dir, "checkpoints", partial[0])
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

    # Load model
    model = load_model_from_checkpoint(ckpt_path, config, device)

    # Generate test data (use ALL triples for probing - we want to probe the representation)
    triples = generate_dlp_problems(p)
    print(f"  Generated {len(triples)} triples")

    # Extract hidden states
    hidden_states, x_vals = extract_hidden_states(
        model, triples, variant, p, crt_factors, device
    )

    # Probe each layer
    results = {
        "name": name,
        "p": p,
        "variant": variant,
        "crt_factors": list(crt_factors),
        "best_test_acc": summary.get("best_test_acc"),
        "total_epochs": summary.get("total_epochs"),
        "d_model": config.get("d_model", 128),
        "n_layers": config.get("n_layers", 2),
        "n_triples": len(triples),
        "layers": {},
    }

    for l in range(model.n_layers):
        H = hidden_states[l].numpy()
        layer_results = {}

        # Full x probe
        acc_x = train_probe(H, x_vals)
        acc_x_shuffled = train_probe_shuffled(H, x_vals)
        layer_results["acc_x"] = acc_x
        layer_results["acc_x_shuffled"] = acc_x_shuffled
        layer_results["selectivity_x"] = acc_x - acc_x_shuffled

        # x mod a probe
        y_a = x_vals % a
        acc_a = train_probe(H, y_a)
        acc_a_shuffled = train_probe_shuffled(H, y_a)
        layer_results["acc_a"] = acc_a
        layer_results["acc_a_shuffled"] = acc_a_shuffled
        layer_results["selectivity_a"] = acc_a - acc_a_shuffled

        # x mod b probe
        y_b = x_vals % b
        acc_b = train_probe(H, y_b)
        acc_b_shuffled = train_probe_shuffled(H, y_b)
        layer_results["acc_b"] = acc_b
        layer_results["acc_b_shuffled"] = acc_b_shuffled
        layer_results["selectivity_b"] = acc_b - acc_b_shuffled

        results["layers"][l] = layer_results
        print(f"  Layer {l}: A_x={acc_x:.4f} (sel={acc_x-acc_x_shuffled:.4f}) | "
              f"A_a={acc_a:.4f} (sel={acc_a-acc_a_shuffled:.4f}) | "
              f"A_b={acc_b:.4f} (sel={acc_b-acc_b_shuffled:.4f})")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}_probes.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {output_path}")

    return results


def probe_scaling_ladder(lumi_dir: str, output_dir: str, device_str: str = "auto"):
    """Probe all M12 CRT checkpoints in the scaling ladder."""
    base_dir = os.path.join(lumi_dir, "checkpoints", "lumi_download")
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} not found")
        return

    # Find all M12 CRT directories with final.pt
    m12_dirs = []
    for d in sorted(os.listdir(base_dir)):
        if d.startswith("M12_CRT_"):
            ckpt_path = os.path.join(base_dir, d, "checkpoints", "final.pt")
            if os.path.exists(ckpt_path):
                m12_dirs.append(os.path.join(base_dir, d))

    print(f"Found {len(m12_dirs)} M12 CRT checkpoints with final.pt")
    for d in m12_dirs:
        print(f"  {os.path.basename(d)}")

    all_results = []
    for ckpt_dir in m12_dirs:
        result = probe_checkpoint(ckpt_dir, output_dir, device_str)
        if result:
            all_results.append(result)

    # Save combined
    combined_path = os.path.join(output_dir, "scaling_ladder_probes.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to {combined_path}")

    # Print summary table
    print(f"\n{'='*80}")
    print("SCALING LADDER PROBE SUMMARY")
    print(f"{'='*80}")
    print(f"{'Name':<30} {'p':>6} {'Best Acc':>8} {'L0 A_x':>7} {'L0 A_a':>7} {'L0 A_b':>7} {'L1 A_x':>7} {'L1 A_a':>7} {'L1 A_b':>7}")
    print("-" * 80)
    for r in all_results:
        l0 = r["layers"].get("0", {})
        l1 = r["layers"].get("1", {})
        print(f"{r['name']:<30} {r['p']:>6} {r.get('best_test_acc',0):>8.4f} "
              f"{l0.get('acc_x',0):>7.4f} {l0.get('acc_a',0):>7.4f} {l0.get('acc_b',0):>7.4f} "
              f"{l1.get('acc_x',0):>7.4f} {l1.get('acc_a',0):>7.4f} {l1.get('acc_b',0):>7.4f}")


def probe_key_checkpoints(lumi_dir: str, output_dir: str, device_str: str = "auto"):
    """Probe the key RAW vs CRT checkpoints at p=113."""
    base_dir = os.path.join(lumi_dir, "checkpoints", "lumi_download")
    if not os.path.exists(base_dir):
        print(f"Error: {base_dir} not found")
        return

    key_dirs = []
    # RAW M04
    for d in ["M04_p113_s42", "M04_p113_f0.30_s123", "M04_p113_f0.30_s456", "M04_p113_f0.30_s789"]:
        path = os.path.join(base_dir, d)
        if os.path.exists(os.path.join(path, "checkpoints", "final.pt")):
            key_dirs.append(path)

    # CRT-BOTH M04
    for d in ["CRT-BOTH_p113_s42", "CRT-BOTH_p113_s123", "CRT-BOTH_p113_s7"]:
        path = os.path.join(base_dir, d)
        if os.path.exists(os.path.join(path, "checkpoints", "final.pt")):
            key_dirs.append(path)

    # CRT-16, CRT-7
    for d in ["CRT-16_p113_s42", "CRT-7_p113_s42"]:
        path = os.path.join(base_dir, d)
        if os.path.exists(os.path.join(path, "checkpoints", "final.pt")):
            key_dirs.append(path)

    print(f"Found {len(key_dirs)} key checkpoints")
    for d in key_dirs:
        print(f"  {os.path.basename(d)}")

    all_results = []
    for ckpt_dir in key_dirs:
        result = probe_checkpoint(ckpt_dir, output_dir, device_str)
        if result:
            all_results.append(result)

    combined_path = os.path.join(output_dir, "key_checkpoints_probes.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined results saved to {combined_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", help="Single checkpoint directory to probe")
    parser.add_argument("--scaling-ladder", action="store_true", help="Probe all M12 CRT scaling ladder checkpoints")
    parser.add_argument("--key-checkpoints", action="store_true", help="Probe key RAW vs CRT checkpoints at p=113")
    parser.add_argument("--output-dir", default="analysis/lumi_probes")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.checkpoint_dir:
        probe_checkpoint(args.checkpoint_dir, args.output_dir, args.device)
    elif args.scaling_ladder:
        probe_scaling_ladder(".", args.output_dir, args.device)
    elif args.key_checkpoints:
        probe_key_checkpoints(".", args.output_dir, args.device)
    else:
        # Default: probe key checkpoints
        probe_key_checkpoints(".", args.output_dir, args.device)
        probe_scaling_ladder(".", args.output_dir, args.device)


if __name__ == "__main__":
    main()
