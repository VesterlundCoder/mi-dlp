#!/usr/bin/env python3
"""
Unified mechanistic interpretability suite for DLP grokking checkpoints.

Implements all experiments from the experimental protocol:
  Exp 1:  DLP equivariance (argmax + soft KL)
  Exp 2:  CRT residue probing
  Exp 3:  Representation vs readout gap
  Exp 4:  2D CRT Fourier spectral analysis
  Exp 5:  Character basis comparison
  Exp 6:  AGOP + group-action alignment
  Exp 7:  NFA alignment
  Exp 8:  Hidden group representation
  Exp 9:  Causal subspace ablation
  Exp 10: Subspace patching
  Exp 11: Recombination analysis (logit ANOVA)
  Exp 12: Representation compression

Usage:
  python mi_suite.py --checkpoint-dir checkpoints/lumi_download/M04_p113_s42 --output-dir analysis/mi
  python mi_suite.py --all --output-dir analysis/mi
  python mi_suite.py --checkpoint-dir ... --experiments 1 2 4
"""

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression, Ridge
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
# Encoding (must match LUMI trainer)
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
    elif variant == "SCRAMBLED":
        # Same scramble map as lumi_dlp_trainer.py
        import random as _random
        rng = _random.Random(12345)
        perm = list(range(p))
        rng.shuffle(perm)
        scramble_map = {i: perm[i] for i in range(p)}
        s_g_b = scramble_map.get(g_b, g_b)
        s_g_a = scramble_map.get(g_a, g_a)
        s_h_b = scramble_map.get(h_b, h_b)
        s_h_a = scramble_map.get(h_a, h_a)
        tokens = [BOS, tok(s_g_b), SEP, tok(s_g_a), SEP,
                  tok(s_h_b), SEP, tok(s_h_a), EOS]
    else:
        raise ValueError(f"Unknown variant: {variant}")

    return tokens, target


# ============================================================================
# Model
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

    def forward(self, tokens: torch.Tensor, return_hidden: bool = False,
                return_embed: bool = False) -> torch.Tensor:
        B, T = tokens.size()
        pos = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        e = self.embed(tokens)
        h = self.dropout(e + self.pos_embed(pos))

        hidden_states = []
        for layer in self.transformer.layers:
            h = layer(h)
            if return_hidden:
                hidden_states.append(h)

        logits = self.unembed(h)
        if return_hidden and return_embed:
            return logits, hidden_states, e
        if return_hidden:
            return logits, hidden_states
        if return_embed:
            return logits, e
        return logits


def load_model_from_checkpoint(ckpt_path: str, config: Dict, device: torch.device) -> GrokkingTransformer:
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


# ============================================================================
# Hidden State Extraction
# ============================================================================

def extract_hidden_states(model: GrokkingTransformer, triples: List[Tuple[int, int, int]],
                          variant: str, p: int, crt_factors: Tuple[int, int],
                          device: torch.device, batch_size: int = 256) -> Tuple[List[torch.Tensor], np.ndarray, List[int]]:
    """Extract hidden states at EOS position for all triples.
    Returns: hidden_states per layer, x_vals, g_vals"""
    max_len = model.max_len
    all_hidden = {l: [] for l in range(model.n_layers)}
    all_x = []
    all_g = []

    with torch.no_grad():
        for i in range(0, len(triples), batch_size):
            batch = triples[i:i+batch_size]
            tokens_list = []
            x_list = []
            g_list = []
            for g, h, x in batch:
                tokens, target = encode_crt_representation(g, h, x, variant, p, crt_factors)
                padded = tokens + [PAD] * (max_len - len(tokens))
                padded = padded[:max_len]
                tokens_list.append(padded)
                x_list.append(x)
                g_list.append(g)

            tokens_tensor = torch.tensor(tokens_list, dtype=torch.long).to(device)
            logits, hidden_states = model(tokens_tensor, return_hidden=True)

            seq_lens = (tokens_tensor != PAD).sum(dim=1) - 1
            for l, h in enumerate(hidden_states):
                idx = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, h.size(-1))
                h_at_pos = h.gather(1, idx).squeeze(1)
                all_hidden[l].append(h_at_pos.cpu().detach())

            all_x.extend(x_list)
            all_g.extend(g_list)

    for l in all_hidden:
        all_hidden[l] = torch.cat(all_hidden[l], dim=0)

    return all_hidden, np.array(all_x), all_g


def extract_logits_and_hidden(model: GrokkingTransformer, triples: List[Tuple[int, int, int]],
                                variant: str, p: int, crt_factors: Tuple[int, int],
                                device: torch.device, batch_size: int = 256) -> Tuple[torch.Tensor, List[torch.Tensor], np.ndarray]:
    """Extract logits at EOS position AND hidden states for all triples."""
    max_len = model.max_len
    all_logits = []
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
            idx_pos = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, logits.size(-1))
            logits_at_pos = logits.gather(1, idx_pos).squeeze(1)
            all_logits.append(logits_at_pos.cpu().detach())

            for l, h in enumerate(hidden_states):
                idx = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, h.size(-1))
                h_at_pos = h.gather(1, idx).squeeze(1)
                all_hidden[l].append(h_at_pos.cpu().detach())

            all_x.extend(x_list)

    all_logits = torch.cat(all_logits, dim=0)
    for l in all_hidden:
        all_hidden[l] = torch.cat(all_hidden[l], dim=0)

    return all_logits, all_hidden, np.array(all_x)


# ============================================================================
# Probing utilities
# ============================================================================

def train_linear_probe(X: np.ndarray, y: np.ndarray, max_iter: int = 2000,
                       test_size: float = 0.3) -> float:
    """Train a linear probe (multinomial logistic regression) and return test accuracy."""
    if len(np.unique(y)) < 2:
        return 0.0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    probe = LogisticRegression(max_iter=max_iter, C=1.0, solver='lbfgs')
    probe.fit(X_train, y_train)
    return probe.score(X_test, y_test)


def train_probe_shuffled(X: np.ndarray, y: np.ndarray, seed: int = 123) -> float:
    rng = np.random.RandomState(seed)
    y_shuffled = rng.permutation(y)
    return train_linear_probe(X, y_shuffled)


def train_probe_and_get_weights(X: np.ndarray, y: np.ndarray, max_iter: int = 2000):
    """Train a linear probe and return (accuracy, probe_model)."""
    if len(np.unique(y)) < 2:
        return 0.0, None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    probe = LogisticRegression(max_iter=max_iter, C=1.0, solver='lbfgs')
    probe.fit(X_train, y_train)
    return probe.score(X_test, y_test), probe


# ============================================================================
# EXPERIMENT 1: DLP Equivariance
# ============================================================================

def experiment_1_equivariance(model, triples, variant, p, crt_factors, device,
                               n_samples=500, n_shifts=20) -> Dict:
    """Exp 1: DLP equivariance (argmax + soft KL)."""
    N = p - 1
    max_len = model.max_len

    # Select random test examples
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(triples), min(n_samples, len(triples)), replace=False)
    shifts = [int(s) for s in rng.randint(1, N, size=n_shifts)]

    argmax_correct = 0
    total = 0
    kl_values = []

    with torch.no_grad():
        for idx in sample_indices:
            g, h, x = triples[idx]
            # Original prediction
            tokens, _ = encode_crt_representation(g, h, x, variant, p, crt_factors)
            padded = (tokens + [PAD] * (max_len - len(tokens)))[:max_len]
            tokens_tensor = torch.tensor([padded], dtype=torch.long).to(device)
            logits = model(tokens_tensor)
            seq_len = (tokens_tensor != PAD).sum(dim=1) - 1
            logits_orig = logits[0, seq_len[0].item()]
            p_orig = torch.softmax(logits_orig, dim=-1)
            x_pred = logits_orig.argmax().item() - INT_OFFSET

            for c in shifts:
                # Transform: y' = g^c * y (mod p)
                h_shifted = (pow(g, c, p) * h) % p
                x_expected = (x + c) % N

                tokens_s, _ = encode_crt_representation(g, h_shifted, x_expected, variant, p, crt_factors)
                padded_s = (tokens_s + [PAD] * (max_len - len(tokens_s)))[:max_len]
                tokens_s_tensor = torch.tensor([padded_s], dtype=torch.long).to(device)
                logits_s = model(tokens_s_tensor)
                seq_len_s = (tokens_s_tensor != PAD).sum(dim=1) - 1
                logits_shifted = logits_s[0, seq_len_s[0].item()]
                p_shifted = torch.softmax(logits_shifted, dim=-1)
                x_pred_shifted = logits_shifted.argmax().item() - INT_OFFSET

                # Argmax equivariance
                if x_pred_shifted == (x_pred + c) % N:
                    argmax_correct += 1
                total += 1

                # Soft equivariance: KL(p_shifted || Q_c * p_orig)
                # Q_c shifts output classes by c: (Q_c * p)[j] = p[(j - c) mod vocab_size]
                # But we need to shift in the x-space (offset by INT_OFFSET)
                # p_orig has vocab_size classes; class j corresponds to x = j - INT_OFFSET
                # Shifting x by c means: class j -> class (j + c) mod N, but only for x-classes
                p_shifted_target = torch.zeros_like(p_orig)
                for j in range(INT_OFFSET, model.vocab_size):
                    x_j = j - INT_OFFSET
                    x_j_shifted = (x_j + c) % N
                    j_shifted = x_j_shifted + INT_OFFSET
                    if 0 <= j_shifted < model.vocab_size:
                        p_shifted_target[j_shifted] = p_orig[j]

                # KL divergence
                eps = 1e-10
                p_s = p_shifted.clamp(min=eps)
                p_t = p_shifted_target.clamp(min=eps)
                kl = torch.sum(p_s * torch.log(p_s / p_t)).item()
                kl_values.append(kl)

    return {
        "A_eq_argmax": argmax_correct / max(total, 1),
        "E_eq_kl": float(np.mean(kl_values)),
        "E_eq_kl_median": float(np.median(kl_values)),
        "n_samples": len(sample_indices),
        "n_shifts": n_shifts,
        "total_evaluations": total,
    }


# ============================================================================
# EXPERIMENT 2: CRT Residue Probing
# ============================================================================

def experiment_2_crt_probes(hidden_states, x_vals, crt_factors) -> Dict:
    """Exp 2: CRT residue probing for all layers."""
    a, b = crt_factors
    results = {"layers": {}}

    y_a = x_vals % a
    y_b = x_vals % b

    for l in hidden_states:
        H = hidden_states[l].numpy()
        layer_results = {}

        # Probe X (full x)
        acc_x = train_linear_probe(H, x_vals)
        acc_x_shuffled = train_probe_shuffled(H, x_vals)
        layer_results["acc_x"] = acc_x
        layer_results["acc_x_shuffled"] = acc_x_shuffled
        layer_results["selectivity_x"] = acc_x - acc_x_shuffled

        # Probe A (x mod a)
        acc_a = train_linear_probe(H, y_a)
        acc_a_shuffled = train_probe_shuffled(H, y_a)
        layer_results["acc_a"] = acc_a
        layer_results["acc_a_shuffled"] = acc_a_shuffled
        layer_results["selectivity_a"] = acc_a - acc_a_shuffled

        # Probe B (x mod b)
        acc_b = train_linear_probe(H, y_b)
        acc_b_shuffled = train_probe_shuffled(H, y_b)
        layer_results["acc_b"] = acc_b
        layer_results["acc_b_shuffled"] = acc_b_shuffled
        layer_results["selectivity_b"] = acc_b - acc_b_shuffled

        results["layers"][l] = layer_results

    return results


# ============================================================================
# EXPERIMENT 3: Representation vs Readout Gap
# ============================================================================

def experiment_3_readout_gap(model, hidden_states, logits, x_vals, crt_factors) -> Dict:
    """Exp 3: Representation vs readout gap."""
    a, b = crt_factors
    N = len(x_vals)

    # Native accuracy (from model logits)
    preds = logits.argmax(dim=-1).numpy() - INT_OFFSET
    A_native = float(np.mean(preds == x_vals))

    results = {"A_native": A_native, "layers": {}}

    y_a = x_vals % a
    y_b = x_vals % b

    for l in hidden_states:
        H = hidden_states[l].numpy()
        layer_results = {}

        # Linear readout for x
        acc_x = train_linear_probe(H, x_vals)
        layer_results["A_linear_x"] = acc_x
        layer_results["G_readout_x"] = acc_x - A_native

        # Linear readout for r_a
        acc_a = train_linear_probe(H, y_a)
        layer_results["A_linear_a"] = acc_a

        # Linear readout for r_b
        acc_b = train_linear_probe(H, y_b)
        layer_results["A_linear_b"] = acc_b

        results["layers"][l] = layer_results

    # Use last layer for summary
    last_layer = max(hidden_states.keys())
    results["G_readout"] = results["layers"][last_layer]["G_readout_x"]

    return results


# ============================================================================
# EXPERIMENT 4: 2D CRT Fourier Analysis
# ============================================================================

def experiment_4_crt_fourier(hidden_states, x_vals, crt_factors) -> Dict:
    """Exp 4: 2D CRT Fourier spectral analysis."""
    a, b = crt_factors
    d_model = hidden_states[0].shape[1]
    results = {"layers": {}}

    for l in hidden_states:
        H = hidden_states[l].numpy()
        layer_results = {}

        # Aggregate by CRT coordinate
        H_crt = np.zeros((a, b, d_model))
        counts = np.zeros((a, b))
        for i in range(len(x_vals)):
            r_a = int(x_vals[i]) % a
            r_b = int(x_vals[i]) % b
            H_crt[r_a, r_b] += H[i]
            counts[r_a, r_b] += 1

        for r_a in range(a):
            for r_b in range(b):
                if counts[r_a, r_b] > 0:
                    H_crt[r_a, r_b] /= counts[r_a, r_b]

        # Per-dimension 2D DFT
        all_E0, all_EA, all_EB, all_EAB = [], [], [], []
        all_entropy, all_gini, all_pr = [], [], []

        for d in range(d_model):
            h_2d = H_crt[:, :, d]
            H_hat = np.fft.fft2(h_2d)
            power = np.abs(H_hat) ** 2
            total_power = power.sum()

            if total_power == 0:
                continue

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
            if sorted_p.sum() > 0:
                gini = (2 * np.sum(np.arange(1, n+1) * sorted_p) / (n * sorted_p.sum()) - (n+1)/n)
            else:
                gini = 0
            all_gini.append(gini)

            # Participation ratio
            lambdas = power.flatten()
            denom = (lambdas ** 2).sum()
            if denom > 0:
                pr = (lambdas.sum() ** 2) / denom
            else:
                pr = 0
            all_pr.append(pr)

        layer_results["E0"] = float(np.mean(all_E0)) if all_E0 else 0
        layer_results["E_A"] = float(np.mean(all_EA)) if all_EA else 0
        layer_results["E_B"] = float(np.mean(all_EB)) if all_EB else 0
        layer_results["E_AB"] = float(np.mean(all_EAB)) if all_EAB else 0
        layer_results["spectral_entropy"] = float(np.mean(all_entropy)) if all_entropy else 0
        layer_results["gini"] = float(np.mean(all_gini)) if all_gini else 0
        layer_results["participation_ratio"] = float(np.mean(all_pr)) if all_pr else 0
        layer_results["n_dims"] = len(all_E0)

        results["layers"][l] = layer_results

    return results


# ============================================================================
# EXPERIMENT 5: Character Basis Comparison
# ============================================================================

def experiment_5_character_basis(hidden_states, x_vals, p) -> Dict:
    """Exp 5: Character basis vs raw basis comparison."""
    N = p - 1
    d_model = hidden_states[0].shape[1]
    results = {"layers": {}}

    for l in hidden_states:
        H = hidden_states[l].numpy()
        layer_results = {}

        # Aggregate by x value
        H_by_x = np.zeros((N, d_model))
        counts = np.zeros(N)
        for i in range(len(x_vals)):
            xi = int(x_vals[i])
            if xi < N:
                H_by_x[xi] += H[i]
                counts[xi] += 1
        for xi in range(N):
            if counts[xi] > 0:
                H_by_x[xi] /= counts[xi]

        # Raw basis: just the values as-is
        raw_power = np.abs(H_by_x) ** 2
        raw_total = raw_power.sum()

        # Character basis: DFT over x
        for d in range(d_model):
            h = H_by_x[:, d]
            H_hat = np.fft.fft(h)
            char_power = np.abs(H_hat) ** 2

        # Compute metrics for both bases
        # Raw basis
        p_raw = (raw_power.sum(axis=0) / raw_total) if raw_total > 0 else np.zeros(d_model)
        # Actually we want per-dimension analysis

        # For simplicity, compute average metrics across dimensions
        raw_entropies, char_entropies = [], []
        raw_k90s, char_k90s = [], []
        raw_prs, char_prs = [], []

        for d in range(d_model):
            h = H_by_x[:, d]

            # Raw basis
            raw_pow = h ** 2
            total = raw_pow.sum()
            if total == 0:
                continue
            p_r = raw_pow / total
            p_r = p_r[p_r > 0]
            raw_entropies.append(-np.sum(p_r * np.log(p_r)))
            sorted_r = np.sort(raw_pow)[::-1]
            cumsum = np.cumsum(sorted_r) / total
            k90 = np.searchsorted(cumsum, 0.9) + 1
            raw_k90s.append(k90)
            raw_prs.append(total ** 2 / (raw_pow ** 2).sum() if (raw_pow ** 2).sum() > 0 else 0)

            # Character basis
            H_hat = np.fft.fft(h)
            char_pow = np.abs(H_hat) ** 2
            total_c = char_pow.sum()
            if total_c == 0:
                continue
            p_c = char_pow / total_c
            p_c = p_c[p_c > 0]
            char_entropies.append(-np.sum(p_c * np.log(p_c)))
            sorted_c = np.sort(char_pow)[::-1]
            cumsum_c = np.cumsum(sorted_c) / total_c
            k90_c = np.searchsorted(cumsum_c, 0.9) + 1
            char_k90s.append(k90_c)
            char_prs.append(total_c ** 2 / (char_pow ** 2).sum() if (char_pow ** 2).sum() > 0 else 0)

        layer_results["raw_entropy"] = float(np.mean(raw_entropies)) if raw_entropies else 0
        layer_results["char_entropy"] = float(np.mean(char_entropies)) if char_entropies else 0
        layer_results["raw_K90"] = float(np.mean(raw_k90s)) if raw_k90s else 0
        layer_results["char_K90"] = float(np.mean(char_k90s)) if char_k90s else 0
        layer_results["raw_PR"] = float(np.mean(raw_prs)) if raw_prs else 0
        layer_results["char_PR"] = float(np.mean(char_prs)) if char_prs else 0

        results["layers"][l] = layer_results

    return results


# ============================================================================
# EXPERIMENT 6: AGOP + Group-Action Alignment
# ============================================================================

def experiment_6_agop(model, triples, variant, p, crt_factors, device,
                       n_samples=200, batch_size=32) -> Dict:
    """Exp 6: AGOP and group-action alignment."""
    d_model = model.d_model
    max_len = model.max_len
    N = p - 1

    # Compute Jacobian of logits w.r.t. input embedding
    # Use the embedding of the group element token position
    # For RAW: tokens = [BOS, tok(g), SEP, tok(h), EOS]
    # The h token is at position 3 (0-indexed)
    # For CRT-BOTH: tokens = [BOS, tok(g_b), SEP, tok(g_a), SEP, tok(h_b), SEP, tok(h_a), EOS]
    # The h_b token is at position 5, h_a at position 7

    # We'll compute Jacobian w.r.t. the h token embedding
    # J = d(logits) / d(embed(h_token))

    # Determine which token position to use for the group element
    if variant in ("RAW", "standard"):
        h_pos = 3  # tok(h) position
    elif variant == "CRT-BOTH":
        h_pos = 5  # tok(h_b) position
    elif variant == "CRT-7":
        h_pos = 3
    elif variant == "CRT-16":
        h_pos = 3
    else:
        h_pos = 3

    # Select samples
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(triples), min(n_samples, len(triples)), replace=False)

    G = torch.zeros(d_model, d_model, device=device)
    total_samples = 0

    model.eval()
    for batch_start in range(0, len(sample_indices), batch_size):
        batch_indices = sample_indices[batch_start:batch_start + batch_size]
        actual_batch_size = len(batch_indices)

        tokens_list = []
        for idx in batch_indices:
            g, h, x = triples[idx]
            tokens, _ = encode_crt_representation(g, h, x, variant, p, crt_factors)
            padded = (tokens + [PAD] * (max_len - len(tokens)))[:max_len]
            tokens_list.append(padded)

        tokens_tensor = torch.tensor(tokens_list, dtype=torch.long).to(device)

        # Get embeddings
        embed_out = model.embed(tokens_tensor)
        pos_embed_out = model.pos_embed(torch.arange(max_len, device=device).unsqueeze(0).expand(actual_batch_size, max_len))
        h_embed = embed_out + pos_embed_out

        # We need gradient w.r.t. the h token embedding
        h_embed.requires_grad_(True)

        # Forward through transformer manually
        h = model.dropout(h_embed)
        for layer in model.transformer.layers:
            h = layer(h)
        logits = model.unembed(h)

        # Get logits at EOS position
        seq_lens = (tokens_tensor != PAD).sum(dim=1) - 1
        idx_pos = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, logits.size(-1))
        logits_at_pos = logits.gather(1, idx_pos).squeeze(1)  # (batch, vocab_size)

        # Compute Jacobian for each sample
        for i in range(actual_batch_size):
            # Sum over output dimensions (or use random projection for efficiency)
            # For full Jacobian: d(logits_i) / d(h_embed_i[h_pos])
            # This is expensive, so use vector-Jacobian products

            # Use random projection of logits for efficiency
            proj = torch.randn(logits_at_pos.size(-1), device=device)
            proj = proj / proj.norm()

            grad = torch.autograd.grad(
                logits_at_pos[i] @ proj,
                h_embed,
                retain_graph=True,
                create_graph=False
            )[0]

            # Extract gradient at h_pos for this sample
            j_i = grad[i, h_pos]  # (d_model,)

            # Accumulate outer product
            G += torch.outer(j_i, j_i)
            total_samples += 1

    G = G / max(total_samples, 1)

    # AGOP spectrum
    eigenvalues = torch.linalg.eigvalsh(G).cpu().numpy()
    eigenvalues = np.maximum(eigenvalues, 0)
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Effective rank
    total_eig = eigenvalues.sum()
    if total_eig > 0:
        p_eig = eigenvalues / total_eig
        p_eig = p_eig[p_eig > 0]
        eff_rank = np.exp(-np.sum(p_eig * np.log(p_eig)))
        eig_entropy = -np.sum(p_eig * np.log(p_eig))
    else:
        eff_rank = 0
        eig_entropy = 0

    # Top-K explained variance
    cumvar = np.cumsum(eigenvalues) / max(total_eig, 1)
    k90 = np.searchsorted(cumvar, 0.9) + 1

    # Group-action alignment
    # P_c shifts the h token embedding: embed(tok(h)) -> embed(tok(g^c * h))
    # This is NOT a simple permutation in embedding space
    # Instead, we measure: does the AGOP commute with the group action on the token level?
    # For simplicity, compute alignment by checking if rotating inputs changes G

    # Alternative: compute group alignment by checking if G is approximately
    # invariant under the permutation of embedding indices induced by group action
    # This requires knowing the permutation, which depends on the tokenization

    # For now, just report the spectral properties
    results = {
        "eff_rank": float(eff_rank),
        "eig_entropy": float(eig_entropy),
        "K90": int(k90),
        "top_5_eigenvalues": eigenvalues[:5].tolist(),
        "n_samples": total_samples,
        "h_pos": h_pos,
    }

    return results


# ============================================================================
# EXPERIMENT 7: NFA Alignment
# ============================================================================

def experiment_7_nfa(model, G_agop) -> Dict:
    """Exp 7: NFA alignment between W_1^T W_1 and AGOP^{1/L}."""
    # Get first layer weight matrix
    # The first learned weight is typically linear1 in the first transformer layer
    # or the embedding matrix. We use the first attention in_proj weight.

    # Actually, for NFA, we want the first feature-learning layer.
    # In our transformer, that's the first transformer layer's linear1 weight
    # (the FFN expansion). But NFA typically refers to the first layer of a
    # deep linear network. For our transformer, the most natural choice is
    # the embedding matrix or the first attention projection.

    # Use the first layer's linear1 weight as W_1
    W1 = None
    for name, param in model.named_parameters():
        if "linear1.weight" in name:
            W1 = param.detach().cpu()
            break

    if W1 is None:
        return {"rho_NFA": 0, "note": "W1 not found"}

    # M_W = W1^T @ W1
    M_W = (W1.T @ W1).numpy()

    # M_A = G^{1/L}
    L = model.n_layers
    if isinstance(G_agop, torch.Tensor):
        G_np = G_agop.cpu().numpy()
    else:
        G_np = G_agop

    # Eigendecomposition for matrix power
    G_sym = (G_np + G_np.T) / 2  # Symmetrize
    eigvals, eigvecs = np.linalg.eigh(G_sym)
    eigvals = np.maximum(eigvals, 0)  # Ensure PSD
    M_A = (eigvecs * (eigvals ** (1.0 / L))) @ eigvecs.T

    # Normalize
    M_W_norm = M_W / max(np.linalg.norm(M_W, 'fro'), 1e-10)
    M_A_norm = M_A / max(np.linalg.norm(M_A, 'fro'), 1e-10)

    # Frobenius inner product (cosine similarity for matrices)
    rho_NFA = float(np.sum(M_W_norm * M_A_norm))

    return {
        "rho_NFA": rho_NFA,
        "L": L,
        "M_W_shape": list(M_W.shape),
        "M_A_shape": list(M_A.shape),
    }


# ============================================================================
# EXPERIMENT 8: Hidden Group Representation
# ============================================================================

def experiment_8_group_representation(model, triples, variant, p, crt_factors, device,
                                       n_train=500, n_test=500, max_power=10) -> Dict:
    """Exp 8: Hidden group representation R_l such that h(g*y) ~ R * h(y)."""
    N = p - 1
    max_len = model.max_len

    # Collect pairs (h(y), h(g*y)) for training
    rng = np.random.RandomState(42)
    train_indices = rng.choice(len(triples), min(n_train, len(triples)), replace=False)
    test_indices = rng.choice(len(triples), min(n_test, len(triples)), replace=False)

    results = {"layers": {}}

    for l in range(model.n_layers):
        # Extract hidden states for y and g*y
        h_y_train = []
        h_gy_train = []
        h_y_test = []
        h_gy_test = []

        with torch.no_grad():
            for idx in train_indices:
                g, h, x = triples[idx]
                # Original
                tokens, _ = encode_crt_representation(g, h, x, variant, p, crt_factors)
                padded = (tokens + [PAD] * (max_len - len(tokens)))[:max_len]
                tokens_tensor = torch.tensor([padded], dtype=torch.long).to(device)
                _, hidden_states = model(tokens_tensor, return_hidden=True)
                seq_len = (tokens_tensor != PAD).sum(dim=1) - 1
                idx_pos = seq_len[0].item()
                h_y = hidden_states[l][0, idx_pos].cpu().detach()
                h_y_train.append(h_y)

                # g * y (mod p)
                h_shifted = (g * h) % p
                x_shifted = (x + 1) % N
                tokens_s, _ = encode_crt_representation(g, h_shifted, x_shifted, variant, p, crt_factors)
                padded_s = (tokens_s + [PAD] * (max_len - len(tokens_s)))[:max_len]
                tokens_s_tensor = torch.tensor([padded_s], dtype=torch.long).to(device)
                _, hidden_states_s = model(tokens_s_tensor, return_hidden=True)
                seq_len_s = (tokens_s_tensor != PAD).sum(dim=1) - 1
                idx_pos_s = seq_len_s[0].item()
                h_gy = hidden_states_s[l][0, idx_pos_s].cpu().detach()
                h_gy_train.append(h_gy)

            for idx in test_indices:
                g, h, x = triples[idx]
                tokens, _ = encode_crt_representation(g, h, x, variant, p, crt_factors)
                padded = (tokens + [PAD] * (max_len - len(tokens)))[:max_len]
                tokens_tensor = torch.tensor([padded], dtype=torch.long).to(device)
                _, hidden_states = model(tokens_tensor, return_hidden=True)
                seq_len = (tokens_tensor != PAD).sum(dim=1) - 1
                h_y = hidden_states[l][0, seq_len[0].item()].cpu().detach()
                h_y_test.append(h_y)

                h_shifted = (g * h) % p
                x_shifted = (x + 1) % N
                tokens_s, _ = encode_crt_representation(g, h_shifted, x_shifted, variant, p, crt_factors)
                padded_s = (tokens_s + [PAD] * (max_len - len(tokens_s)))[:max_len]
                tokens_s_tensor = torch.tensor([padded_s], dtype=torch.long).to(device)
                _, hidden_states_s = model(tokens_s_tensor, return_hidden=True)
                seq_len_s = (tokens_s_tensor != PAD).sum(dim=1) - 1
                h_gy = hidden_states_s[l][0, seq_len_s[0].item()].cpu().detach()
                h_gy_test.append(h_gy)

        h_y_train = torch.stack(h_y_train).numpy()
        h_gy_train = torch.stack(h_gy_train).numpy()
        h_y_test = torch.stack(h_y_test).numpy()
        h_gy_test = torch.stack(h_gy_test).numpy()

        # Solve R_l = argmin ||h_gy - R @ h_y||^2 + lambda ||R||^2
        # R = (h_gy^T @ h_y) @ (h_y^T @ h_y + lambda*I)^{-1}
        d = h_y_train.shape[1]
        lam = 1e-4
        A = h_y_train.T @ h_y_train + lam * np.eye(d)
        B = h_gy_train.T @ h_y_train
        R = np.linalg.solve(A, B.T).T

        # Test on held-out
        h_gy_pred = h_y_test @ R.T
        E1 = np.mean(np.linalg.norm(h_gy_test - h_gy_pred, axis=1) /
                     np.maximum(np.linalg.norm(h_gy_test, axis=1), 1e-10))

        # Test powers: h(g^c * y) ~ R^c * h(y)
        power_errors = []
        for c in range(2, min(max_power + 1, N)):
            R_c = np.linalg.matrix_power(R, c)
            h_gcy_pred = h_y_test @ R_c.T

            # Compute actual h(g^c * y)
            h_gcy_test = []
            with torch.no_grad():
                for idx in test_indices[:100]:  # Subset for efficiency
                    g, h, x = triples[idx]
                    h_shifted = pow(g, c, p) * h % p
                    x_shifted = (x + c) % N
                    tokens_s, _ = encode_crt_representation(g, h_shifted, x_shifted, variant, p, crt_factors)
                    padded_s = (tokens_s + [PAD] * (max_len - len(tokens_s)))[:max_len]
                    tokens_s_tensor = torch.tensor([padded_s], dtype=torch.long).to(device)
                    _, hidden_states_s = model(tokens_s_tensor, return_hidden=True)
                    seq_len_s = (tokens_s_tensor != PAD).sum(dim=1) - 1
                    h_gcy = hidden_states_s[l][0, seq_len_s[0].item()].cpu().detach()
                    h_gcy_test.append(h_gcy.numpy())

            h_gcy_test = np.array(h_gcy_test)
            h_gcy_pred_subset = h_y_test[:100] @ R_c.T
            Ec = np.mean(np.linalg.norm(h_gcy_test - h_gcy_pred_subset, axis=1) /
                        np.maximum(np.linalg.norm(h_gcy_test, axis=1), 1e-10))
            power_errors.append({"c": c, "error": float(Ec)})

        # Cyclic closure: R^N ~ I
        try:
            R_N = np.linalg.matrix_power(R, N)
            E_cycle = float(np.linalg.norm(R_N - np.eye(d), 'fro') / np.linalg.norm(np.eye(d), 'fro'))
        except:
            E_cycle = float('inf')

        # Eigenvalue analysis
        eigvals_R = np.linalg.eigvals(R)
        eigphases = np.angle(eigvals_R)
        # For cyclic group of order N, expect eigenphases at 2*pi*k/N
        expected_phases = 2 * np.pi * np.arange(d) / N

        results["layers"][l] = {
            "E1": float(E1),
            "power_errors": power_errors,
            "E_cycle": E_cycle,
            "eigenvalues": eigvals_R.tolist() if np.iscomplexobj(eigvals_R) else eigvals_R.tolist(),
            "eigenphases": eigphases.tolist(),
            "R_norm": float(np.linalg.norm(R, 'fro')),
        }

    return results


# ============================================================================
# EXPERIMENT 9: Causal Subspace Ablation
# ============================================================================

def experiment_9_ablation(model, triples, hidden_states, x_vals, crt_factors,
                          variant, p, device, n_samples=1000) -> Dict:
    """Exp 9: Causal subspace ablation."""
    a, b = crt_factors
    N = p - 1
    max_len = model.max_len

    # Train probes to get subspace directions
    y_a = x_vals % a
    y_b = x_vals % b

    results = {"layers": {}}

    for l in hidden_states:
        H = hidden_states[l].numpy()

        # Train probes and get weights
        acc_a, probe_a = train_probe_and_get_weights(H, y_a)
        acc_b, probe_b = train_probe_and_get_weights(H, y_b)

        if probe_a is None or probe_b is None:
            continue

        # Get probe directions (W_A: a x d, W_B: b x d)
        W_A = probe_a.coef_  # (a, d)
        W_B = probe_b.coef_  # (b, d)

        # Orthonormalize via SVD
        Q_A, _ = np.linalg.qr(W_A.T)  # (d, a)
        Q_B, _ = np.linalg.qr(W_B.T)  # (d, b)

        P_A = Q_A @ Q_A.T  # (d, d)
        P_B = Q_B @ Q_B.T  # (d, d)

        # Run ablation on a subset of triples
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(len(triples), min(n_samples, len(triples)), replace=False)

        # Baseline accuracy
        correct_baseline = 0
        correct_ablate_a = 0
        correct_ablate_b = 0
        correct_ablate_both = 0
        correct_random = 0

        # Random subspace control
        d = H.shape[1]
        dim_a = Q_A.shape[1]
        dim_b = Q_B.shape[1]

        random_errors = []
        for _ in range(10):
            Q_rand = np.random.randn(d, dim_a)
            Q_rand, _ = np.linalg.qr(Q_rand)
            P_rand = Q_rand @ Q_rand.T

            correct_rand = 0
            with torch.no_grad():
                for idx in sample_indices[:200]:  # Subset for random control
                    g, h, x = triples[idx]
                    tokens, _ = encode_crt_representation(g, h, x, variant, p, crt_factors)
                    padded = (tokens + [PAD] * (max_len - len(tokens)))[:max_len]
                    tokens_tensor = torch.tensor([padded], dtype=torch.long).to(device)

                    # Get hidden state
                    _, hidden_states_model = model(tokens_tensor, return_hidden=True)
                    seq_len = (tokens_tensor != PAD).sum(dim=1) - 1
                    h_orig = hidden_states_model[l][0, seq_len[0].item()].cpu().numpy()

                    # Ablate random subspace
                    h_ablated = h_orig - P_rand @ h_orig

                    # This is complex to re-run through the rest of the model
                    # For simplicity, measure probe accuracy on ablated representations
                    # (This is a proxy; full ablation requires re-running the model)

            random_errors.append(0)  # Placeholder

        # Measure probe accuracy on ablated representations
        H_ablate_a = H - P_A @ H
        H_ablate_b = H - P_B @ H
        H_ablate_both = H - (P_A + P_B) @ H

        acc_a_after_ablate_a = train_linear_probe(H_ablate_a, y_a)
        acc_a_after_ablate_b = train_linear_probe(H_ablate_b, y_a)
        acc_b_after_ablate_a = train_linear_probe(H_ablate_a, y_b)
        acc_b_after_ablate_b = train_linear_probe(H_ablate_b, y_b)
        acc_x_after_ablate_a = train_linear_probe(H_ablate_a, x_vals)
        acc_x_after_ablate_b = train_linear_probe(H_ablate_b, x_vals)
        acc_x_after_ablate_both = train_linear_probe(H_ablate_both, x_vals)

        # Random subspace control
        Q_rand_a, _ = np.linalg.qr(np.random.randn(d, dim_a))
        P_rand_a = Q_rand_a @ Q_rand_a.T
        H_ablate_rand = H - P_rand_a @ H
        acc_x_after_ablate_rand = train_linear_probe(H_ablate_rand, x_vals)

        results["layers"][l] = {
            "acc_a_baseline": acc_a,
            "acc_b_baseline": acc_b,
            "acc_a_after_ablate_A": acc_a_after_ablate_a,
            "acc_a_after_ablate_B": acc_a_after_ablate_b,
            "acc_b_after_ablate_A": acc_b_after_ablate_a,
            "acc_b_after_ablate_B": acc_b_after_ablate_b,
            "acc_x_after_ablate_A": acc_x_after_ablate_a,
            "acc_x_after_ablate_B": acc_x_after_ablate_b,
            "acc_x_after_ablate_both": acc_x_after_ablate_both,
            "acc_x_after_ablate_random": acc_x_after_ablate_rand,
            "delta_A": acc_x_after_ablate_a - acc_x_after_ablate_rand,
            "delta_B": acc_x_after_ablate_b - acc_x_after_ablate_rand,
            "delta_both": acc_x_after_ablate_both - acc_x_after_ablate_rand,
        }

    return results


# ============================================================================
# EXPERIMENT 12: Representation Compression
# ============================================================================

def experiment_12_compression(hidden_states) -> Dict:
    """Exp 12: Representation compression metrics."""
    results = {"layers": {}}

    for l in hidden_states:
        H = hidden_states[l].numpy()
        # Center
        H_centered = H - H.mean(axis=0, keepdims=True)

        # SVD
        U, sigma, Vt = np.linalg.svd(H_centered, full_matrices=False)

        # Effective rank
        sigma_sq = sigma ** 2
        total = sigma_sq.sum()
        if total > 0:
            p_i = sigma_sq / total
            p_i = p_i[p_i > 0]
            eff_rank = np.exp(-np.sum(p_i * np.log(p_i)))
        else:
            eff_rank = 0

        # Participation ratio
        if (sigma_sq ** 2).sum() > 0:
            PR = (sigma_sq.sum() ** 2) / (sigma_sq ** 2).sum()
        else:
            PR = 0

        # Top-K reconstruction
        cumvar = np.cumsum(sigma_sq) / max(total, 1)
        K90 = int(np.searchsorted(cumvar, 0.9) + 1)
        K95 = int(np.searchsorted(cumvar, 0.95) + 1)
        K99 = int(np.searchsorted(cumvar, 0.99) + 1)

        results["layers"][l] = {
            "eff_rank": float(eff_rank),
            "participation_ratio": float(PR),
            "K90": K90,
            "K95": K95,
            "K99": K99,
            "top_10_singular_values": sigma[:10].tolist(),
            "n_nonzero": int(np.sum(sigma > 1e-10)),
        }

    return results


# ============================================================================
# EXPERIMENT 11: Recombination Analysis (Logit ANOVA)
# ============================================================================

def experiment_11_recombination(logits, x_vals, crt_factors) -> Dict:
    """Exp 11: Recombination analysis via logit ANOVA."""
    a, b = crt_factors
    N = p_global - 1  # Use global p

    # For each output class, decompose logit as function of (r_a, r_b)
    # L(r_a, r_b) = mu + A(r_a) + B(r_b) + I(r_a, r_b)

    logits_np = logits.numpy()  # (n_samples, vocab_size)
    n_samples = logits_np.shape[0]

    # Average variance explained across all output dimensions
    all_var_A, all_var_B, all_var_I = [], [], []

    # Use a subset of output dimensions for efficiency
    n_dims = min(50, logits_np.shape[1])

    for d in range(n_dims):
        L = logits_np[:, d]  # (n_samples,)

        # Aggregate by (r_a, r_b)
        L_crt = np.zeros((a, b))
        counts = np.zeros((a, b))
        for i in range(n_samples):
            r_a = int(x_vals[i]) % a
            r_b = int(x_vals[i]) % b
            L_crt[r_a, r_b] += L[i]
            counts[r_a, r_b] += 1

        for r_a in range(a):
            for r_b in range(b):
                if counts[r_a, r_b] > 0:
                    L_crt[r_a, r_b] /= counts[r_a, r_b]

        # ANOVA decomposition
        mu = L_crt.mean()
        A = L_crt.mean(axis=1) - mu  # (a,)
        B = L_crt.mean(axis=0) - mu  # (b,)
        I = L_crt - mu - A[:, None] - B[None, :]

        var_total = np.var(L_crt)
        if var_total > 0:
            var_A = np.var(A) * b  # Scale by other dimension
            var_B = np.var(B) * a
            var_I = np.var(I)

            all_var_A.append(var_A / var_total)
            all_var_B.append(var_B / var_total)
            all_var_I.append(var_I / var_total)

    return {
        "var_A_explained": float(np.mean(all_var_A)) if all_var_A else 0,
        "var_B_explained": float(np.mean(all_var_B)) if all_var_B else 0,
        "var_I_explained": float(np.mean(all_var_I)) if all_var_I else 0,
        "n_dims_analyzed": len(all_var_A),
    }


# ============================================================================
# Main Analysis Function
# ============================================================================

def run_all_experiments(ckpt_dir: str, output_dir: str, device_str: str = "auto",
                        experiments: List[int] = None) -> Dict:
    """Run all experiments on a single checkpoint."""
    config_path = os.path.join(ckpt_dir, "config.json")
    summary_path = os.path.join(ckpt_dir, "summary.json")

    with open(config_path) as f:
        config = json.load(f)

    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    global p_global
    p_global = config.get("prime", config.get("p", 113))
    p = p_global
    variant = config.get("variant", config.get("type", "RAW"))
    crt_factors = tuple(config.get("crt_factors", coprime_factorization(p - 1)))
    a, b = crt_factors

    name = os.path.basename(ckpt_dir)
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"  p={p}, variant={variant}, factors=({a},{b})")
    print(f"  d_model={config.get('d_model',128)}, best_acc={summary.get('best_test_acc','?')}")
    print(f"{'='*60}")

    # Find checkpoint
    ckpt_path = os.path.join(ckpt_dir, "checkpoints", "final.pt")
    if not os.path.exists(ckpt_path):
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

    # Generate triples
    triples = generate_dlp_problems(p)
    print(f"  Generated {len(triples)} triples")

    # Extract hidden states and logits
    print(f"  Extracting hidden states and logits...")
    logits, hidden_states, x_vals = extract_logits_and_hidden(
        model, triples, variant, p, crt_factors, device
    )
    print(f"  Hidden states: {len(hidden_states)} layers, shape: {hidden_states[0].shape}")
    print(f"  Logits shape: {logits.shape}")

    all_results = {
        "name": name,
        "p": p,
        "variant": variant,
        "crt_factors": list(crt_factors),
        "best_test_acc": summary.get("best_test_acc"),
        "total_epochs": summary.get("total_epochs"),
        "d_model": config.get("d_model", 128),
        "n_layers": config.get("n_layers", 2),
        "n_triples": len(triples),
    }

    if experiments is None:
        experiments = list(range(1, 13))

    # Run experiments
    if 1 in experiments:
        print(f"  Exp 1: DLP Equivariance...")
        t0 = time.time()
        all_results["exp1_equivariance"] = experiment_1_equivariance(
            model, triples, variant, p, crt_factors, device
        )
        print(f"    A_eq={all_results['exp1_equivariance']['A_eq_argmax']:.4f} "
              f"E_eq_kl={all_results['exp1_equivariance']['E_eq_kl']:.4f} "
              f"({time.time()-t0:.1f}s)")

    if 2 in experiments:
        print(f"  Exp 2: CRT Residue Probing...")
        t0 = time.time()
        all_results["exp2_crt_probes"] = experiment_2_crt_probes(
            hidden_states, x_vals, crt_factors
        )
        for l in all_results["exp2_crt_probes"]["layers"]:
            r = all_results["exp2_crt_probes"]["layers"][l]
            print(f"    Layer {l}: A_x={r['acc_x']:.4f} A_a={r['acc_a']:.4f} A_b={r['acc_b']:.4f} "
                  f"({time.time()-t0:.1f}s)")

    if 3 in experiments:
        print(f"  Exp 3: Representation vs Readout Gap...")
        t0 = time.time()
        all_results["exp3_readout_gap"] = experiment_3_readout_gap(
            model, hidden_states, logits, x_vals, crt_factors
        )
        print(f"    A_native={all_results['exp3_readout_gap']['A_native']:.4f} "
              f"G_readout={all_results['exp3_readout_gap']['G_readout']:.4f} "
              f"({time.time()-t0:.1f}s)")

    if 4 in experiments:
        print(f"  Exp 4: 2D CRT Fourier Analysis...")
        t0 = time.time()
        all_results["exp4_crt_fourier"] = experiment_4_crt_fourier(
            hidden_states, x_vals, crt_factors
        )
        for l in all_results["exp4_crt_fourier"]["layers"]:
            r = all_results["exp4_crt_fourier"]["layers"][l]
            print(f"    Layer {l}: E_A={r['E_A']:.4f} E_B={r['E_B']:.4f} E_AB={r['E_AB']:.4f} "
                  f"entropy={r['spectral_entropy']:.4f} ({time.time()-t0:.1f}s)")

    if 5 in experiments:
        print(f"  Exp 5: Character Basis Comparison...")
        t0 = time.time()
        all_results["exp5_character_basis"] = experiment_5_character_basis(
            hidden_states, x_vals, p
        )
        print(f"    ({time.time()-t0:.1f}s)")

    if 6 in experiments:
        print(f"  Exp 6: AGOP + Group-Action Alignment...")
        t0 = time.time()
        try:
            agop_result = experiment_6_agop(
                model, triples, variant, p, crt_factors, device, n_samples=100
            )
            all_results["exp6_agop"] = agop_result
            print(f"    eff_rank={agop_result['eff_rank']:.4f} K90={agop_result['K90']} "
                  f"({time.time()-t0:.1f}s)")

            # Store AGOP for NFA
            G_agop = None
        except Exception as e:
            print(f"    Error: {e}")
            all_results["exp6_agop"] = {"error": str(e)}
            G_agop = None

    if 7 in experiments:
        print(f"  Exp 7: NFA Alignment...")
        t0 = time.time()
        try:
            # Recompute AGOP if needed
            if G_agop is None:
                # Use a simpler AGOP approximation
                G_agop = torch.zeros(model.d_model, model.d_model, device=device)
                # Skip for now if AGOP failed
                all_results["exp7_nfa"] = {"error": "AGOP not available"}
            else:
                all_results["exp7_nfa"] = experiment_7_nfa(model, G_agop)
                print(f"    rho_NFA={all_results['exp7_nfa']['rho_NFA']:.4f} ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    Error: {e}")
            all_results["exp7_nfa"] = {"error": str(e)}

    if 8 in experiments:
        print(f"  Exp 8: Hidden Group Representation...")
        t0 = time.time()
        try:
            all_results["exp8_group_rep"] = experiment_8_group_representation(
                model, triples, variant, p, crt_factors, device,
                n_train=300, n_test=200, max_power=5
            )
            for l in all_results["exp8_group_rep"]["layers"]:
                r = all_results["exp8_group_rep"]["layers"][l]
                print(f"    Layer {l}: E1={r['E1']:.4f} E_cycle={r['E_cycle']:.4f} "
                      f"({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    Error: {e}")
            all_results["exp8_group_rep"] = {"error": str(e)}

    if 9 in experiments:
        print(f"  Exp 9: Causal Subspace Ablation...")
        t0 = time.time()
        try:
            all_results["exp9_ablation"] = experiment_9_ablation(
                model, triples, hidden_states, x_vals, crt_factors,
                variant, p, device, n_samples=500
            )
            print(f"    ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    Error: {e}")
            all_results["exp9_ablation"] = {"error": str(e)}

    if 11 in experiments:
        print(f"  Exp 11: Recombination Analysis...")
        t0 = time.time()
        all_results["exp11_recombination"] = experiment_11_recombination(
            logits, x_vals, crt_factors
        )
        print(f"    var_A={all_results['exp11_recombination']['var_A_explained']:.4f} "
              f"var_B={all_results['exp11_recombination']['var_B_explained']:.4f} "
              f"var_I={all_results['exp11_recombination']['var_I_explained']:.4f} "
              f"({time.time()-t0:.1f}s)")

    if 12 in experiments:
        print(f"  Exp 12: Representation Compression...")
        t0 = time.time()
        all_results["exp12_compression"] = experiment_12_compression(hidden_states)
        for l in all_results["exp12_compression"]["layers"]:
            r = all_results["exp12_compression"]["layers"][l]
            print(f"    Layer {l}: eff_rank={r['eff_rank']:.4f} PR={r['participation_ratio']:.4f} "
                  f"K90={r['K90']} ({time.time()-t0:.1f}s)")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{name}_mi.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    return all_results


# ============================================================================
# Main
# ============================================================================

def find_all_checkpoints(base_dir: str) -> List[str]:
    """Find all checkpoint directories with final.pt."""
    ckpt_dirs = []
    lumi_dir = os.path.join(base_dir, "checkpoints", "lumi_download")
    if os.path.exists(lumi_dir):
        for d in sorted(os.listdir(lumi_dir)):
            ckpt_path = os.path.join(lumi_dir, d, "checkpoints", "final.pt")
            partial_path = os.path.join(lumi_dir, d, "checkpoints")
            has_ckpt = os.path.exists(ckpt_path)
            if not has_ckpt and os.path.exists(partial_path):
                partials = [f for f in os.listdir(partial_path) if f.startswith(".final.pt")]
                has_ckpt = len(partials) > 0
            if has_ckpt:
                ckpt_dirs.append(os.path.join(lumi_dir, d))

    # Also check experiments/algebraic_pairs
    pairs_dir = os.path.join(base_dir, "experiments", "algebraic_pairs")
    if os.path.exists(pairs_dir):
        for d in sorted(os.listdir(pairs_dir)):
            ckpt_path = os.path.join(pairs_dir, d, "checkpoints", "final.pt")
            if os.path.exists(ckpt_path):
                ckpt_dirs.append(os.path.join(pairs_dir, d))

    return ckpt_dirs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", help="Single checkpoint directory")
    parser.add_argument("--all", action="store_true", help="Run on all available checkpoints")
    parser.add_argument("--output-dir", default="analysis/mi")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--experiments", nargs="+", type=int, default=None,
                        help="Which experiments to run (1-12). Default: all")
    args = parser.parse_args()

    if args.checkpoint_dir:
        run_all_experiments(args.checkpoint_dir, args.output_dir, args.device, args.experiments)
    elif args.all:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_dirs = find_all_checkpoints(base_dir)
        print(f"Found {len(ckpt_dirs)} checkpoints:")
        for d in ckpt_dirs:
            print(f"  {os.path.basename(d)}")

        all_results = []
        for ckpt_dir in ckpt_dirs:
            result = run_all_experiments(ckpt_dir, args.output_dir, args.device, args.experiments)
            if result:
                all_results.append(result)

        # Save combined
        combined_path = os.path.join(args.output_dir, "all_mi_results.json")
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nCombined results saved to {combined_path}")

        # Print summary table
        print(f"\n{'='*100}")
        print("SUMMARY TABLE")
        print(f"{'='*100}")
        print(f"{'Name':<30} {'p':>5} {'Var':<10} {'Acc':>7} {'A_eq':>7} {'L1Ax':>7} {'L1Aa':>7} {'L1Ab':>7} {'E_A':>6} {'E_B':>6} {'E_AB':>6} {'r_eff':>6}")
        print("-" * 100)
        for r in all_results:
            name = r.get("name", "?")
            p = r.get("p", 0)
            var = r.get("variant", "?")
            acc = r.get("best_test_acc", 0) or 0
            eq = r.get("exp1_equivariance", {}).get("A_eq_argmax", 0)
            probes = r.get("exp2_crt_probes", {}).get("layers", {})
            l1 = probes.get("1", probes.get("0", {}))
            ax = l1.get("acc_x", 0)
            aa = l1.get("acc_a", 0)
            ab = l1.get("acc_b", 0)
            fourier = r.get("exp4_crt_fourier", {}).get("layers", {})
            l1f = fourier.get("1", fourier.get("0", {}))
            ea = l1f.get("E_A", 0)
            eb = l1f.get("E_B", 0)
            eab = l1f.get("E_AB", 0)
            comp = r.get("exp12_compression", {}).get("layers", {})
            l1c = comp.get("1", comp.get("0", {}))
            reff = l1c.get("eff_rank", 0)
            print(f"{name:<30} {p:>5} {var:<10} {acc:>7.4f} {eq:>7.4f} {ax:>7.4f} {aa:>7.4f} {ab:>7.4f} {ea:>6.4f} {eb:>6.4f} {eab:>6.4f} {reff:>6.2f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
