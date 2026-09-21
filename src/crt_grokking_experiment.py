#!/usr/bin/env python3
"""
CRT Grokking Experiment — AISTATS Paper 1 (Mechanistic Interpretability Focus)

Implements:
  C0: RAW              — no decomposition
  C1: RAW + FULL CRT   — raw + both CRT components
  C4: FULL CRT ONLY    — both CRT components, no raw

Saves dense checkpoints for mechanistic analysis:
  - Model state dicts at predefined fractions of expected T_grok
  - Hidden representations at each checkpoint
  - Component accuracies A_a, A_b, A_full throughout training
  - Fiber mass M_a, M_b

Usage:
  python crt_grokking_experiment.py --conditions RAW RAW+CRT-BOTH CRT-BOTH --seeds 42 123 456 --max-epochs 100000
"""

import argparse
import json
import math
import os
import random
import time
from itertools import product as iproduct
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

# ============================================================================
# Constants
# ============================================================================
PAD, SEP, BOS, EOS = 0, 1, 2, 3
INT_OFFSET = 4
NULL_TOKEN = INT_OFFSET

p_global = 113


# ============================================================================
# CRT Math
# ============================================================================

def coprime_factorization(q: int) -> Tuple[int, int]:
    """Factor q into two coprime factors (a, b) with a <= b."""
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
        raise ValueError(f"q={q} has only one prime factor; no coprime split possible.")

    best = None
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


def random_balanced_partition(p: int, n_buckets: int, seed: int = 12345) -> Dict[int, int]:
    rng = random.Random(seed)
    vals = list(range(p))
    rng.shuffle(vals)
    partition = {}
    for i, v in enumerate(vals):
        partition[v] = i % n_buckets
    return partition


# ============================================================================
# Encoding
# ============================================================================

def encode_condition(g: int, h: int, x: int, condition: str,
                    p: int, crt_factors: Tuple[int, int],
                    rand_partition: Optional[Dict[int, int]] = None) -> Tuple[List[int], int]:
    a, b = crt_factors
    g_a = project_factor(g, p, a)
    g_b = project_factor(g, p, b)
    h_a = project_factor(h, p, a)
    h_b = project_factor(h, p, b)

    def tok(val: int) -> int:
        return val + INT_OFFSET

    target = tok(x)

    if condition == "RAW":
        tokens = [BOS, tok(g), SEP, tok(h), EOS]
    elif condition == "RAW+CRT-BOTH":
        tokens = [BOS, tok(g), SEP, tok(h), SEP,
                  tok(g_b), SEP, tok(g_a), SEP,
                  tok(h_b), SEP, tok(h_a), EOS]
    elif condition == "RAW+CRT-A":
        tokens = [BOS, tok(g), SEP, tok(h), SEP,
                  tok(g_a), SEP, tok(h_a), EOS]
    elif condition == "RAW+CRT-B":
        tokens = [BOS, tok(g), SEP, tok(h), SEP,
                  tok(g_b), SEP, tok(h_b), EOS]
    elif condition == "CRT-BOTH":
        tokens = [BOS, tok(g_b), SEP, tok(g_a), SEP,
                  tok(h_b), SEP, tok(h_a), EOS]
    elif condition == "CRT-A":
        tokens = [BOS, tok(g_a), SEP, tok(h_a), EOS]
    elif condition == "CRT-B":
        tokens = [BOS, tok(g_b), SEP, tok(h_b), EOS]
    elif condition == "RAW+NULL":
        tokens = [BOS, tok(g), SEP, tok(h), SEP,
                  NULL_TOKEN, SEP, NULL_TOKEN, EOS]
    elif condition == "RAW+RAND-A":
        if rand_partition is None:
            rand_partition = random_balanced_partition(p, a, seed=12345)
        q_g = rand_partition.get(g, 0)
        q_h = rand_partition.get(h, 0)
        tokens = [BOS, tok(g), SEP, tok(h), SEP,
                  tok(q_g), SEP, tok(q_h), EOS]
    elif condition == "RAW+RAND-B":
        if rand_partition is None:
            rand_partition = random_balanced_partition(p, b, seed=54321)
        q_g = rand_partition.get(g, 0)
        q_h = rand_partition.get(h, 0)
        tokens = [BOS, tok(g), SEP, tok(h), SEP,
                  tok(q_g), SEP, tok(q_h), EOS]
    else:
        raise ValueError(f"Unknown condition: {condition}")

    return tokens, target


# ============================================================================
# Dataset
# ============================================================================

def generate_dlp_problems(p: int) -> List[Tuple[int, int, int]]:
    """Generate all (g, h, x) triples using ALL primitive roots."""
    triples = []
    for g in range(2, p):
        if _is_primitive_root(g, p):
            for x in range(p - 1):
                h = pow(g, x, p)
                triples.append((g, h, x))
    return triples


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


class CRTDataset:
    def __init__(self, triples: List[Tuple[int, int, int]], condition: str,
                 p: int, crt_factors: Tuple[int, int], max_len: int,
                 rand_partition: Optional[Dict[int, int]] = None):
        self.examples = []
        for g, h, x in triples:
            tokens, target = encode_condition(g, h, x, condition, p, crt_factors, rand_partition)
            padded = tokens + [PAD] * (max_len - len(tokens))
            padded = padded[:max_len]
            self.examples.append((
                torch.tensor(padded, dtype=torch.long),
                target,
                x,
            ))

    def __len__(self):
        return len(self.examples)


def split_triples(triples: List[Tuple[int, int, int]], n_train: int, seed: int = 42):
    rng = random.Random(seed)
    indices = list(range(len(triples)))
    rng.shuffle(indices)
    train_idx = set(indices[:n_train])
    train = [triples[i] for i in sorted(train_idx)]
    test = [triples[i] for i in indices[n_train:]]
    return train, test


# ============================================================================
# Model
# ============================================================================

class GrokkingTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, max_len: int = 13, dropout: float = 0.0):
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

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# Evaluation with Component Accuracies
# ============================================================================

def evaluate_model(model, dataset, device, crt_factors, p, batch_size=512,
                   return_hidden: bool = False):
    model.eval()
    a, b = crt_factors

    correct_full = 0
    correct_a = 0
    correct_b = 0
    total = 0

    fiber_mass_a_list = []
    fiber_mass_b_list = []

    all_hidden = {l: [] for l in range(model.n_layers)} if return_hidden else None
    all_x_vals = [] if return_hidden else None

    # Precompute fiber masks
    a_fiber_masks = {}
    b_fiber_masks = {}
    for r in range(a):
        mask = torch.zeros(p + INT_OFFSET, device=device)
        for v in range(p):
            if v % a == r:
                mask[v + INT_OFFSET] = 1.0
        a_fiber_masks[r] = mask
    for s in range(b):
        mask = torch.zeros(p + INT_OFFSET, device=device)
        for v in range(p):
            if v % b == s:
                mask[v + INT_OFFSET] = 1.0
        b_fiber_masks[s] = mask

    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch = [dataset.examples[j] for j in range(i, min(i + batch_size, len(dataset)))]
            tokens = torch.stack([x[0] for x in batch]).to(device)
            targets = torch.stack([torch.tensor(x[1], dtype=torch.long) for x in batch]).to(device)
            x_vals = torch.tensor([x[2] for x in batch], device=device)

            if return_hidden:
                logits, hidden_states = model(tokens, return_hidden=True)
                for l, h in enumerate(hidden_states):
                    # Get hidden at EOS position
                    seq_lens = (tokens != PAD).sum(dim=1) - 1
                    idx = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, h.size(-1))
                    h_at_pos = h.gather(1, idx).squeeze(1)
                    all_hidden[l].append(h_at_pos.cpu())
                all_x_vals.extend(x_vals.cpu().tolist())
            else:
                logits = model(tokens)

            seq_lens = (tokens != PAD).sum(dim=1) - 1
            idx_pos = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, logits.size(-1))
            logits_at_pos = logits.gather(1, idx_pos).squeeze(1)

            preds = logits_at_pos.argmax(dim=-1)
            probs = torch.softmax(logits_at_pos, dim=-1)

            pred_vals = preds - INT_OFFSET
            target_vals = targets - INT_OFFSET

            correct_full += (preds == targets).sum().item()
            correct_a += ((pred_vals % a) == (x_vals % a)).sum().item()
            correct_b += ((pred_vals % b) == (x_vals % b)).sum().item()
            total += len(batch)

            for j in range(len(batch)):
                x_val = x_vals[j].item()
                m_a = (probs[j] * a_fiber_masks[x_val % a]).sum().item()
                m_b = (probs[j] * b_fiber_masks[x_val % b]).sum().item()
                fiber_mass_a_list.append(m_a)
                fiber_mass_b_list.append(m_b)

    result = {
        "acc_full": correct_full / max(total, 1),
        "acc_a": correct_a / max(total, 1),
        "acc_b": correct_b / max(total, 1),
        "fiber_mass_a": float(np.mean(fiber_mass_a_list)) if fiber_mass_a_list else 0.0,
        "fiber_mass_b": float(np.mean(fiber_mass_b_list)) if fiber_mass_b_list else 0.0,
        "n_eval": total,
    }

    if return_hidden:
        for l in all_hidden:
            all_hidden[l] = torch.cat(all_hidden[l], dim=0)
        result["hidden_states"] = all_hidden
        result["x_vals"] = all_x_vals

    return result


# ============================================================================
# Checkpoint Schedule
# ============================================================================

def make_checkpoint_schedule(t_grok_estimate: int, max_epochs: int) -> List[int]:
    """Create a dense checkpoint schedule based on estimated grokking time.
    Uses fractions of t_grok for dense coverage around the transition."""
    if t_grok_estimate is None or t_grok_estimate <= 0:
        t_grok_estimate = 30000  # Default estimate for RAW

    fractions = [
        0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4,
        0.6, 0.7, 0.8, 0.85, 0.90, 0.925, 0.95, 0.975,
        1.0, 1.025, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0
    ]

    checkpoints = set()
    for f in fractions:
        ep = int(f * t_grok_estimate)
        if 0 <= ep <= max_epochs:
            checkpoints.add(ep)

    # Also add regular intervals for safety
    for ep in range(0, min(max_epochs, t_grok_estimate * 3), 5000):
        checkpoints.add(ep)

    return sorted(checkpoints)


# ============================================================================
# Training Loop with Checkpoints
# ============================================================================

def train_condition(
    condition: str,
    p: int,
    crt_factors: Tuple[int, int],
    seed: int,
    max_epochs: int = 100000,
    train_frac: float = 0.3,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 2,
    lr: float = 1e-3,
    wd_start: float = 0.05,
    wd_max: float = 0.30,
    wd_ramp_interval: int = 1000,
    wd_step: float = 0.05,
    eval_interval: int = 500,
    output_dir: str = "results",
    device_str: str = "auto",
    early_stop_patience: int = 200,
    early_stop_threshold: float = 0.99,
    t_grok_estimate: int = 30000,
    save_checkpoints: bool = True,
    save_hidden: bool = True,
):
    global p_global
    p_global = p

    a, b = crt_factors
    print(f"\n{'='*60}")
    print(f"Condition: {condition} | p={p} | q={p-1} | factors=({a},{b}) | seed={seed}")
    print(f"{'='*60}")

    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")

    triples = generate_dlp_problems(p)
    n_train = min(int(len(triples) * train_frac), len(triples) - 1)
    train_triples, test_triples = split_triples(triples, n_train, seed=seed)
    print(f"Total triples: {len(triples)} | Train: {len(train_triples)} | Test: {len(test_triples)}")

    rand_partition = None
    if "RAND" in condition:
        rand_seed = 12345 if "RAND-A" in condition else 54321
        bucket_size = a if "RAND-A" in condition else b
        rand_partition = random_balanced_partition(p, bucket_size, seed=rand_seed)

    sample_tokens, _ = encode_condition(
        train_triples[0][0], train_triples[0][1], train_triples[0][2],
        condition, p, crt_factors, rand_partition)
    max_len = len(sample_tokens)
    print(f"Sequence length: {max_len}")

    train_ds = CRTDataset(train_triples, condition, p, crt_factors, max_len, rand_partition)
    test_ds = CRTDataset(test_triples, condition, p, crt_factors, max_len, rand_partition)

    vocab_size = p + INT_OFFSET
    print(f"Vocab size: {vocab_size}")

    torch.manual_seed(seed)
    model = GrokkingTransformer(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, max_len=max_len, dropout=0.0
    ).to(device)
    n_params = model.count_parameters()
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd_start)

    train_tokens = torch.stack([ex[0] for ex in train_ds.examples]).to(device)
    train_targets = torch.tensor([ex[1] for ex in train_ds.examples], dtype=torch.long).to(device)

    run_name = f"{condition}_p{p}_s{seed}"
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    hidden_dir = os.path.join(run_dir, "hidden_states")
    os.makedirs(hidden_dir, exist_ok=True)

    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    metrics_file = open(metrics_path, "w")

    config = {
        "condition": condition,
        "p": p,
        "q": p - 1,
        "crt_factors": list(crt_factors),
        "alpha_a": math.log(a) / math.log(p - 1),
        "alpha_b": math.log(b) / math.log(p - 1),
        "seed": seed,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "lr": lr,
        "wd_start": wd_start,
        "wd_max": wd_max,
        "wd_ramp_interval": wd_ramp_interval,
        "wd_step": wd_step,
        "train_frac": train_frac,
        "n_train": len(train_triples),
        "n_test": len(test_triples),
        "max_epochs": max_epochs,
        "eval_interval": eval_interval,
        "vocab_size": vocab_size,
        "max_len": max_len,
        "n_params": n_params,
        "device": str(device),
        "info_ceiling_a": 1.0 / b,
        "info_ceiling_b": 1.0 / a,
        "t_grok_estimate": t_grok_estimate,
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Checkpoint schedule
    ckpt_epochs = make_checkpoint_schedule(t_grok_estimate, max_epochs)
    ckpt_set = set(ckpt_epochs)
    print(f"Checkpoint epochs: {ckpt_epochs[:10]}...{ckpt_epochs[-5:]} ({len(ckpt_epochs)} total)")

    best_test_acc = 0.0
    best_epoch = 0
    t_mem = None
    t_50 = t_90 = t_95 = t_99 = None
    no_improve_count = 0
    start_time = time.time()

    for epoch in range(max_epochs):
        if epoch > 0 and epoch % wd_ramp_interval == 0:
            current_wd = min(wd_max, wd_start + wd_step * (epoch // wd_ramp_interval))
            for param_group in optimizer.param_groups:
                param_group["weight_decay"] = current_wd

        model.train()
        optimizer.zero_grad()
        logits = model(train_tokens)
        seq_lens = (train_tokens != PAD).sum(dim=1) - 1
        idx_pos = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, logits.size(-1))
        logits_at_pos = logits.gather(1, idx_pos).squeeze(1)
        loss = nn.functional.cross_entropy(logits_at_pos, train_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Evaluation
        if epoch % eval_interval == 0 or epoch == max_epochs - 1:
            test_metrics = evaluate_model(model, test_ds, device, crt_factors, p)
            train_metrics = evaluate_model(model, train_ds, device, crt_factors, p)

            test_acc = test_metrics["acc_full"]
            train_acc = train_metrics["acc_full"]

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_epoch = epoch
                no_improve_count = 0
            else:
                no_improve_count += 1

            if t_mem is None and train_acc >= 0.995:
                t_mem = epoch
            if t_50 is None and test_acc >= 0.50:
                t_50 = epoch
            if t_90 is None and test_acc >= 0.90:
                t_90 = epoch
            if t_95 is None and test_acc >= 0.95:
                t_95 = epoch
            if t_99 is None and test_acc >= 0.99:
                t_99 = epoch

            elapsed = time.time() - start_time
            record = {
                "epoch": epoch,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_loss": loss.item(),
                "test_acc_a": test_metrics["acc_a"],
                "test_acc_b": test_metrics["acc_b"],
                "train_acc_a": train_metrics["acc_a"],
                "train_acc_b": train_metrics["acc_b"],
                "test_fiber_mass_a": test_metrics["fiber_mass_a"],
                "test_fiber_mass_b": test_metrics["fiber_mass_b"],
                "wd": optimizer.param_groups[0]["weight_decay"],
                "elapsed_s": elapsed,
            }
            metrics_file.write(json.dumps(record) + "\n")
            metrics_file.flush()

            if epoch % (eval_interval * 10) == 0 or epoch < 2000:
                print(f"  ep {epoch:6d} | tr {train_acc:.4f} | te {test_acc:.4f} "
                      f"| A_a {test_metrics['acc_a']:.4f} | A_b {test_metrics['acc_b']:.4f} "
                      f"| M_a {test_metrics['fiber_mass_a']:.3f} | M_b {test_metrics['fiber_mass_b']:.3f}")

            if best_test_acc >= early_stop_threshold and no_improve_count >= early_stop_patience:
                print(f"  Early stop at epoch {epoch} (best={best_test_acc:.4f} @ {best_epoch})")
                break

        # Save checkpoint
        if save_checkpoints and epoch in ckpt_set:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "test_acc": test_acc if epoch % eval_interval == 0 else None,
                "train_acc": train_acc if epoch % eval_interval == 0 else None,
            }, ckpt_path)

            # Save hidden states for probing
            if save_hidden:
                test_metrics = evaluate_model(model, test_ds, device, crt_factors, p,
                                                return_hidden=True)
                for l in range(n_layers):
                    h = test_metrics["hidden_states"][l]
                    torch.save(h, os.path.join(hidden_dir, f"layer{l}_epoch{epoch}.pt"))
                torch.save(torch.tensor(test_metrics["x_vals"]),
                          os.path.join(hidden_dir, f"x_vals_epoch{epoch}.pt"))

            print(f"  [CKPT] Saved checkpoint at epoch {epoch}")

    metrics_file.close()

    summary = {
        "condition": condition,
        "p": p,
        "seed": seed,
        "best_test_acc": best_test_acc,
        "best_epoch": best_epoch,
        "t_mem": t_mem,
        "t_50": t_50,
        "t_90": t_90,
        "t_95": t_95,
        "t_99": t_99,
        "total_epochs": epoch + 1,
        "n_params": n_params,
        "elapsed_s": time.time() - start_time,
        "n_checkpoints": len([f for f in os.listdir(ckpt_dir) if f.endswith('.pt')]),
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Summary: best_acc={best_test_acc:.4f} @ epoch {best_epoch}")
    print(f"  T_mem={t_mem} | T_50={t_50} | T_90={t_90} | T_95={t_95} | T_99={t_99}")
    print(f"  Checkpoints saved: {summary['n_checkpoints']}")
    print(f"  Results saved to {run_dir}")

    return summary


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CRT Grokking Experiment (MI-focused)")
    parser.add_argument("--conditions", nargs="+",
                        default=["RAW", "RAW+CRT-BOTH", "CRT-BOTH"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--p", type=int, default=113)
    parser.add_argument("--max-epochs", type=int, default=100000)
    parser.add_argument("--train-frac", type=float, default=0.3)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd-start", type=float, default=0.05)
    parser.add_argument("--wd-max", type=float, default=0.30)
    parser.add_argument("--output-dir", default="experiments/crt_mi")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--t-grok-estimate", type=int, default=30000,
                        help="Estimated grokking time for checkpoint schedule")
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--no-hidden", action="store_true")
    args = parser.parse_args()

    crt_factors = coprime_factorization(args.p - 1)
    a, b = crt_factors
    print(f"p={args.p}, q={args.p-1}, CRT factors: ({a}, {b})")
    print(f"alpha_a = {math.log(a)/math.log(args.p-1):.3f}")
    print(f"alpha_b = {math.log(b)/math.log(args.p-1):.3f}")

    all_summaries = []
    for condition in args.conditions:
        # Use different t_grok estimate for CRT conditions (much faster)
        t_est = args.t_grok_estimate
        if "CRT" in condition and "RAW" not in condition:
            t_est = max(2000, args.t_grok_estimate // 20)
        elif "RAW+CRT" in condition:
            t_est = max(5000, args.t_grok_estimate // 5)

        for seed in args.seeds:
            summary = train_condition(
                condition=condition, p=args.p, crt_factors=crt_factors, seed=seed,
                max_epochs=args.max_epochs, train_frac=args.train_frac,
                d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
                lr=args.lr, wd_start=args.wd_start, wd_max=args.wd_max,
                eval_interval=args.eval_interval, output_dir=args.output_dir,
                device_str=args.device, t_grok_estimate=t_est,
                save_checkpoints=not args.no_checkpoints,
                save_hidden=not args.no_hidden,
            )
            all_summaries.append(summary)

    print(f"\n{'='*80}")
    print("FINAL RESULTS TABLE")
    print(f"{'='*80}")
    print(f"{'Condition':<20} {'Seed':>5} {'Best Acc':>8} {'T_mem':>7} {'T_50':>7} {'T_90':>7} {'T_95':>7} {'T_99':>7} {'Epochs':>8} {'Ckpts':>6}")
    print("-" * 90)
    for s in all_summaries:
        print(f"{s['condition']:<20} {s['seed']:>5} {s['best_test_acc']:>8.4f} "
              f"{str(s['t_mem'] or '-'):>7} {str(s['t_50'] or '-'):>7} "
              f"{str(s['t_90'] or '-'):>7} {str(s['t_95'] or '-'):>7} "
              f"{str(s['t_99'] or '-'):>7} {s['total_epochs']:>8} {s.get('n_checkpoints',0):>6}")

    with open(os.path.join(args.output_dir, "all_summaries.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)


if __name__ == "__main__":
    main()
