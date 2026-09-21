#!/usr/bin/env python3
"""
Training script for new representation controls:
  R3:  GLOBAL_RANDOM_BIJECTION  (destroys group law, preserves info)
  R3O: ORDER_MATCHED_RANDOM     (preserves element orders, destroys group law)
  R4:  MIXED_RADIX              (non-algebraic two-coordinate)
  R2:  PERMUTED_CRT             (per-factor permutations, preserves partition)

All representations use the SAME architecture, optimizer, and training
protocol as the LUMI trainer to ensure fair comparison.

Usage:
  python train_controls.py --variant GLOBAL_RANDOM_BIJECTION --mapping-seed 42 --train-seed 42
  python train_controls.py --variant MIXED_RADIX --train-seed 42
  python train_controls.py --all
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add parent dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mi_suite import (
    coprime_factorization, project_factor, _is_primitive_root,
    PAD, SEP, BOS, EOS, INT_OFFSET, GrokkingTransformer
)


# ============================================================================
# Number Theory
# ============================================================================

def element_order(y: int, p: int) -> int:
    if y % p == 0:
        return 0
    N = p - 1
    for d in range(1, N + 1):
        if N % d == 0 and pow(y, d, p) == 1:
            return d
    return N


def primitive_roots(p: int) -> List[int]:
    return [g for g in range(2, p) if _is_primitive_root(g, p)]


# ============================================================================
# Representation Maps
# ============================================================================

def build_global_random_bijection(p: int, a: int, b: int,
                                    mapping_seed: int) -> Dict[int, Tuple[int, int]]:
    """R3: Random bijection G -> Z_a x Z_b. Destroys group law."""
    rng = random.Random(mapping_seed)
    N = p - 1
    elements = sorted(range(1, p))  # deterministic order, NOT by exponent
    pairs = [(i % a, i % b) for i in range(N)]
    rng.shuffle(pairs)
    return {elements[i]: pairs[i] for i in range(N)}


def build_order_matched_bijection(p: int, a: int, b: int,
                                    mapping_seed: int) -> Dict[int, Tuple[int, int]]:
    """R3O: Order-matched random bijection. Preserves element orders, destroys group law."""
    rng = random.Random(mapping_seed)
    N = p - 1
    elements = sorted(range(1, p))

    # Compute element orders
    elem_orders = {y: element_order(y, p) for y in elements}

    # Compute coordinate-pair orders
    def coord_order(i: int, j: int) -> int:
        oi = a // math.gcd(i, a) if i > 0 else 1
        oj = b // math.gcd(j, b) if j > 0 else 1
        return int(np.lcm(oi, oj))

    pair_orders = {(i, j): coord_order(i, j) for i in range(a) for j in range(b)}

    # Group by order
    elems_by_order = defaultdict(list)
    for y in elements:
        elems_by_order[elem_orders[y]].append(y)

    pairs_by_order = defaultdict(list)
    for pair, order in pair_orders.items():
        pairs_by_order[order].append(pair)

    # Match within each order class
    bijection = {}
    all_pairs = [(i, j) for i in range(a) for j in range(b)]
    for order in elems_by_order:
        elems = elems_by_order[order]
        pairs = pairs_by_order.get(order, [])
        if len(elems) == len(pairs):
            shuffled = pairs.copy()
            rng.shuffle(shuffled)
            for idx, y in enumerate(elems):
                bijection[y] = shuffled[idx]
        else:
            # Fallback: random assignment from all pairs
            rng.shuffle(all_pairs)
            for idx, y in enumerate(elems):
                bijection[y] = all_pairs[idx % len(all_pairs)]

    return bijection


def build_role_decoupled_crt_maps(p: int, a: int, b: int,
                                    mapping_seed: int) -> Tuple[Dict, Dict, Dict, Dict]:
    """ROLE-DECOUPLED CRT: Different permutations for generator vs result positions.
    Preserves A/B partition, cardinalities, sequence length, information.
    Destroys the shared relabeling conjugacy that made SCRAMBLED trivial.
    """
    rng = random.Random(mapping_seed)
    g = primitive_roots(p)[0]
    g_a = project_factor(g, p, a)
    g_b = project_factor(g, p, b)
    sub_A = [pow(g_a, i, p) for i in range(a)]
    sub_B = [pow(g_b, i, p) for i in range(b)]

    # Four independent permutations: g_A, g_B, h_A, h_B
    perms = {}
    for name, size, sub in [("g_A", a, sub_A), ("g_B", b, sub_B),
                            ("h_A", a, sub_A), ("h_B", b, sub_B)]:
        perm = list(range(size))
        rng.shuffle(perm)
        perms[name] = {sub[i]: sub[perm[i]] for i in range(size)}
    return perms["g_A"], perms["g_B"], perms["h_A"], perms["h_B"]


def build_mixed_radix_map(p: int, a: int, b: int) -> Dict[int, Tuple[int, int]]:
    """R4: Mixed radix by sorted element order. Non-algebraic."""
    elements = sorted(range(1, p))
    elem_to_idx = {y: i for i, y in enumerate(elements)}
    return {y: (elem_to_idx[y] // b, elem_to_idx[y] % b) for y in elements}


def build_permuted_crt_maps(p: int, a: int, b: int,
                              mapping_seed: int) -> Tuple[Dict[int,int], Dict[int,int]]:
    """R2: Per-factor permutations of CRT coordinates. Preserves partition."""
    rng = random.Random(mapping_seed)
    # Build subgroup elements
    g = primitive_roots(p)[0]
    g_a = project_factor(g, p, a)
    g_b = project_factor(g, p, b)
    sub_A = [pow(g_a, i, p) for i in range(a)]
    sub_B = [pow(g_b, i, p) for i in range(b)]

    perm_a = list(range(a))
    perm_b = list(range(b))
    rng.shuffle(perm_a)
    rng.shuffle(perm_b)

    # Map: subgroup element -> permuted subgroup element
    map_a = {sub_A[i]: sub_A[perm_a[i]] for i in range(a)}
    map_b = {sub_B[i]: sub_B[perm_b[i]] for i in range(b)}
    return map_a, map_b


# ============================================================================
# Encoding
# ============================================================================

def encode(g: int, h: int, x: int, variant: str, p: int, a: int, b: int,
           bijection: Dict = None, perm_maps: Tuple = None) -> Tuple[List[int], int]:
    def tok(v): return v + INT_OFFSET
    target = tok(x)

    if variant == "RAW":
        return [BOS, tok(g), SEP, tok(h), EOS], target

    elif variant == "CRT-BOTH":
        g_a = project_factor(g, p, a)
        g_b = project_factor(g, p, b)
        h_a = project_factor(h, p, a)
        h_b = project_factor(h, p, b)
        return [BOS, tok(g_b), SEP, tok(g_a), SEP, tok(h_b), SEP, tok(h_a), EOS], target

    elif variant == "GLOBAL_RANDOM_BIJECTION":
        # Use bijection: y -> (r_a, r_b)
        r_a_g, r_b_g = bijection[g]
        r_a_h, r_b_h = bijection[h]
        return [BOS, tok(r_b_g), SEP, tok(r_a_g), SEP, tok(r_b_h), SEP, tok(r_a_h), EOS], target

    elif variant == "ORDER_MATCHED_RANDOM":
        r_a_g, r_b_g = bijection[g]
        r_a_h, r_b_h = bijection[h]
        return [BOS, tok(r_b_g), SEP, tok(r_a_g), SEP, tok(r_b_h), SEP, tok(r_a_h), EOS], target

    elif variant == "MIXED_RADIX":
        q_g, r_g = bijection[g]
        q_h, r_h = bijection[h]
        return [BOS, tok(q_g), SEP, tok(r_g), SEP, tok(q_h), SEP, tok(r_h), EOS], target

    elif variant == "PERMUTED_CRT":
        map_a, map_b = perm_maps
        g_a = project_factor(g, p, a)
        g_b = project_factor(g, p, b)
        h_a = project_factor(h, p, a)
        h_b = project_factor(h, p, b)
        s_g_a = map_a.get(g_a, g_a)
        s_g_b = map_b.get(g_b, g_b)
        s_h_a = map_a.get(h_a, h_a)
        s_h_b = map_b.get(h_b, h_b)
        return [BOS, tok(s_g_b), SEP, tok(s_g_a), SEP, tok(s_h_b), SEP, tok(s_h_a), EOS], target

    elif variant == "ROLE_DECOUPLED_CRT":
        g_A_map, g_B_map, h_A_map, h_B_map = perm_maps
        g_a = project_factor(g, p, a)
        g_b = project_factor(g, p, b)
        h_a = project_factor(h, p, a)
        h_b = project_factor(h, p, b)
        s_g_a = g_A_map.get(g_a, g_a)
        s_g_b = g_B_map.get(g_b, g_b)
        s_h_a = h_A_map.get(h_a, h_a)
        s_h_b = h_B_map.get(h_b, h_b)
        return [BOS, tok(s_g_b), SEP, tok(s_g_a), SEP, tok(s_h_b), SEP, tok(s_h_a), EOS], target

    else:
        raise ValueError(f"Unknown variant: {variant}")


# ============================================================================
# Dataset
# ============================================================================

class DLPDataset(torch.utils.data.Dataset):
    def __init__(self, triples, variant, p, a, b, max_len,
                 bijection=None, perm_maps=None):
        self.examples = []
        for g, h, x in triples:
            tokens, target = encode(g, h, x, variant, p, a, b, bijection, perm_maps)
            padded = (tokens + [PAD] * (max_len - len(tokens)))[:max_len]
            self.examples.append((
                torch.tensor(padded, dtype=torch.long),
                target,
            ))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def split_specs(triples, n_train, seed=42):
    rng = random.Random(seed)
    shuffled = triples.copy()
    rng.shuffle(shuffled)
    return shuffled[:n_train], shuffled[n_train:]


# ============================================================================
# Training (matches LUMI trainer protocol)
# ============================================================================

def compute_wd_max(n_core, anchor_p_core=393_000, min_wd=0.15, max_wd=0.30):
    """WD max from LUMI trainer formula."""
    wd = min(max_wd, max_wd * math.sqrt(anchor_p_core / max(n_core, 1)))
    return max(min_wd, wd)


def compute_accuracy(model, dataset, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i in range(0, len(dataset), 256):
            batch = dataset.examples[i:i+256]
            tokens = torch.stack([b[0] for b in batch]).to(device)
            targets = torch.tensor([b[1] for b in batch], dtype=torch.long).to(device)
            logits = model(tokens)
            if logits.dim() == 3:
                seq_lens = (tokens != 0).sum(dim=1) - 1
                idx = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, logits.size(-1))
                logits = logits.gather(1, idx).squeeze(1)
            preds = logits.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += len(batch)
    return correct / max(total, 1)


def train_model(variant, p, mapping_seed, train_seed, output_dir,
                device="cpu", epochs=100000, eval_interval=50,
                checkpoint_interval=5000, early_stop_patience=200,
                early_stop_threshold=0.99, train_frac=0.30,
                d_model=128, n_heads=4, n_layers=2, lr=1e-3,
                wd_start=0.05, wd_step=0.05, wd_ramp_interval=1000,
                lr_drop_factor=0.1, grad_clip=1.0, resume=False):

    a, b = coprime_factorization(p - 1)
    N = p - 1
    max_len = 9  # Same as CRT-BOTH

    # Build representation map
    bijection = None
    perm_maps = None
    if variant == "GLOBAL_RANDOM_BIJECTION":
        bijection = build_global_random_bijection(p, a, b, mapping_seed)
    elif variant == "ORDER_MATCHED_RANDOM":
        bijection = build_order_matched_bijection(p, a, b, mapping_seed)
    elif variant == "MIXED_RADIX":
        bijection = build_mixed_radix_map(p, a, b)
    elif variant == "PERMUTED_CRT":
        perm_maps = build_permuted_crt_maps(p, a, b, mapping_seed)
    elif variant == "ROLE_DECOUPLED_CRT":
        perm_maps = build_role_decoupled_crt_maps(p, a, b, mapping_seed)

    # Generate triples
    gens = primitive_roots(p)
    triples = [(g, pow(g, x, p), x) for g in gens for x in range(N)]

    # Split
    n_train = min(int(len(triples) * train_frac), len(triples) - 1)
    train_triples, test_triples = split_specs(triples, n_train, seed=train_seed)

    # Build datasets
    train_ds = DLPDataset(train_triples, variant, p, a, b, max_len, bijection, perm_maps)
    test_ds = DLPDataset(test_triples, variant, p, a, b, max_len, bijection, perm_maps)

    # Vocab size: same as CRT-BOTH (p + INT_OFFSET + 1)
    vocab_size = p + INT_OFFSET + 1  # 118 for p=113

    # Model
    random.seed(train_seed)
    torch.manual_seed(train_seed)
    model = GrokkingTransformer(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, max_len=max_len, dropout=0.0,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_core = n_params - model.embed.weight.numel() - model.pos_embed.weight.numel()
    wd_max = compute_wd_max(n_core)

    name = f"{variant}_p{p}_m{mapping_seed}_s{train_seed}"
    outdir = Path(output_dir) / name
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = outdir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Save config
    config = {
        "name": name, "variant": variant, "p": p,
        "mapping_seed": mapping_seed, "train_seed": train_seed,
        "a": a, "b": b, "N": N,
        "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
        "vocab_size": vocab_size, "max_len": max_len,
        "n_params": n_params, "n_core": n_core,
        "epochs": epochs, "lr": lr,
        "wd_start": wd_start, "wd_max": wd_max, "wd_step": wd_step,
        "wd_ramp_interval": wd_ramp_interval, "lr_drop_factor": lr_drop_factor,
        "train_size": len(train_ds), "test_size": len(test_ds),
        "eval_interval": eval_interval, "checkpoint_interval": checkpoint_interval,
        "early_stop_patience": early_stop_patience,
        "early_stop_threshold": early_stop_threshold,
        "train_frac": train_frac,
    }
    with open(outdir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"[{name}] variant={variant} p={p} a={a} b={b}")
    print(f"[{name}] params={n_params:,} core={n_core:,} wd_max={wd_max:.4f}")
    print(f"[{name}] train={len(train_ds)} test={len(test_ds)}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd_start)
    current_wd = wd_start
    current_lr = lr

    # Pre-load to device
    train_tokens = torch.stack([train_ds.examples[i][0] for i in range(len(train_ds))]).to(device)
    train_targets = torch.tensor([train_ds.examples[i][1] for i in range(len(train_ds))], dtype=torch.long).to(device)

    # Training
    memorized = False
    mem_epoch = -1
    best_test_acc = 0.0
    consecutive_high = 0
    early_stopped = False
    metrics_path = outdir / "metrics.jsonl"
    start_time = time.time()
    start_epoch = 0

    # Resume from latest checkpoint if requested
    if resume and ckpt_dir.exists():
        ckpts = sorted(ckpt_dir.glob("ckpt_epoch_*.pt"),
                       key=lambda f: int(f.stem.split("_")[-1]))
        if ckpts:
            latest = ckpts[-1]
            ckpt = torch.load(latest, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            current_wd = ckpt.get("current_wd", wd_start)
            for pg in optimizer.param_groups:
                pg["weight_decay"] = current_wd
            if ckpt.get("train_acc", 0) > 0.99:
                memorized = True
                mem_epoch = ckpt.get("mem_epoch", start_epoch)
            best_test_acc = ckpt.get("best_test_acc", 0.0)
            print(f"[{name}] Resumed from {latest.name} at epoch {start_epoch}, wd={current_wd:.4f}", flush=True)

    for epoch in range(start_epoch, epochs):
        model.train()
        perm = torch.randperm(len(train_ds), device=device)
        xb = train_tokens[perm]
        yb = train_targets[perm]
        logits = model(xb)

        if logits.dim() == 3:
            seq_lens = (xb != 0).sum(dim=1) - 1
            idx = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, logits.size(-1))
            logits = logits.gather(1, idx).squeeze(1)
        loss = F.cross_entropy(logits, yb)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        avg_loss = loss.item()

        if epoch % eval_interval == 0 or epoch == epochs - 1:
            train_acc = compute_accuracy(model, train_ds, device)
            test_acc = compute_accuracy(model, test_ds, device)

            if not memorized and train_acc > 0.99:
                memorized = True
                mem_epoch = epoch
                print(f"[{name}] MEMORIZED at epoch {epoch}", flush=True)

            if memorized and current_wd < wd_max:
                ramps = (epoch - mem_epoch) // wd_ramp_interval
                target_wd = min(wd_start + (ramps + 1) * wd_step, wd_max)
                if target_wd > current_wd:
                    for pg in optimizer.param_groups:
                        pg["weight_decay"] = target_wd
                        pg["lr"] = current_lr * lr_drop_factor
                    print(f"[{name}] WD ramp: {current_wd:.4f} -> {target_wd:.4f}", flush=True)
                    current_wd = target_wd

            if test_acc > best_test_acc:
                best_test_acc = test_acc

            if epoch % 500 == 0 or test_acc > 0.01:
                elapsed = time.time() - start_time
                print(f"[{name}] Epoch {epoch:6d} | train={train_acc:.4f} test={test_acc:.4f} "
                      f"loss={avg_loss:.4f} wd={current_wd:.4f} t={elapsed:.1f}s", flush=True)

            with open(metrics_path, "a") as f:
                f.write(json.dumps({
                    "epoch": epoch, "train_acc": float(train_acc),
                    "test_acc": float(test_acc), "loss": float(avg_loss),
                    "wd": float(current_wd),
                    "elapsed_s": float(time.time() - start_time),
                    "best_test_acc": float(best_test_acc),
                }) + "\n")

            if test_acc > early_stop_threshold:
                consecutive_high += 1
                if consecutive_high >= early_stop_patience:
                    print(f"[{name}] Early stopped at epoch {epoch}", flush=True)
                    early_stopped = True
            else:
                consecutive_high = 0

        if (epoch + 1) % checkpoint_interval == 0:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "current_wd": current_wd, "best_test_acc": best_test_acc,
                "train_acc": train_acc, "test_acc": test_acc,
            }, ckpt_dir / f"ckpt_epoch_{epoch:06d}.pt")

        if early_stopped:
            break

    # Save final
    torch.save({
        "epoch": epoch, "model_state_dict": model.state_dict(),
        "current_wd": current_wd, "best_test_acc": best_test_acc,
        "train_acc": train_acc, "test_acc": test_acc,
    }, ckpt_dir / "final.pt")

    summary = {
        "name": name, "variant": variant, "p": p,
        "mapping_seed": mapping_seed, "train_seed": train_seed,
        "total_epochs": epoch, "best_test_acc": float(best_test_acc),
        "mem_epoch": mem_epoch, "n_params": n_params,
        "train_size": len(train_ds), "test_size": len(test_ds),
        "early_stopped": early_stopped,
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[{name}] Done. Best test acc: {best_test_acc:.4f}, epochs: {epoch}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["GLOBAL_RANDOM_BIJECTION", "ORDER_MATCHED_RANDOM",
                             "MIXED_RADIX", "PERMUTED_CRT", "ROLE_DECOUPLED_CRT",
                             "RAW", "CRT-BOTH"])
    ap.add_argument("--p", type=int, default=113)
    ap.add_argument("--mapping-seed", type=int, default=42)
    ap.add_argument("--train-seed", type=int, default=42)
    ap.add_argument("--output-dir", default="experiments/controls")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=100000)
    ap.add_argument("--eval-interval", type=int, default=50)
    ap.add_argument("--checkpoint-interval", type=int, default=5000)
    ap.add_argument("--early-stop-patience", type=int, default=200)
    ap.add_argument("--train-frac", type=float, default=0.30)
    ap.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = ap.parse_args()

    train_model(
        variant=args.variant, p=args.p,
        mapping_seed=args.mapping_seed, train_seed=args.train_seed,
        output_dir=args.output_dir, device=args.device,
        epochs=args.epochs, eval_interval=args.eval_interval,
        checkpoint_interval=args.checkpoint_interval,
        early_stop_patience=args.early_stop_patience,
        train_frac=args.train_frac,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
