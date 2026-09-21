#!/usr/bin/env python3
"""
Generate figures for the CRT Grokking AISTATS paper from experiment metrics.

Usage:
  python generate_crt_figures.py --input-dir experiments/crt_intervention --output-dir paper/figures
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ============================================================================
# Load metrics
# ============================================================================

def load_metrics(run_dir: str) -> List[Dict]:
    """Load metrics.jsonl from a run directory."""
    path = os.path.join(run_dir, "metrics.jsonl")
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_config(run_dir: str) -> Dict:
    """Load config.json from a run directory."""
    with open(os.path.join(run_dir, "config.json")) as f:
        return json.load(f)


def load_summary(run_dir: str) -> Dict:
    """Load summary.json from a run directory."""
    with open(os.path.join(run_dir, "summary.json")) as f:
        return json.load(f)


def get_run_dir(input_dir: str, condition: str, p: int, seed: int) -> str:
    return os.path.join(input_dir, f"{condition}_p{p}_s{seed}")


# ============================================================================
# Figure 1: The killer figure — 4 panels
# ============================================================================

def fig1_main(input_dir: str, output_dir: str, p: int = 113, seed: int = 42):
    """
    Figure 1: The entire empirical story in one figure.

    Panel A: Mathematical representation (schematic)
    Panel B: Three matched learning curves (RAW, RAW+HALF, RAW+FULL)
    Panel C: Component accuracies A_a(t), A_b(t)
    Panel D: Fiber mass M_a(t), M_b(t)
    """
    conditions = {
        "RAW": ("#E74C3C", "RAW (no decomposition)"),
        "RAW+CRT-A": ("#F39C12", "RAW + CRT-A (half)"),
        "RAW+CRT-BOTH": ("#2ECC71", "RAW + FULL CRT"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- Panel A: Schematic (top-left) ---
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(A) CRT Decomposition of DLP', fontsize=12, fontweight='bold')

    # Draw the decomposition tree
    ax.text(5, 9.5, r'$g^x \equiv h \pmod{p}$', ha='center', fontsize=14,
            bbox=dict(boxstyle='round', facecolor='#EBF5FB', edgecolor='#3498DB'))
    ax.annotate('', xy=(2.5, 7.5), xytext=(5, 8.8),
                arrowprops=dict(arrowstyle='->', color='#7F8C8D'))
    ax.annotate('', xy=(7.5, 7.5), xytext=(5, 8.8),
                arrowprops=dict(arrowstyle='->', color='#7F8C8D'))
    ax.text(2.5, 7, r'$g_a^{x \bmod a} \equiv h_a$', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='#D5F5E3', edgecolor='#2ECC71'))
    ax.text(7.5, 7, r'$g_b^{x \bmod b} \equiv h_b$', ha='center', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='#D5F5E3', edgecolor='#2ECC71'))
    ax.text(5, 5.5, r'$x \leftrightarrow (x \bmod a,\; x \bmod b)$', ha='center', fontsize=11)
    ax.text(5, 4, f'p={p}, q={p-1}\nCRT factors: see config', ha='center', fontsize=10,
            style='italic', color='#7F8C8D')

    # --- Panel B: Learning curves (top-right) ---
    ax = axes[0, 1]
    ax.set_title('(B) Test Accuracy: RAW vs HALF vs FULL', fontsize=12, fontweight='bold')

    for cond, (color, label) in conditions.items():
        run_dir = get_run_dir(input_dir, cond, p, seed)
        if not os.path.exists(run_dir):
            continue
        records = load_metrics(run_dir)
        epochs = [r["epoch"] for r in records]
        test_acc = [r["test_acc"] for r in records]
        ax.plot(epochs, test_acc, color=color, label=label, linewidth=2)

    ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5, label='Grokking threshold')
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='center right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel C: Component accuracies (bottom-left) ---
    ax = axes[1, 0]
    ax.set_title('(C) CRT Component Accuracies', fontsize=12, fontweight='bold')

    for cond, (color, label) in conditions.items():
        run_dir = get_run_dir(input_dir, cond, p, seed)
        if not os.path.exists(run_dir):
            continue
        records = load_metrics(run_dir)
        epochs = [r["epoch"] for r in records]
        acc_a = [r["test_acc_a"] for r in records]
        acc_b = [r["test_acc_b"] for r in records]
        ax.plot(epochs, acc_a, color=color, linewidth=2, linestyle='-',
                label=f'{label}: $A_a$')
        ax.plot(epochs, acc_b, color=color, linewidth=2, linestyle='--',
                label=f'{label}: $A_b$')

    ax.axhline(y=1/7, color='#F39C12', linestyle=':', alpha=0.3)
    ax.axhline(y=1/16, color='#2ECC71', linestyle=':', alpha=0.3)
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Component Accuracy')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='center right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- Panel D: Fiber mass (bottom-right) ---
    ax = axes[1, 1]
    ax.set_title('(D) CRT Fiber Mass', fontsize=12, fontweight='bold')

    for cond, (color, label) in conditions.items():
        run_dir = get_run_dir(input_dir, cond, p, seed)
        if not os.path.exists(run_dir):
            continue
        records = load_metrics(run_dir)
        epochs = [r["epoch"] for r in records]
        mass_a = [r["test_fiber_mass_a"] for r in records]
        mass_b = [r["test_fiber_mass_b"] for r in records]
        ax.plot(epochs, mass_a, color=color, linewidth=2, linestyle='-',
                label=f'{label}: $M_a$')
        ax.plot(epochs, mass_b, color=color, linewidth=2, linestyle='--',
                label=f'{label}: $M_b$')

    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('Fiber Mass $M$')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='center right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig1_main.pdf')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


# ============================================================================
# Figure 2: All conditions comparison
# ============================================================================

def fig2_all_conditions(input_dir: str, output_dir: str, p: int = 113, seed: int = 42):
    """Bar chart comparing all conditions."""
    all_conditions = [
        ("RAW", "#E74C3C"),
        ("RAW+CRT-BOTH", "#2ECC71"),
        ("RAW+CRT-A", "#F39C12"),
        ("RAW+CRT-B", "#E67E22"),
        ("CRT-BOTH", "#27AE60"),
        ("CRT-A", "#95A5A6"),
        ("CRT-B", "#7F8C8D"),
        ("RAW+NULL", "#BDC3C7"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    names = []
    accs = []
    t90s = []
    colors = []

    for cond, color in all_conditions:
        run_dir = get_run_dir(input_dir, cond, p, seed)
        if not os.path.exists(run_dir):
            continue
        summary = load_summary(run_dir)
        names.append(cond)
        accs.append(summary["best_test_acc"] * 100)
        t90 = summary.get("t_90")
        t90s.append(t90 if t90 is not None else 100000)
        colors.append(color)

    x = range(len(names))
    bars = ax1.bar(x, accs, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.set_ylabel('Best Test Accuracy (%)')
    ax1.set_title('Best Accuracy by Condition')
    ax1.axhline(y=95, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add ceiling lines
    config_path = os.path.join(get_run_dir(input_dir, "CRT-A", p, seed), "config.json")
    if os.path.exists(config_path):
        config = load_config(get_run_dir(input_dir, "CRT-A", p, seed))
        ceiling_a = config["info_ceiling_a"] * 100
        ceiling_b = config["info_ceiling_b"] * 100
        for i, name in enumerate(names):
            if name == "CRT-A":
                ax1.plot([i-0.3, i+0.3], [ceiling_a, ceiling_a], 'r--', linewidth=2)
                ax1.annotate(f'ceiling={ceiling_a:.1f}%', (i, ceiling_a),
                            textcoords='offset points', xytext=(0, 5), ha='center', fontsize=7, color='red')
            elif name == "CRT-B":
                ax1.plot([i-0.3, i+0.3], [ceiling_b, ceiling_b], 'r--', linewidth=2)
                ax1.annotate(f'ceiling={ceiling_b:.1f}%', (i, ceiling_b),
                            textcoords='offset points', xytext=(0, 5), ha='center', fontsize=7, color='red')

    # T_90 comparison (log scale)
    ax2.bar(x, t90s, color=colors, edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('$T_{90}$ (epochs)')
    ax2.set_title('Grokking Time by Condition')
    ax2.set_yscale('log')
    ax2.set_ylim(100, 200000)
    ax2.grid(True, alpha=0.3, axis='y')

    for i, t in enumerate(t90s):
        if t < 100000:
            ax2.annotate(f'{t:,}', (i, t), textcoords='offset points', xytext=(0, 5),
                        ha='center', fontsize=8)
        else:
            ax2.annotate('no grok', (i, t), textcoords='offset points', xytext=(0, 5),
                        ha='center', fontsize=8, color='red')

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig2_all_conditions.pdf')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


# ============================================================================
# Figure 3: Component accuracy detail for RAW+HALF conditions
# ============================================================================

def fig3_component_detail(input_dir: str, output_dir: str, p: int = 113, seed: int = 42):
    """Detailed component accuracy analysis for the partial-structure trap."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    conditions = [
        ("RAW", "#E74C3C"),
        ("RAW+CRT-A", "#F39C12"),
        ("RAW+CRT-B", "#E67E22"),
        ("RAW+CRT-BOTH", "#2ECC71"),
    ]

    # Panel 1: A_a(t)
    ax = axes[0]
    ax.set_title('$A_a$: Accuracy of $\\hat{x} \\bmod a$ vs $x \\bmod a$', fontsize=11)
    for cond, color in conditions:
        run_dir = get_run_dir(input_dir, cond, p, seed)
        if not os.path.exists(run_dir):
            continue
        records = load_metrics(run_dir)
        epochs = [r["epoch"] for r in records]
        acc_a = [r["test_acc_a"] for r in records]
        ax.plot(epochs, acc_a, color=color, linewidth=2, label=cond)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('$A_a$')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: A_b(t)
    ax = axes[1]
    ax.set_title('$A_b$: Accuracy of $\\hat{x} \\bmod b$ vs $x \\bmod b$', fontsize=11)
    for cond, color in conditions:
        run_dir = get_run_dir(input_dir, cond, p, seed)
        if not os.path.exists(run_dir):
            continue
        records = load_metrics(run_dir)
        epochs = [r["epoch"] for r in records]
        acc_b = [r["test_acc_b"] for r in records]
        ax.plot(epochs, acc_b, color=color, linewidth=2, label=cond)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('$A_b$')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: A_full(t)
    ax = axes[2]
    ax.set_title('$A_{full}$: Full Test Accuracy', fontsize=11)
    for cond, color in conditions:
        run_dir = get_run_dir(input_dir, cond, p, seed)
        if not os.path.exists(run_dir):
            continue
        records = load_metrics(run_dir)
        epochs = [r["epoch"] for r in records]
        acc_full = [r["test_acc"] for r in records]
        ax.plot(epochs, acc_full, color=color, linewidth=2, label=cond)
    ax.axhline(y=0.95, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('$A_{full}$')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig3_component_detail.pdf')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.savefig(path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="experiments/crt_intervention")
    parser.add_argument("--output-dir", default="paper/figures")
    parser.add_argument("--p", type=int, default=113)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fig1_main(args.input_dir, args.output_dir, args.p, args.seed)
    fig2_all_conditions(args.input_dir, args.output_dir, args.p, args.seed)
    fig3_component_detail(args.input_dir, args.output_dir, args.p, args.seed)

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()
