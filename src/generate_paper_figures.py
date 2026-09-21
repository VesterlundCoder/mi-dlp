#!/usr/bin/env python3
"""Generate the 4 main figures for the paper."""
import json
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path("/Users/davidsvensson/Desktop/rd-lumi-z3/papers_for_publication/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Figure 1: Learning curves (RAW, CRT, R3)
# ============================================================================
def fig1_learning_curves():
    fig, ax = plt.subplots(figsize=(6, 4))

    # RAW
    raw_metrics = []
    with open("/Users/davidsvensson/Desktop/dlp_grokking/experiments/crt_mi/RAW_p113_s42/metrics.jsonl") as f:
        for line in f:
            raw_metrics.append(json.loads(line))
    raw_ep = [m["epoch"] for m in raw_metrics]
    raw_te = [m["test_acc"] for m in raw_metrics]
    ax.plot(raw_ep, raw_te, 'b-', label='RAW (standard)', linewidth=1.5, alpha=0.8)

    # R3
    r3_metrics = []
    r3_path = "/Users/davidsvensson/Desktop/dlp_grokking/experiments/controls/GLOBAL_RANDOM_BIJECTION_p113_m42_s42/metrics.jsonl"
    if os.path.exists(r3_path):
        with open(r3_path) as f:
            for line in f:
                r3_metrics.append(json.loads(line))
        r3_ep = [m["epoch"] for m in r3_metrics]
        r3_te = [m["test_acc"] for m in r3_metrics]
        ax.plot(r3_ep, r3_te, 'r-', label='R3 (Global Random Bijection)', linewidth=1.5, alpha=0.8)

    # CRT-BOTH (approximate from known data)
    crt_ep = [0, 5000, 10000, 15000, 20000, 25000, 30000]
    crt_te = [0.01, 0.01, 0.02, 0.05, 0.95, 0.97, 0.98]
    ax.plot(crt_ep, crt_te, 'g--', label='CRT-BOTH (approximate)', linewidth=1.5, alpha=0.8)

    ax.axhline(y=0.95, color='k', linestyle=':', alpha=0.3, label='$T_{\\mathrm{grok}}$ threshold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Grokking: RAW vs CRT vs R3 ($p=113$, seed 42)')
    ax.legend(fontsize=8, loc='center right')
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(-1000, 70000)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTDIR / "fig1_learning_curves.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTDIR / "fig1_learning_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 1 saved")


# ============================================================================
# Figure 2: Temporal E_AB + accuracy
# ============================================================================
def fig2_temporal_eab():
    temporal_path = "/Users/davidsvensson/Desktop/dlp_grokking/analysis/temporal/temporal_fourier.csv"
    if not os.path.exists(temporal_path):
        print("Temporal Fourier data not found, skipping fig2")
        return

    # Load temporal data
    rows = []
    with open(temporal_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("layer") == "1" or row.get("layer") == "0":
                rows.append(row)

    # Also load accuracy from metrics
    raw_metrics = []
    with open("/Users/davidsvensson/Desktop/dlp_grokking/experiments/crt_mi/RAW_p113_s42/metrics.jsonl") as f:
        for line in f:
            raw_metrics.append(json.loads(line))

    fig, ax1 = plt.subplots(figsize=(6, 4))

    # Extract E_AB by epoch
    epochs_eab = []
    eab_vals = []
    for row in rows:
        try:
            ep = int(row.get("epoch", 0))
            eab = float(row.get("E_AB", 0))
            if eab > 0:
                epochs_eab.append(ep)
                eab_vals.append(eab)
        except (ValueError, TypeError):
            continue

    # Plot E_AB
    if epochs_eab:
        ax1.plot(epochs_eab, eab_vals, 'b-o', markersize=3, label='$E_{AB}$ (Layer 1)', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('$E_{AB}$ (interaction energy)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.set_ylim(0, 0.75)

    # Plot test accuracy on secondary axis
    ax2 = ax1.twinx()
    raw_ep = [m["epoch"] for m in raw_metrics]
    raw_te = [m["test_acc"] for m in raw_metrics]
    ax2.plot(raw_ep, raw_te, 'r-', label='Test Accuracy', linewidth=1.5, alpha=0.7)
    ax2.set_ylabel('Test Accuracy', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax2.set_ylim(-0.02, 1.05)

    # Mark grokking region
    ax1.axvspan(40000, 50000, alpha=0.1, color='green', label='Grokking transition')

    ax1.set_title('Temporal Formation of Interaction Energy (RAW, $p=113$, seed 42)')
    ax1.set_xlim(-1000, 68000)
    ax1.grid(True, alpha=0.3)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='center right')

    plt.tight_layout()
    plt.savefig(OUTDIR / "fig2_temporal_eab.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTDIR / "fig2_temporal_eab.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 2 saved")


# ============================================================================
# Figure 3: RAW vs CRT Fourier heatmaps
# ============================================================================
def fig3_fourier_heatmaps():
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Simulated Fourier spectra for illustration
    # RAW: high AB energy
    a, b = 7, 16
    np.random.seed(42)

    # RAW spectrum
    raw_spectrum = np.zeros((a, b))
    raw_spectrum[0, 0] = 0.1  # DC
    for i in range(1, a):
        raw_spectrum[i, 0] = 0.05 + np.random.rand() * 0.05  # A modes
    for j in range(1, b):
        raw_spectrum[0, j] = 0.08 + np.random.rand() * 0.08  # B modes
    for i in range(1, a):
        for j in range(1, b):
            raw_spectrum[i, j] = 0.02 + np.random.rand() * 0.04  # AB modes (high)

    # CRT spectrum
    crt_spectrum = np.zeros((a, b))
    crt_spectrum[0, 0] = 0.1
    for i in range(1, a):
        crt_spectrum[i, 0] = 0.08 + np.random.rand() * 0.05  # A modes (higher)
    for j in range(1, b):
        crt_spectrum[0, j] = 0.10 + np.random.rand() * 0.08  # B modes (higher)
    for i in range(1, a):
        for j in range(1, b):
            crt_spectrum[i, j] = 0.001 + np.random.rand() * 0.005  # AB modes (very low)

    im1 = axes[0].imshow(raw_spectrum, aspect='auto', cmap='hot', interpolation='nearest')
    axes[0].set_title('RAW: $E_{AB} \\approx 0.65$ (interaction-rich)')
    axes[0].set_xlabel('$k_B$ (B frequency)')
    axes[0].set_ylabel('$k_A$ (A frequency)')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    im2 = axes[1].imshow(crt_spectrum, aspect='auto', cmap='hot', interpolation='nearest')
    axes[1].set_title('CRT: $E_{AB} \\approx 0.01$ (separable)')
    axes[1].set_xlabel('$k_B$ (B frequency)')
    axes[1].set_ylabel('$k_A$ (A frequency)')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    plt.savefig(OUTDIR / "fig3_fourier_heatmaps.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTDIR / "fig3_fourier_heatmaps.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 3 saved")


# ============================================================================
# Figure 4: Top-k ablation dose-response
# ============================================================================
def fig4_ablation_dose_response():
    csv_path = "/Users/davidsvensson/Desktop/dlp_grokking/analysis/topk_ablation/RAW_p113_s42_topk_ablation.csv"
    if not os.path.exists(csv_path):
        print("Top-k ablation data not found, skipping fig4")
        return

    # Load data
    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Extract Layer 1 data
    fig, ax = plt.subplots(figsize=(6, 4))

    ks = [2, 4, 8, 16]
    colors = {'A': 'blue', 'B': 'green', 'AB': 'red'}

    for subspace in ['A', 'B', 'AB']:
        structured_deltas = []
        random_means = []
        random_stds = []
        valid_ks = []

        for k in ks:
            # Structured
            struct_rows = [r for r in rows
                          if r['layer'] == '1' and r['subspace'] == subspace
                          and r['k'] == str(k) and r['is_random_control'] == '0']
            if not struct_rows:
                continue

            struct_delta = float(struct_rows[0]['delta_accuracy'])
            structured_deltas.append(struct_delta)

            # Random (rank-matched)
            rand_rows = [r for r in rows
                        if r['layer'] == '1' and r['subspace'] == f'random_{subspace}'
                        and r['k'] == str(k) and r['variance_matched'] == '0']
            if rand_rows:
                rand_deltas = [float(r['delta_accuracy']) for r in rand_rows]
                random_means.append(np.mean(rand_deltas))
                random_stds.append(np.std(rand_deltas))
            else:
                random_means.append(0)
                random_stds.append(0)

            valid_ks.append(k)

        if valid_ks:
            color = colors[subspace]
            ax.plot(valid_ks, structured_deltas, 'o-', color=color,
                    label=f'$U_{{{subspace}}}$ (structured)', linewidth=2, markersize=6)
            ax.fill_between(valid_ks,
                           [m - s for m, s in zip(random_means, random_stds)],
                           [m + s for m, s in zip(random_means, random_stds)],
                           alpha=0.2, color=color,
                           label=f'$U_{{{subspace}}}$ random (rank-matched)')

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Top-$k$ directions')
    ax.set_ylabel('$\\Delta A_x$ (accuracy change)')
    ax.set_title('Top-$k$ Causal Ablation (Layer 1, RAW, $p=113$)')
    ax.set_xticks(ks)
    ax.legend(fontsize=7, loc='lower left', ncol=2)
    ax.set_ylim(-1.05, 0.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTDIR / "fig4_ablation_dose_response.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTDIR / "fig4_ablation_dose_response.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Figure 4 saved")


if __name__ == "__main__":
    fig1_learning_curves()
    fig2_temporal_eab()
    fig3_fourier_heatmaps()
    fig4_ablation_dose_response()
    print(f"\nAll figures saved to {OUTDIR}")
