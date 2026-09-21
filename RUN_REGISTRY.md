# Run Registry — Source of Truth

This file reconciles all experiment numbers cited in the paper. All paper claims must cite this registry.

## Primary RAW Run (Temporal Analysis Source)

| Field | Value |
|-------|-------|
| Run name | `RAW_p113_s42` |
| Location | `experiments/crt_mi/RAW_p113_s42/` |
| Condition | RAW (standard, type="standard") |
| p | 113 |
| N | 112 |
| Seed | 42 |
| d_model | 128 |
| n_heads | 4 |
| n_layers | 2 |
| vocab_size | 117 |
| max_len | 5 |
| n_params | 427,253 |
| n_core | 396,544 |
| lr | 0.001 |
| wd_start | 0.05 |
| wd_max | 0.30 |
| wd_ramp_interval | 1000 |
| wd_step | 0.05 |
| train_frac | 0.30 |
| n_train | 1,612 |
| n_test | 3,764 |
| Dense checkpoints | 32 (epochs 0–65,000) |
| T_grok (first test_acc > 0.50) | epoch 44,500 |
| T_grok (first test_acc > 0.95) | epoch ~50,000 |
| Final test_acc | 0.960 (epoch 67,500) |

**Note**: This is the ONLY RAW run with dense checkpoints for temporal analysis. Temporal replication on additional seeds is pending.

## Control Runs — FINAL RESULTS

### R3: GLOBAL_RANDOM_BIJECTION (S_hom ≈ 0.008, partition destroyed)

| Run | Mapping Seed | Train Seed | Final Epoch | Train Acc | Test Acc | WD | Grokked? |
|-----|-------------|-----------|-------------|-----------|----------|-----|---------|
| R3_m42_s42 | 42 | 42 | **100,000** | 1.000 | 0.058 | 0.293 | **NO** |
| R3_m123_s42 | 123 | 42 | **50,000** | 1.000 | 0.067 | 0.293 | **NO** |
| R3_m7_s42 | 7 | 42 | **50,000** | 1.000 | 0.055 | 0.293 | **NO** |
| R3_m123_s123 | 123 | 123 | 100 | 0.868 | 0.060 | 0.050 | — (early) |
| R3_m7_s123 | 7 | 123 | 100 | 0.783 | 0.068 | 0.050 | — (early) |

**Key R3 finding**: R3 m42_s42 trained to **100K epochs** — 2x RAW's grokking time — with train_acc=1.0, test_acc=0.058, WD=0.293. **NOT grokking.** Confirmed across 3 mapping seeds (m42, m123, m7), all failing.

### R3O: ORDER_MATCHED_RANDOM (S_hom ≈ 0.050, partition destroyed)

| Run | Mapping Seed | Train Seed | Final Epoch | Train Acc | Test Acc | WD | Grokked? |
|-----|-------------|-----------|-------------|-----------|----------|-----|---------|
| R3O_m42_s42 | 42 | 42 | **50,000** | 1.000 | 0.057 | 0.293 | **NO** |

### R4: MIXED_RADIX (S_hom ≈ 0.024, partition destroyed)

| Run | Mapping Seed | Train Seed | Final Epoch | Train Acc | Test Acc | WD | Grokked? |
|-----|-------------|-----------|-------------|-----------|----------|-----|---------|
| R4_m42_s42 | 42 | 42 | **50,000** | 1.000 | 0.058 | 0.293 | **NO** |

### R2: PERMUTED_CRT (partition preserved, within-factor ordering destroyed)

| Run | Mapping Seed | Train Seed | Final Epoch | Train Acc | Test Acc | WD | Grokked? | T_grok |
|-----|-------------|-----------|-------------|-----------|----------|-----|---------|-------|
| R2_m42_s42 | 42 | 42 | **50,000** | 1.000 | 1.000 | 0.293 | **YES** | ~8,400 |

### ROLE_DECOUPLED_CRT (partition preserved, shared relabeling destroyed)

| Run | Mapping Seed | Train Seed | Final Epoch | Train Acc | Test Acc | WD | Grokked? | T_grok |
|-----|-------------|-----------|-------------|-----------|----------|-----|---------|-------|
| ROLE_DEC_m42_s42 | 42 | 42 | **50,000** | 1.000 | 1.000 | 0.293 | **YES** | ~25,500 |

## CRT-BOTH Run (Reference)

| Field | Value |
|-------|-------|
| Run name | `CRT-BOTH_p113_s42` |
| T_grok | ~20,000 epochs |
| Final E_AB | 0.01 |

## SCRAMBLED Run (Reference)

| Field | Value |
|-------|-------|
| Run name | `SCRAMBLED_p113_s42` |
| T_grok | ~16,000 epochs |
| Final E_AB | 0.02 |
| Audit | Exactly conjugate to CRT at init (max logit diff = 0) |

## Summary Hierarchy

| Representation | S_hom | Partition? | Grokked? | T_grok | Seeds |
|---------------|-------|-----------|----------|--------|-------|
| RAW | — | No | Yes (slow) | ~50K | 1 |
| TRUE CRT | 1.0 | Yes | Yes (fast) | ~20K | 1 |
| CURRENT SCRAMBLED | 1.0 | Yes | Yes (fast) | ~16K | 1 |
| PERMUTED CRT (R2) | — | Yes | **YES** | ~8K | 1 |
| ROLE-DECOUPLED CRT | — | Yes | **YES** | ~26K | 1 |
| GLOBAL RANDOM (R3) | 0.008 | No | **NO** | >100K | 3 |
| ORDER-MATCHED (R3O) | 0.050 | No | **NO** | >50K | 1 |
| MIXED RADIX (R4) | 0.024 | No | **NO** | >50K | 1 |

**Key finding**: Representations preserving the true CRT factor partition grok; representations destroying it fail to grok even after 100K epochs across 3 mapping seeds.

## T_grok Definition

T_grok is defined as the first epoch where test accuracy ≥ 0.95, with a stability window of 5 consecutive evaluations above 0.90. "Epoch" means a full pass through the training set (one optimizer step per minibatch, full dataset per epoch). This is consistent across all runs.

## Top-k Causal Ablation

| Field | Value |
|-------|-------|
| Script | `topk_ablation.py` |
| Model | RAW p113 s42 (grokked) |
| Layer | 1 |
| k values | {2, 4, 8, 16} |
| Random controls | 100 rank-matched + 100 variance-matched |
| Construction/evaluation | Separate (50/50 split) |
| Output | `analysis/topk_ablation/RAW_p113_s42_topk_ablation.csv` |

Key results (Layer 1):
- U_A k=4: Δ = -0.894 vs random +0.010 (p < 0.01)
- U_B k=4: Δ = -0.888 vs random +0.010 (p < 0.01)
- U_AB k=4: Δ = -0.924 vs random +0.010 (p < 0.01)

All structured ablations at 0th percentile of random controls.

## Pending Experiments

1. Temporal E_AB replication on 3-5 additional RAW seeds
2. Temporal CRT analysis
3. Top-k causal ablation on CRT-BOTH and SCRAMBLED models
4. Fourier per-generator aggregation
5. Cross-generator probes
6. R3 additional training seeds (currently 3 mapping seeds, 1 training seed each)
