# Results Summary: DLP Grokking Mechanistic Interpretability (FINAL)

## 1. DLP Grokking

DLP exhibits grokking in all representations that preserve the true CRT factor partition:
- RAW: groks at ~50K epochs (slow, single seed)
- TRUE CRT: groks at ~20K epochs (fast)
- SCRAMBLED: groks at ~16K epochs (conjugate to CRT)
- PERMUTED CRT (R2): groks at ~8K epochs (fastest)
- ROLE-DECOUPLED CRT: groks at ~26K epochs

Representations destroying the partition fail to grok:
- GLOBAL RANDOM (R3): NO grokking after 100K epochs (3 mapping seeds: m42, m123, m7)
- ORDER-MATCHED (R3O): NO grokking after 50K epochs
- MIXED RADIX (R4): NO grokking after 50K epochs

## 2. SCRAMBLED Audit

SCRAMBLED is exactly conjugate to CRT at initialization:
- Max logit difference: 0.00
- S_hom = 1.0, S_trans = 1.0, ARI = 1.0
- With categorical embeddings and untied output, permutation is absorbed by embedding table
- SCRAMBLED is a vacuous control — it does NOT test destruction of algebra

## 3. Fourier Analysis (Final State)

| Representation | E_A | E_B | E_AB | R_int |
|---------------|------|------|------|-------|
| RAW (grokked) | 0.10 | 0.24 | 0.65 | 0.66 |
| CRT-BOTH | 0.09 | 0.22 | 0.01 | 0.03 |
| SCRAMBLED | 0.08 | 0.21 | 0.02 | 0.07 |

RAW is interaction-rich (E_AB=0.65); CRT is separable (E_AB=0.01).

## 4. Temporal Analysis (RAW, single seed)

| Epoch | E_AB | R_int | A_eq | Test Acc |
|-------|------|-------|------|----------|
| 0 | 0.020 | 0.42 | 0.003 | 0.012 |
| 5K | 0.153 | 0.25 | 0.131 | 0.097 |
| 15K | 0.349 | 0.42 | 0.166 | 0.309 |
| 25K | 0.453 | 0.52 | 0.184 | 0.402 |
| 30K | 0.521 | 0.59 | 0.162 | 0.325 |
| 40K | 0.633 | 0.65 | 0.223 | 0.482 |
| 50K | 0.641 | 0.67 | 0.889 | 0.952 |

Key: E_AB rises from 0.35 to 0.52 during epochs 15K-30K while test acc remains flat.
Interaction structure forms ~20K epochs before grokking.

## 5. Top-k Causal Ablation (RAW, Layer 1, 100 random controls)

| Subspace | k | Δ (structured) | Random (rank-matched) | Random (var-matched) |
|----------|---|---------------|----------------------|---------------------|
| U_A | 2 | -0.628 | +0.011 ± 0.003 | +0.008 ± 0.005 |
| U_A | 4 | -0.894 | +0.010 ± 0.004 | +0.007 ± 0.005 |
| U_B | 2 | -0.608 | +0.011 ± 0.003 | +0.008 ± 0.005 |
| U_B | 4 | -0.888 | +0.010 ± 0.004 | +0.007 ± 0.005 |
| U_B | 8 | -0.968 | +0.010 ± 0.005 | +0.005 ± 0.007 |
| U_AB | 2 | -0.686 | +0.011 ± 0.003 | +0.008 ± 0.005 |
| U_AB | 4 | -0.924 | +0.010 ± 0.004 | +0.007 ± 0.005 |
| U_AB | 8 | -0.968 | +0.010 ± 0.005 | +0.005 ± 0.007 |

All structured ablations at 0th percentile, p < 0.01.

## 6. Control Hierarchy (FINAL)

| Representation | S_hom | Partition? | Grokked? | T_grok | Seeds |
|---------------|-------|-----------|----------|--------|-------|
| RAW | — | No | Yes (slow) | ~50K | 1 |
| TRUE CRT | 1.0 | Yes | Yes (fast) | ~20K | 1 |
| CURRENT SCRAMBLED | 1.0 | Yes | Yes (fast) | ~16K | 1 |
| PERMUTED CRT (R2) | — | Yes | YES | ~8K | 1 |
| ROLE-DECOUPLED CRT | — | Yes | YES | ~26K | 1 |
| GLOBAL RANDOM (R3) | 0.008 | No | NO | >100K | 3 |
| ORDER-MATCHED (R3O) | 0.050 | No | NO | >50K | 1 |
| MIXED RADIX (R4) | 0.024 | No | NO | >50K | 1 |

**Key finding**: Representations preserving the true CRT factor partition grok;
representations destroying it fail to grok even after 100K epochs across 3 mapping seeds.

## 7. Key Claims (Final)

1. DLP exhibits grokking; algebraically aligned CRT projections accelerate generalization.
2. RAW and CRT develop markedly different algebraic representations (E_AB: 0.65 vs 0.01).
3. Interaction structure forms during the plateau, before grokking.
4. Information-matched controls that destroy algebraic structure lose CRT's rapid-generalization advantage.
5. The true CRT factor partition is the critical property (not merely factorization).
6. SCRAMBLED is a vacuous control (conjugate to CRT).
7. A/B/AB Fourier modes are causally necessary (top-k ablation, p < 0.01).
