# Experimental Protocol: Mechanistic Interpretability of DLP Grokking

## Overview

This document specifies the full experimental protocol for mechanistic
interpretability analysis of Discrete Logarithm Problem (DLP) grokking and
CRT-accelerated generalization. Each experiment is specified precisely enough
to implement without interpretation.

## Primary Models

- **RAW**: Model receives original DLP representation `y = g^x`, trained to
  approximate `f_theta(y) = x`.
- **FULL-CRT**: Same mathematical DLP problem, same dataset, matched
  architecture, but representation includes CRT factorization
  `x <-> (r_a, r_b) = (x mod a, x mod b)`.

## Baseline Protocol (applies to all experiments)

### Dataset
For each example, store:
- exponent `x`
- group element `y = g^x`
- `x mod a` (CRT component A)
- `x mod b` (CRT component B)
- train/test membership
- input features
- CRT representation (if applicable)

All mechanistic analyses must be able to index examples by exact exponent `x`,
which is the natural algebraic coordinate.

### Matched Seeds
RAW and CRT trained with same seed list. For each seed, match:
- initialization RNG
- dataset split
- minibatch ordering
- optimizer, learning rate, weight decay
- training length

Differences due to input representation are documented explicitly.

### Checkpoints
Dense checkpointing:
- initialization
- early training
- before memorization
- directly after memorization
- multiple checkpoints during plateau
- dense around grokking
- after grokking

For each checkpoint: `theta_t`, train loss, test loss, train acc, test acc,
weight norms, gradient norms.

---

## Experiment 1: DLP Equivariance

### Research Question
Has the model learned the cyclic group operation behind DLP?

For `y = g^x` and arbitrary `c in Z_N`:
```
g^c * y = g^{x+c}
log_g(g^c * y) = x + c (mod N)
```

The true DLP function is equivariant under `y -> g^c * y`.

### 1A: Argmax Equivariance
For checkpoint `t`:
1. Choose test example `y_i`
2. Choose random `c`
3. Compute `x_hat_i = argmax f_{theta_t}(y_i)`
4. Transform input: `y_i' = g^c * y_i (mod p)`
5. Compute `x_hat_i' = argmax f_{theta_t}(y_i')`
6. Check: `x_hat_i' == x_hat_i + c (mod N)`

```
A_eq(t) = (1/M) * sum_i 1[x_hat_i' == x_hat_i + c_i (mod N)]
```

### 1B: Soft Equivariance (KL Divergence)
Let `z(y)` be logits, `p(y) = softmax(z(y))`.
Let `Q_c` be the permutation matrix shifting output classes by `c`:
`(Q_c * z)_j = z_{j-c}`.

```
E_eq_KL(t) = E_{y,c} [ D_KL( p(g^c * y) || Q_c * p(y) ) ]
```

Perfect equivariance: `E_eq -> 0`.

### 1C: Equivariance Through Training
Compute `A_eq(t)` and `E_eq(t)` for every checkpoint.
Plot with `A_train(t)` and `A_test(t)`.
Look for: `A_eq` rising before `A_test`.

### 1D: RAW vs CRT Comparison
Compare `A_eq^RAW(t)` with `A_eq^CRT(t)`.

Three possible outcomes:
- **A**: CRT becomes equivariant immediately, RAW slowly develops it.
- **B**: Both become equivariant at similar times, CRT generalizes earlier.
- **C**: RAW groks without strong equivariance (hypothesis needs revision).

---

## Experiment 2: CRT Residue Probing

### Research Question
Does the RAW model spontaneously build the algebraic subproblems that
FULL-CRT gets explicitly?

### Hidden Representation Extraction
For each checkpoint `t`, layer `l`, example `i`:
```
h_{l,t}(y_i)
```

Data format: seed, condition, checkpoint, layer, sample_id, x, r_a, r_b,
hidden_vector.

### Three Probes
For each `(l, t)`, train separate linear probes:
- **Probe A**: `h_{l,t} -> r_a` (x mod a)
- **Probe B**: `h_{l,t} -> r_b` (x mod b)
- **Probe X**: `h_{l,t} -> x` (full discrete log)

Implementation: multinomial logistic regression (linear, no hidden layer).
Regularization via probe-validation set. No massive hyperparameter search.

### Controls
For each probe:
- **Shuffled target**: permute labels
- **Random network**: same probe on untrained model representations
- **Random projection**: project hidden vectors to same dimension as
  identified subspaces

### Selectivity
```
S_a = A_a^{true} - A_a^{shuffle}
S_b = A_b^{true} - A_b^{shuffle}
S_x = A_x^{true} - A_x^{shuffle}
```

### Key Question for RAW
If RAW truly discovers CRT internally:
```
A_a^{probe}(t) and A_b^{probe}(t) rise before A_x^{native}(t)
```
The ordering is decisive.

### Heatmaps
For each quantity `A_a(l,t)`, `A_b(l,t)`, `A_x(l,t)`:
- x-axis: training step
- y-axis: layer depth
- color: probe selectivity

---

## Experiment 3: Representation vs Readout

### Research Question
Has the model already learned the solution internally before the output head
can use it?

### Native Accuracy
```
A_native(t) = standard model output accuracy
```

### Frozen Representation Readout
Freeze entire backbone. Train new linear classifier:
```
W * h_t(y) -> x
```
Measure `A_{linear-x}(t)`. Same for `r_a` and `r_b`.

### Interpretation
- **Scenario 1**: `A_{linear-x} ~ A_{native}` — representation and output
  develop together.
- **Scenario 2**: `A_{linear-x} >> A_{native}` during plateau — solution
  is linearly extractable but model's readout hasn't aligned.
- **Scenario 3**: Residues are linearly available (`A_a, A_b >> chance`)
  but full `A_x` is low — factorization without recombination.

### Readout Gap
```
G_readout(t) = A_{linear-x}(t) - A_{native}(t)
```
Large positive gap: feature representation advanced further than native
classifier usage.

---

## Experiment 4: 2D CRT Fourier Analysis

### Research Question
How is the hidden representation organized algebraically?

### Reindex by CRT
For hidden dimension/neuron `j`:
```
H_j[r_a, r_b] = h_j(x(r_a, r_b))
```
Dimension: `a x b`.

### 2D DFT
```
H_hat_j(k_a, k_b) = sum_{r_a, r_b} H_j(r_a, r_b) * exp(-2*pi*i*(k_a*r_a/a + k_b*r_b/b))
```

### Four Spectral Categories
- **DC**: `(k_a, k_b) = (0, 0)`
- **A-only**: `k_a != 0, k_b = 0`
- **B-only**: `k_a = 0, k_b != 0`
- **Interaction**: `k_a != 0, k_b != 0`

### Energies
```
E_A = sum_{k_a != 0} |H_hat(k_a, 0)|^2
E_B = sum_{k_b != 0} |H_hat(0, k_b)|^2
E_AB = sum_{k_a != 0, k_b != 0} |H_hat(k_a, k_b)|^2
```
Normalized: `E_tilde_i = E_i / E_total`.

### Population Analysis
Per neuron, per layer, per checkpoint.
Summarize with median: `E_tilde_A(l,t)`, `E_tilde_B(l,t)`, `E_tilde_AB(l,t)`.

### Spectral Entropy
```
p_k = |H_hat(k)|^2 / sum_j |H_hat(j)|^2
H_spec = -sum_k p_k * log(p_k)
```

### Participation Ratio
```
PR = 1 / sum_k p_k^2
```

### Expected RAW Trajectory (hypothesis)
- Early: diffuse spectrum
- Plateau: `E_A, E_B` begin increasing
- Before grokking: clear A and B components
- At grokking: `E_AB` rises when components recombine

---

## Experiment 5: Character Basis vs Raw Basis

### Research Question
In which basis is the representation sparsest?

### Three Coordinate Systems
1. **Raw indexing**: natural order
2. **DLP/exponent indexing**: `x = 0, ..., N-1`
3. **CRT factor coordinates**: `(r_a, r_b)`

### Group Characters
For `Z_N`:
```
chi_k(x) = exp(2*pi*i*k*x/N)
```

For each hidden feature `h_j(x)`:
```
c_{j,k} = (1/N) * sum_x h_j(x) * conj(chi_k(x))
```

### Basis Complexity Comparison
For each basis, measure:
- spectral entropy
- top-K reconstruction (`K_90 = min K: sum_{i=1}^K E_{(i)} / sum_j E_j >= 0.9`)
- participation ratio
- Gini coefficient

Lower `K_90` = more compact algebraic representation.

---

## Experiment 6: AGOP (Average Gradient Outer Product)

### Research Question
How does task-relevant feature geometry develop?

### Jacobian
For input embedding `e(y)`, logits `z(e)`:
```
J(e) = d z(e) / d e
```

For large output dimensions, use:
- vector-Jacobian products
- random projections of logits
- Jacobian sketching

Document the approximation.

### AGOP
```
G_t = (1/M) * sum_{i=1}^M J_t(e_i)^T * J_t(e_i)
```
Result: `G_t in R^{d x d}`.

### AGOP Spectrum
Eigenvalues `lambda_1 >= lambda_2 >= ...`.
Measure:
- effective rank
- eigenvalue entropy
- top-K explained variance

### Group-Action Alignment
If group action in input embedding is `P_c`:
```
E_AGOP-group = E_c [ ||P_c G_t P_c^T - G_t||_F / ||G_t||_F ]
```

### RAW vs CRT AGOP
- Does CRT reach low-complexity feature metric earlier?
- Does group alignment improve earlier?
- Does RAW converge to the same feature metric later?

---

## Experiment 7: NFA (Neural Feature Ansatz) Alignment

### Research Question
Is grokking associated with alignment between learned weights and
task-gradient geometry?

### Weight Gram Matrix
For first learned weight matrix `W_1`:
```
M_W = W_1^T * W_1
```

### AGOP Transform
For depth `L`:
```
M_A = G_t^{1/L}
```
Via eigendecomposition: `G = Q Lambda Q^T`, `G^{1/L} = Q Lambda^{1/L} Q^T`.

### Alignment Metric
```
rho_NFA = <M_W_tilde, M_A_tilde>_F
```
where `M_tilde = M / ||M||_F`.

### Track Over Training
Plot `rho_NFA(t)` with train acc, test acc, equivariance, CRT probe.

### Weight Decay Experiment
- Fixed WD baseline
- Progressive WD (existing schedule)
- Track `rho_NFA(t)` for both

---

## Experiment 8: Hidden Group Representation

### Research Question
Do hidden states carry a representation of the cyclic group?

### Group Action
For generator `T_1(y) = g*y`, find `R_l` such that:
```
h_l(g*y) ~ R_l * h_l(y)
```

### Learn R_l
Collect pairs `(h_l(y_i), h_l(g*y_i))` from training subset.
```
R_l* = argmin_R sum_i ||h_l(g*y_i) - R*h_l(y_i)||^2 + lambda*||R||_F^2
```

### Test on Held-Out Samples
```
E_1 = ||h_l(g*y) - R_l*h_l(y)|| / ||h_l(g*y)||
```

### Test Powers
```
h_l(g^c * y) ~ R_l^c * h_l(y)
E_c = ||h_l(g^c * y) - R_l^c * h_l(y)|| / ||h_l(g^c * y)||
```

### Group Law
```
E_law = E_{c,d} [ ||R_{c+d} - R_c * R_d||_F / ||R_{c+d}||_F ]
```

### Cyclic Closure
```
E_cycle = ||R_1^N - I||_F / ||I||_F
```

### Eigenvalue Analysis
For exact unitary representation of cyclic group, eigenvalues lie on roots
of unity: `exp(2*pi*i*k/N)`.
Study spectrum of `R_l`. Do eigenphases approach `2*pi*k/N`?

---

## Experiment 9: Causal Subspace Ablation

### Research Question
Are the identified CRT subspaces necessary for the computation?

### Identify U_A and U_B
Two methods:
1. **Probe directions**: span of `W_A^T` (probe weights)
2. **Spectral/CCA directions**: directions maximally associated with CRT-A/B

### Projection Matrices
Orthonormalize: `Q_A`, `P_A = Q_A * Q_A^T`. Same for B.

### Ablation A
```
h' = h - P_A * h
```
Measure: `A_x`, `A_a`, `A_b`.
Hypothesis: removing A-space damages `A_a` and possibly full DLP.

### Ablation B
Analogous.

### Joint Ablation
```
h' = h - (P_A + P_B) * h  (if orthogonal)
```

### Random Subspace Control
Random subspace with same dimension, matched activation variance, 100 repeats.
Compare `Delta A_structured` vs random distribution.

### Dose-Response
```
h' = h - alpha * P_A * h,  alpha in {0, 0.25, 0.5, 0.75, 1.0}
```

---

## Experiment 10: Subspace Patching

### Research Question
Are the identified subspaces sufficient for the computation?

### CRT to RAW
Take `h_R` from pre-grok RAW, `h_C` from grokked CRT.
Align representations. Patch:
```
h'_R = h_R + P_A * (h_C - h_R)
```
Test: A only, B only, A+B.

### Within-RAW Temporal Patching
Early checkpoint `t_1 < t_g`, late checkpoint `t_2 > t_g`.
```
h_{t_1}' = h_{t_1} + P_U * (h_{t_2} - h_{t_1})
```

### Reverse Patching Control
Patch grokked model with pre-grok representation. Performance should fall.

---

## Experiment 11: Recombination Analysis

### Research Question
Where is full `x` created from CRT components?

### Layer-wise Transition
For each layer: `A_a(l)`, `A_b(l)`, `A_x(l)`.
Look for: `A_a, A_b` high in early layers, `A_x` high only later.

### Interaction Spectral Energy
Test if `E_AB` increases where full-x decodability begins.

### Logit ANOVA
For output logit to class `x`, view as function `L(r_a, r_b)`:
```
L(r_a, r_b) = mu + A(r_a) + B(r_b) + I(r_a, r_b)
```
Measure variance explained by A, B, interaction I.

---

## Experiment 12: Representation Compression

### Research Question
Does grokking coincide with representation compression?

### SVD
```
H = U Sigma V^T
```

### Effective Rank
```
p_i = sigma_i^2 / sum_j sigma_j^2
r_eff = exp(-sum_i p_i * log(p_i))
```

### Participation Ratio
```
PR = (sum_i sigma_i^2)^2 / sum_i sigma_i^4
```

### Top-K Reconstruction
```
E_K = sum_{i>K} sigma_i^2 / sum_i sigma_i^2
```

---

## Experiment 13: Unified Timeline

### For Each Checkpoint
Collect: `T_train`, `A_test`, `A_eq`, `A_a`, `A_b`, `E_A`, `E_B`, `E_AB`,
`rho_NFA`, `E_AGOP-group`, `r_eff`.

### Normalized Time
```
tau = t / T_grok
```
Use only grokked runs for relative analysis. Keep censored runs separate.

### Event Ordering
Define thresholds for: `T_eq`, `T_A`, `T_B`, `T_spec`, `T_NFA`, `T_g`.
Analyze ordering.

### Progress Measure Prediction
Use early checkpoints to predict `T_grok`:
```
X_t = [A_eq, A_a, A_b, E_A, E_B, rho_NFA]
```
Simple regression on training seeds, test on held-out seeds.

---

## Confirmatory vs Exploratory

### Confirmatory (preregistered)
- RAW vs FULL-CRT: `T_grok`
- DLP equivariance
- CRT-A/B probes
- 2D CRT Fourier
- Readout gap
- At least one causal ablation

### Exploratory (initially)
- NFA alignment
- Full AGOP group analysis
- Hidden group representation
- Patching
- Compression
- CCA
- Detailed group-character decomposition

---

## Minimal World-Class Story

1. `T_CRT << T_RAW`
2. RAW develops `A_eq` long before generalization
3. CRT residues become decodable in RAW before grokking
4. CRT Fourier shows increasing energy in factorized modes
5. Ablation of these modes/subspaces destroys full DLP generalization

Conclusion: RAW grokking coincides with and functionally depends on the
emergence of a representation aligned with DLP group's algebraic structure.

### Stronger Story (if supported)
If `RAW_{late} ~ CRT_{early}` under CKA/subspace/spectral analysis:
"Explicit CRT decomposition accelerates grokking by supplying coordinates
aligned with a representation that the undecomposed model eventually discovers
internally."

### Alternative Strong Result
If RAW does NOT resemble CRT:
"Which algebraic representation does RAW discover instead?"
Character/Fourier analysis becomes central.

---

## Evidence Levels (never mix)

1. **Prediction**: `f(y) = x`
2. **Symmetry understanding**: `f(g^c y) = f(y) + c`
3. **Information representation**: can `x mod a` be decoded?
4. **Structural representation**: is hidden space organized by CRT/characters?
5. **Functional use**: does function disappear under ablation?
6. **Causality**: can function be restored by patching?

---

## Implementation Priority

1. RAW/CRT multi-seed behavioral baseline
2. Dense checkpoints
3. DLP equivariance
4. CRT-A/B/full-X probes
5. Native output vs frozen linear readout
6. 2D CRT Fourier
7. Group-character analysis
8. RAW/CRT CKA
9. Structured subspace identification
10. Causal ablation
11. AGOP
12. NFA
13. Hidden group representation
14. Patching
15. Compression/secondary analyses
