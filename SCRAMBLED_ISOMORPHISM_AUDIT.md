# SCRAMBLED ISOMORPHISM AUDIT

## Critical Finding

**CURRENT SCRAMBLED is EXACTLY CONJUGATE to CRT at initialization.**

Max logit difference between conjugate CRT and SCRAMBLED models: **0.00e+00**

---

## Audit 11: Vocabulary-Isomorphism Test

### Construction

Current SCRAMBLED applies a single fixed permutation π: G → G to all
four CRT positions:
```
(g_b, g_a, h_b, h_a) → (π(g_b), π(g_a), π(h_b), π(h_a))
```

For a CRT model with embedding matrix E, construct a SCRAMBLED model
with E'[tok(π(y))] = E[tok(y)] for every group-element token. Keep
every other parameter identical.

### Result

The two models produce **identical outputs** for corresponding inputs.
Max logit difference = 0.00e+00 (machine precision).

### Why This Happens

1. The model uses learned categorical embeddings (nn.Embedding)
2. Token values have no numeric meaning (no positional/scalar encoding)
3. Input embeddings and output classifier are UNTIED
4. The same permutation is applied to ALL input positions
5. The permutation is a bijection (preserves equality relationships)

Therefore, permuting input embedding rows exactly absorbs the token
permutation. The output classifier is unaffected.

---

## Audit 12: Training-Trajectory Conjugacy

Since the models are conjugate at initialization and use:
- Identical underlying minibatches (same triples, same split)
- Identical optimizer (AdamW)
- Identical learning rate
- Identical weight decay schedule
- Identical architecture

The training trajectories should remain equivalent up to the
permutation symmetry, PROVIDED that:
- The permutation doesn't interact with dropout (it doesn't, since
  dropout is applied to hidden states, not tokens)
- The permutation doesn't interact with weight decay (it doesn't,
  since WD is applied to parameters, not activations)

**Conclusion**: Current SCRAMBLED is primarily a vocabulary-relabeling
control. Any behavioral similarity between SCRAMBLED and CRT is
EXPECTED, not surprising. SCRAMBLED does NOT test destruction of
algebraic structure.

---

## Audit 13: Equality Structure Preserved by SCRAMBLED

Because the same global permutation is applied everywhere:

### Token Equality
u = v ⟺ π(u) = π(v) — **PRESERVED** (bijection)

### Factor Partition
A-channel values stay in A-channel, B-channel values stay in B-channel
— **PRESERVED** (permutation is within-channel)

### Subgroup Support Intersection
Original: |sub_A ∩ sub_B| = 1 (only identity element)
Scrambled: |π(sub_A) ∩ π(sub_B)| = 1 — **PRESERVED** (bijection preserves
intersection cardinality)

### Relational Structure
All token equality relationships across positions are preserved:
- g = h (if and only if π(g) = π(h))
- g_a = h_a (if and only if π(g_a) = π(h_a))
- Cross-position equality (g_b = h_b etc.) preserved

**Conclusion**: SCRAMBLED preserves MORE algebraic structure than the
label "destroyed algebra" suggests. It is essentially a relabeling.

---

## Algebraic Structure Scores

| Representation | S_hom | S_trans | ARI_A | ARI_B |
|---------------|------:|--------:|------:|------:|
| TRUE CRT | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CURRENT SCRAMBLED | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| RANDOM BIJECTION (R3) | 0.008 | 0.047 | 0.005 | --- |

SCRAMBLED has IDENTICAL structure scores to TRUE CRT. It does not
destroy any algebraic structure.

---

## Implications for the Paper

1. The previous claim that "SCRAMBLED refutes algebraic alignment" is
   **INVALID** — SCRAMBLED preserves all algebraic structure.

2. The speed of SCRAMBLED grokking is **expected** — it is the same
   representation as CRT with relabeled symbols.

3. A TRUE negative control must use a representation that is NOT
   conjugate to CRT. This requires either:
   - A different mapping per position (breaks the single-permutation
     conjugacy)
   - A mapping that destroys the factor partition (R3: Global Random
     Bijection)
   - A mapping that destroys the group law while preserving the
     partition (R2: Permuted CRT with separate per-factor permutations)

4. The R3 (Global Random Bijection) control with S_hom ≈ 0.008 is the
   correct negative control for testing whether algebraic alignment
   matters.
