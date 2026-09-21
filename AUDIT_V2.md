# AUDIT V2: Complete Methodological Audit

## Summary

This document reports the complete methodological audit of the DLP grokking
project, covering all representations, training protocols, mechanistic
analyses, and negative controls.

---

## Part I: CRT Invertibility (Audit 1)

**Status**: VERIFIED

For all primes (p=113, 337, 601):
- gcd(a,b) = 1 ✓
- a*b = N ✓
- Zero collisions across all group elements ✓
- Bézout reconstruction works for every element ✓

| Prime | N | a | b | Collisions | Bézout Errors | Invertible |
|-------|---|---|---|-----------|--------------|-----------|
| 113 | 112 | 7 | 16 | 0 | 0 | YES |
| 337 | 336 | 3 | 112 | 0 | 0 | YES |
| 601 | 600 | 3 | 200 | 0 | 0 | YES |

**Conclusion**: CRT-BOTH is information-preserving and correctly invertible.

---

## Part II: Generator Verification (Audit 2)

**Status**: VERIFIED

| Prime | N | Generators Found | φ(N) Expected | Match | All Verified |
|-------|---|-----------------|--------------|-------|-------------|
| 113 | 112 | 48 | 48 | YES | YES |
| 337 | 336 | 96 | 96 | YES | YES |
| 601 | 600 | 160 | 160 | YES | YES |

Every generator has order exactly N. No non-generators included.

---

## Part III: Target Uniqueness (Audit 3)

**Status**: VERIFIED

For every prime, all N triples (g, h, x) satisfy:
- h = g^x mod p ✓
- x is unique (exactly one x gives each h) ✓

No malformed examples found.

---

## Part IV: Tokenization and Embedding Audit (Audits 7-10)

### Audit 7: Token Values Have No Numeric Meaning

**Finding**: Token IDs are purely categorical.

The model uses `nn.Embedding(vocab_size, d_model)` — a learned lookup table.
No positional encoding of token value, no binary digits, no scalar
normalization, no sinusoidal encoding based on token ID, no polynomial
features, no arithmetic operations on token IDs, no ordinal distance.

**Implication**: The numeric labels assigned to group elements have no
inherent metric or arithmetic meaning before training. The model must
learn all structure from the embedding table.

### Audit 8: Weight Tying

**Finding**: Input embeddings and output classifier are NOT tied.

- Input: `nn.Embedding(vocab_size, d_model)` — separate
- Output: `nn.Linear(d_model, vocab_size)` — separate
- No weight sharing

**Implication**: An arbitrary input vocabulary permutation can be absorbed
exactly by permuting embedding rows, WITHOUT affecting the output
classifier. This means SCRAMBLED is exactly conjugate to CRT at
initialization.

### Audit 9: Token Namespace

**Finding**: All tokens share the SAME namespace.

`tok(v) = v + INT_OFFSET` is used for:
- Raw group elements (g, h)
- CRT projection values (g_a, g_b, h_a, h_b)
- SCRAMBLED values
- Output exponent labels (x)

**Critical implication**: Group element 5 and output label 5 share the
SAME learned embedding row. This creates unintended input-output coupling.

### Audit 10: Zero Element Issue

**Finding**: No zero-mappings occur in practice.

SCRAMBLED's permutation is defined over 0..p-1, but CRT projections
never produce 0 (since F_p* is closed under multiplication). The
permutation CAN map to 0 in principle, but in practice with seed 12345,
zero projected values are mapped to 0 in 0 cases.

However, the permutation DOES include 0 in its domain, which means
token 0+INT_OFFSET = 4 is a valid token that may appear in SCRAMBLED
but never in CRT-BOTH. This is a minor vocabulary mismatch.

---

## Part V: SCRAMBLED Isomorphism Test (Audits 11-13)

### Audit 11: Vocabulary-Isomorphism Test

**CRITICAL FINDING**: SCRAMBLED is EXACTLY conjugate to CRT at initialization.

- Same permutation applied to all 4 CRT positions: YES
- Equality relationships preserved: YES (bijection)
- Subgroup intersection cardinality preserved: YES (1 → 1)
- Max logit difference at initialization: **0.00e+00**
- Conjugate at initialization: **YES**

**Construction**: For a CRT model with embedding E, construct a SCRAMBLED
model with E'[tok(π(y))] = E[tok(y)]. The two models produce identical
outputs for corresponding inputs.

### Audit 12: Training-Trajectory Conjugacy

Since the models are conjugate at initialization and use the same
optimizer, learning rate, and weight decay, the training trajectories
should remain equivalent up to the permutation symmetry.

**Conclusion**: Current SCRAMBLED is primarily a vocabulary-relabeling
control and must NOT be presented as evidence that algebraic structure
is unnecessary.

### Audit 13: Equality Structure Preserved

SCRAMBLED preserves:
- Token equality relationships (u=v ⟺ π(u)=π(v))
- Factor partition (A-channel values stay in A-channel)
- Subgroup support intersection cardinality
- All relational structure between tokens

**Conclusion**: SCRAMBLED preserves MORE algebraic structure than the
label "destroyed algebra" suggests. It is essentially a relabeling.

---

## Part VI: Algebraic Structure Scores (Audits 14-16)

| Representation | S_hom | S_trans | ARI_A | ARI_B |
|---------------|------:|--------:|------:|------:|
| TRUE CRT | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| CURRENT SCRAMBLED | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| RANDOM BIJECTION (R3) | 0.0083±0.0011 | 0.0474±0.0009 | 0.0053±0.0168 | — |
| MIXED RADIX (R4) | 0.0235 | 0.0327 | 0.0089 | -0.0131 |
| ORDER-MATCHED RANDOM (R3O) | 0.0499±0.0022 | 0.0749±0.0029 | — | — |

**Key findings**:
1. TRUE CRT and CURRENT SCRAMBLED have IDENTICAL structure scores (all 1.0)
2. R3 (RANDOM BIJECTION) destroys the group law (S_hom ≈ 0.008)
3. R4 (MIXED RADIX) destroys the group law (S_hom ≈ 0.024)
4. R3O (ORDER-MATCHED) destroys the multiplication table but preserves
   element orders (S_hom ≈ 0.050, higher than R3 due to order preservation)

**Conclusion**: CURRENT SCRAMBLED does NOT destroy algebraic structure.
R3 and R4 are the true algebra-destroying controls.

---

## Answers to Final Audit Questions

### Q1: Is current SCRAMBLED genuinely different from CRT?

**NO.** SCRAMBLED is exactly conjugate to CRT at initialization (max logit
diff = 0.00). It preserves all algebraic structure scores (S_hom=1.0,
S_trans=1.0, ARI=1.0). It is merely a vocabulary relabeling.

### Q2: Does TRUE CRT preserve information without target leakage?

**YES.** CRT-BOTH uses input-derived subgroup projections h^(N/a) and
h^(N/b), computed from the public group element h. The target x is never
used in the representation. Bézout reconstruction confirms invertibility.

### Q3: Does GLOBAL RANDOM BIJECTION destroy the group law?

**YES.** R3 has S_hom = 0.008 (vs 1.0 for CRT), S_trans = 0.047,
ARI ≈ 0. It preserves information and interface but destroys algebra.

### Q4: Are behavioral differences robust after matching?

**PARTIALLY TESTED.** Sequence length and parameter counts are matched.
Token support is matched. Compute normalization needs further work.

### Q5: Are mechanistic differences robust?

**PARTIALLY TESTED.** Fourier and ablation results are from final
checkpoints. Generator-dimension and centering robustness need testing.

### Q6: Can any claim about algebraic alignment survive negative controls?

**PENDING.** Requires R3/R3O/R4 training results. The previous claim
that "SCRAMBLED refutes algebraic alignment" is INVALID because SCRAMBLED
preserves all algebraic structure.
