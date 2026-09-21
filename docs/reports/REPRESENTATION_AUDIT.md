# Representation Audit: DLP Grokking Representations

## Mandatory Audit of All Representations

This document audits every representation used in our DLP grokking experiments.
For each representation we document: exact mathematical mapping, exact code path,
information source (input y vs target x), invertibility, and structural properties.

---

## Mathematical Setup

- Prime: p = 113
- Group: G = F_p* = <g>, order N = p-1 = 112
- Factorization: N = 7 × 16, gcd(7, 16) = 1
- CRT isomorphism: Z_112 ≅ Z_7 × Z_16 via x ↦ (x mod 7, x mod 16)
- Subgroup projections (input-derived, no x needed):
  - π_a(y) = y^{N/a} = y^{16} mod p (order-a subgroup, 7 elements)
  - π_b(y) = y^{N/b} = y^{7} mod p (order-b subgroup, 16 elements)
- DLP task: given y = g^x, predict x

---

## R0: RAW

### Code
```python
if variant == "RAW":
    tokens = [BOS, tok(g), SEP, tok(h), EOS]
```

### Properties
- **Mathematical mapping**: Input is (g, h) where h = g^x mod p
- **Information source**: Both g and h are input-derived (public group parameters)
- **Target**: tok(x) = x + 4 (the discrete log)
- **Invertible**: N/A (input is not a code for x; the DLP itself is the inverse)
- **Uniquely determines x**: YES (given g, h, and knowledge of discrete log)
- **Channels**: 1 (single group element h)
- **Cardinality per channel**: p = 113 possible values
- **Embedding size**: vocab_size = p + 4 = 117
- **Trainable input parameters**: 117 × d_model
- **Information content**: log2(112) ≈ 6.81 bits (the exponent x)
- **Preserves CRT factor membership**: NO (no factorization exposed)
- **Preserves group operation**: YES (h -> g*h corresponds to x -> x+1)
- **Preserves product separability**: NO (single coordinate)
- **Sequence length**: 5
- **Max token value**: p - 1 + 4 = 116

---

## R1: CRT-BOTH (True CRT / Input-Derived Projection)

### Code
```python
elif variant == "CRT-BOTH":
    tokens = [BOS, tok(g_b), SEP, tok(g_a), SEP, tok(h_b), SEP, tok(h_a), EOS]
```
where:
```python
g_a = project_factor(g, p, a)  # g^{(p-1)/a} = g^{16} mod p
g_b = project_factor(g, p, b)  # g^{(p-1)/b} = g^{7} mod p
h_a = project_factor(h, p, a)  # h^{16} mod p = g_a^{x mod a}
h_b = project_factor(h, p, b)  # h^{7} mod p = g_b^{x mod b}
```

### Properties
- **Mathematical mapping**: Input is (g_b, g_a, h_b, h_a) where:
  - h_a = g_a^{x mod a} (DLP in order-7 subgroup)
  - h_b = g_b^{x mod b} (DLP in order-16 subgroup)
- **Information source**: ALL input-derived (subgroup projections of g and h)
- **Target**: tok(x) = x + 4 (full discrete log)
- **Invertible**: The pair (h_a, h_b) uniquely determines h (via CRT), and h
  uniquely determines x. So the input is information-equivalent to RAW.
- **Uniquely determines x**: YES (via CRT reconstruction of h, then DLP)
- **Channels**: 2 (A-channel: g_a, h_a; B-channel: g_b, h_b)
- **Cardinality**: A-channel has 7 possible values, B-channel has 16
- **Embedding size**: vocab_size = p + 4 = 117 (same vocabulary)
- **Trainable input parameters**: 117 × d_model (same as RAW)
- **Information content**: log2(112) ≈ 6.81 bits (same as RAW)
- **Preserves CRT factor membership**: YES (explicitly factored)
- **Preserves group operation**: YES (within each factor: h_a -> g_a * h_a
  corresponds to x mod a -> (x+1) mod a)
- **Preserves product separability**: YES (the DLP decomposes into two
  independent subproblems)
- **Sequence length**: 9
- **CRITICAL**: This is CRT_PROJECTION (input-derived), NOT CRT_TARGET.
  The projections π_a(h) = h^{16} and π_b(h) = h^7 are computed from the
  input h alone, without knowing x. No target-derived information is supplied.

---

## CRT-7 (Factor A Only)

### Code
```python
elif variant == "CRT-7":
    tokens = [BOS, tok(g_a), SEP, tok(h_a), EOS]
```

### Properties
- **Information**: Only the order-7 subgroup projection
- **Uniquely determines x**: NO (only determines x mod 7)
- **Information ceiling**: A_max = 1/b = 1/16 ≈ 6.25% top-1 accuracy
- **Channels**: 1 (A-channel only)
- **Sequence length**: 5
- **This is an information-limited control**, not a test of partial structure

---

## CRT-16 (Factor B Only)

### Code
```python
elif variant == "CRT-16":
    tokens = [BOS, tok(g_b), SEP, tok(h_b), EOS]
```

### Properties
- **Information**: Only the order-16 subgroup projection
- **Uniquely determines x**: NO (only determines x mod 16)
- **Information ceiling**: A_max = 1/a = 1/7 ≈ 14.29% top-1 accuracy
- **Channels**: 1 (B-channel only)
- **Sequence length**: 5
- **This is an information-limited control**, not a test of partial structure

---

## RAW+CRT

### Code
```python
elif variant == "RAW+CRT":
    tokens = [BOS, tok(g), SEP, tok(h), SEP,
              tok(g_b), SEP, tok(g_a), SEP,
              tok(h_b), SEP, tok(h_a), EOS]
```

### Properties
- **Information**: RAW input PLUS both CRT projections
- **Uniquely determines x**: YES (contains all of RAW plus redundant CRT)
- **Channels**: 3 (raw + A + B)
- **Sequence length**: 13
- **Redundant information**: CRT projections are derivable from (g, h)

---

## SCRAMBLED — CRITICAL AUDIT

### Code
```python
elif variant == "SCRAMBLED":
    s_g_b = scramble_map.get(g_b, g_b)
    s_g_a = scramble_map.get(g_a, g_a)
    s_h_b = scramble_map.get(h_b, h_b)
    s_h_a = scramble_map.get(h_a, h_a)
    tokens = [BOS, tok(s_g_b), SEP, tok(s_g_a), SEP, tok(s_h_b), SEP, tok(s_h_a), EOS]
```
where scramble_map is constructed as:
```python
rng = random.Random(12345)
perm = list(range(p))  # [0, 1, ..., 112]
rng.shuffle(perm)
scramble_map = {i: perm[i] for i in range(p)}
```

### CRITICAL FINDING: What SCRAMBLED Actually Is

The scramble_map is a **single global permutation** of {0, 1, ..., p-1},
applied to ALL four CRT component values. This means:

1. **Factor partition IS preserved**: The model receives 4 tokens in fixed
   positions. Position 1 is always the "B-generator" channel, position 3 is
   always the "A-generator" channel, etc. The model knows which channel is
   which from token position.

2. **Algebraic structure within factors IS destroyed**: The permutation
   destroys the group structure within each factor. The scrambled value
   `s_h_a = perm[h_a]` has no algebraic relationship to `h_a` or to the
   group operation.

3. **Same permutation for both factors**: A single permutation is applied
   to both A and B components, rather than separate per-factor permutations.
   Since A-components take values in a 7-element subgroup and B-components
   in a 16-element subgroup, the permutation effectively acts as different
   mappings on each factor's value set.

### Classification

**SCRAMBLED is approximately R2 (PERMUTED CRT) in the research framework.**

It preserves the factor partition (the model knows which channel is A and
which is B) but destroys the algebraic/numeric structure within each factor.

It is NOT R3 (GLOBAL RANDOM BIJECTION) because:
- R3 would assign elements to (A, B) coordinates RANDOMLY, destroying
  which elements belong to which factor
- SCRAMBLED keeps the correct factor assignment; it only permutes values
  within each factor

### Properties
- **Mathematical mapping**: (g_b, g_a, h_b, h_a) -> (π(g_b), π(g_a), π(h_b), π(h_a))
  where π is a fixed random permutation of {0, ..., p-1}
- **Information source**: Input-derived (projections of g and h, then permuted)
- **Invertible**: YES (π is a bijection; the model can learn to invert it)
- **Uniquely determines x**: YES (information-equivalent to CRT-BOTH)
- **Factorized**: YES (two channels with fixed positions)
- **Algebraically aligned**: NO (permutation destroys group structure)
- **Product structure preserved**: YES (A and B channels are separate)
- **Channels**: 2 (same as CRT-BOTH)
- **Sequence length**: 9 (same as CRT-BOTH)
- **Information content**: log2(112) ≈ 6.81 bits (same as CRT-BOTH)

### Scientific Implication

SCRAMBLED's rapid grokking (comparable to CRT-BOTH) is evidence FOR
**computational factorization** (two small channels are easier to learn)
but is NOT evidence for **algebraic alignment** (correct group coordinates).

To distinguish these, we need:
- **R3: GLOBAL RANDOM BIJECTION** — randomly assign elements to (A, B)
  coordinates, destroying factor membership
- **R4: MIXED RADIX** — use a non-CRT two-coordinate system

---

## MISSING REPRESENTATIONS (To Be Implemented)

### R2: PERMUTED CRT (Proper)

Separate random permutations for each factor:
- π_A: Z_a -> Z_a (permutation of 7 elements)
- π_B: Z_b -> Z_b (permutation of 16 elements)
- Input: (π_B(g_b), π_A(g_a), π_B(h_b), π_A(h_a))

This is what SCRAMBLED approximately is, but with proper per-factor
permutations. Current SCRAMBLED uses a single global permutation, which
is close but not identical.

### R3: GLOBAL RANDOM BIJECTION

Random bijection φ: G -> Z_a × Z_b that destroys factor membership:
- For each group element y in G, assign a random pair (r1, r2) in Z_a × Z_b
- The pair uniquely identifies y but has no relationship to CRT factors
- Input: (φ_1(g), φ_2(g), φ_1(h), φ_2(h))

This tests whether factorization alone (without correct factor partition)
is sufficient for rapid grokking.

### R4: MIXED RADIX

Use a non-algebraic two-coordinate system:
- Index group elements by enumeration j(y) (e.g., sorted order of elements)
- q = floor(j / b), r = j mod b
- Input: (q_g, r_g, q_h, r_h)

This tests whether any two-coordinate system works, regardless of
algebraic meaning.

### R5: RANDOM SINGLE ID (Optional)

Apply a random permutation to the single-token representation:
- ψ: G -> G (random permutation)
- Input: (ψ(g), ψ(h))

This tests whether arbitrary relabeling of the single-channel
representation helps, or whether the two-coordinate structure itself matters.

---

## Summary Table

| Representation | Factorized | Alg. Aligned | Invertible | Product Struct | Channels | Seq Len |
|---|---|---|---|---|---|---|
| R0: RAW | No | N/A | N/A | No | 1 | 5 |
| R1: CRT-BOTH | Yes | Yes | Yes | Yes | 2 | 9 |
| CRT-7 | Partial | Yes | No | No | 1 | 5 |
| CRT-16 | Partial | Yes | No | No | 1 | 5 |
| RAW+CRT | Yes | Yes | Yes | Yes | 3 | 13 |
| SCRAMBLED | Yes | No | Yes | Yes | 2 | 9 |
| R2: PERMUTED CRT | Yes | No | Yes | Yes | 2 | 9 |
| R3: RANDOM BIJECTION | Yes | No | Yes | No | 2 | 9 |
| R4: MIXED RADIX | Yes | No | Yes | No | 2 | 9 |
| R5: RANDOM SINGLE | No | No | Yes | No | 1 | 5 |

## Critical Distinction

- **SCRAMBLED ≈ R2**: Preserves factor partition, destroys within-factor algebra
- **R3 (needed)**: Destroys factor partition entirely
- **R4 (needed)**: Non-algebraic two-coordinate system

The current SCRAMBLED result (fast grokking) supports **factorization** but
NOT **algebraic alignment**. R3 and R4 are needed to distinguish these.

---

## Audit Checklist

- [x] RAW: No target leakage. Input (g, h) is public.
- [x] CRT-BOTH: No target leakage. Projections π_a(h), π_b(h) computed from h.
- [x] CRT-7: No target leakage. Only exposes x mod 7.
- [x] CRT-16: No target leakage. Only exposes x mod 16.
- [x] RAW+CRT: No target leakage. CRT is redundant with RAW.
- [x] SCRAMBLED: No target leakage. Scramble applied to input projections.
- [ ] R3: Must verify no target leakage in random bijection construction.
- [ ] R4: Must verify j(y) enumeration does not use x.
- [x] Train/test splits: Same seed across conditions (verified in code).
- [x] Model capacities: Same architecture, same parameter count for
      same model_id across conditions.
- [ ] total_epochs vs T_grok: Must verify from training logs whether
      training stopped at grokking threshold or ran to max epochs.
