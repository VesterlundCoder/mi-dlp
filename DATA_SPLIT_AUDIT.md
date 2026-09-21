# DATA SPLIT AUDIT

## Audit 4: Split Equivalence

**Status**: VERIFIED (by construction)

All representations use the same underlying (g, h, x) triples. The
train/test split happens at the level of mathematical examples BEFORE
representation encoding.

The split function uses `random.Random(seed)` to shuffle the triples,
then takes the first n_train as train and the rest as test. Since:
- The same triples are generated (all primitive roots × all exponents)
- The same seed (42) is used for splitting
- The split happens before encoding

All representations (RAW, CRT-BOTH, SCRAMBLED, R3, R4, etc.) use
exactly the same train and test triples.

**Verification method**: Hash every (g, h, x) triple in train and test
sets for each representation and compare.

---

## Audit 5: Dataset Coverage

For p=113, N=112:
- Number of generators: 48 (= φ(112))
- Number of distinct x: 112 (0 to 111)
- Examples per x: 48 (one per generator)
- Examples per generator: 112 (one per exponent)
- Total examples: 48 × 112 = 5,376

Train (30%): 1,612 examples
Test (70%): 3,764 examples

Coverage is uniform: every (g, x) pair appears exactly once.
No accidental imbalance.

---

## Audit 6: Encoded Collisions

For each representation, we check whether two distinct mathematical
inputs (g1, h1) ≠ (g2, h2) can produce identical encoded sequences.

| Representation | Collisions? | Expected |
|---------------|-------------|----------|
| RAW | No | None (g, h uniquely determine sequence) |
| CRT-BOTH | No | None (projections are deterministic) |
| SCRAMBLED | No | None (permutation is bijective) |
| R3 Random Bijection | No | None (bijection is invertible) |
| R4 Mixed Radix | No | None (enumeration is bijective) |
| CRT-7 | Yes | Expected (loses B information) |
| CRT-16 | Yes | Expected (loses A information) |

No unintended collisions found in any full-information representation.
