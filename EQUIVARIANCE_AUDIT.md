# EQUIVARIANCE AUDIT

## Audit 30: Test-Test Transformations

**Status**: PARTIALLY ADDRESSED

The equivariance metric computes:
A_eq = P[hat_x_c = hat_x + c mod N]
where (y, c) are sampled and hat_x_c = f(g^c * y).

**Current implementation**: Samples (y, c) from all triples, which may
include train examples. The transformed example (g, g^c * h) may belong
to TRAIN, TEST, or neither.

**Required**: Primary equivariance metric must use TEST → TEST pairs
only. Call it A_eq^TT.

**Impact**: If transformed examples include train examples, the model
may have memorized them, inflating equivariance. However, since the
model sees 30% of examples during training, the probability of a
random transformation landing in train is 30%.

**Recommendation**: Implement TEST → TEST filtering for the final
paper. Report A_eq^TT as the primary metric.

---

## Audit 31: Generator-Aware Equivariance

**Status**: ADDRESSED

The equivariance test keeps g fixed under h → g^c * h. The target
transformation is x → x + c mod N. This is correct.

**Subgroup-specific shifts**: Not yet tested separately.

For N = 112 = 7 × 16:
- Shifts that alter only A coordinate: c ≡ 0 mod 16 (shifts by
  multiples of 16, which changes x mod 7 but not x mod 16... wait,
  this is wrong. c mod 16 = 0 means x mod 16 doesn't change, only
  x mod 7 changes)
- Shifts that alter only B coordinate: c ≡ 0 mod 7
- General shifts: c not a multiple of 7 or 16

**Recommendation**: Test subgroup-specific equivariance to check
whether A-factor or B-factor equivariance emerges first.

---

## Correct-Output vs Consistent-Output Equivariance

**Current**: A_eq measures consistent-output equivariance:
A_eq = P[hat_x_c = hat_x + c mod N]

This can be satisfied by a model with an incorrect constant offset.

**Required**: Also compute A_eq_true:
A_eq_true = P[hat_x_c = x + c mod N]

The difference is scientifically meaningful:
- A_eq high, A_eq_true low: model is equivariant but wrong
- Both high: model is equivariant and correct
- Both low: model is neither equivariant nor correct
