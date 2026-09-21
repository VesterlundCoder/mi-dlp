# ABLATION AUDIT

## Audit 32: Rank-Matched Comparisons

**Status**: PARTIALLY ADDRESSED

For p=113, N=112=7×16:
- A-only Fourier mode space: rank 6 (7-1 non-trivial A modes)
- B-only space: rank 15 (16-1 non-trivial B modes)
- AB interaction space: rank 90 (6×15 non-trivial AB modes)

Ablating all AB directions (rank 90) is NOT a fair comparison with
ablating all A directions (rank 6) because they remove different
amounts of the representation.

**Current implementation**: Uses matched random subspaces of identical
rank. This is correct for comparing structured vs random ablation
within each subspace class.

**Missing**: Cross-class comparison with rank matching. For
k ∈ {2, 4, 8, 16}, take top-k directions from A, B, AB and compare.

**Impact**: The current results show all three ablations cause
catastrophic damage. Rank matching would show whether AB ablation
is disproportionately damaging PER UNIT RANK.

---

## Audit 33: Variance-Matched Ablations

**Status**: PARTIALLY ADDRESSED

**Current**: Random subspaces of matched rank are used as controls.
This controls for rank but NOT for activation variance.

**Missing**: Energy-matched controls that remove the same amount of
representation variance without following CRT modes.

**Impact**: A structured subspace might matter merely because it
contains high activation energy. The current random-rank controls
show ΔA_x ≈ 0.01 (no damage), while structured ablations show
ΔA_x < -0.87. The large gap suggests the result is NOT merely due
to energy content, but energy-matched controls would make this
explicit.

---

## Audit 34: Separate Construction and Evaluation Data

**Status**: NOT YET ADDRESSED

**Current**: Subspaces U_A, U_B, U_AB are identified on the SAME
examples used for causal evaluation. This risks overfitting to
peculiarities of the evaluation examples.

**Required**: Use a separate subspace construction set and
intervention evaluation set.

**Impact**: The current results may slightly overestimate ablation
effects. However, the large gap between structured (ΔA_x < -0.87)
and random (|Δ| < 0.02) ablations suggests the effect is genuine
even with this caveat.

---

## Summary of Ablation Robustness

| Audit | Status | Impact on Results |
|-------|--------|-------------------|
| Rank matching | Partial | Results robust; cross-class comparison needed |
| Variance matching | Partial | Results likely robust; explicit control needed |
| Separate data | Not done | May slightly overestimate effects |
| Dose-response | Done | Monotonic, confirms specificity |
| Random controls | Done | 20 per condition, 0% percentile |
| Layer localization | Done | Layer 1 is primary; Layer 0 less affected |

**Conclusion**: The causal ablation results are strong but would
benefit from variance-matched controls and separate construction/
evaluation data for the strongest possible claim.
