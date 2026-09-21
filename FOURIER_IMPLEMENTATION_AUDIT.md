# FOURIER IMPLEMENTATION AUDIT

## Audit 26: Generator Dimension

**Status**: PARTIALLY ADDRESSED

The hidden representation depends on (g, x), not only x. The current
implementation aggregates over all generators by averaging activations
across all (g, x) pairs with the same x.

**Current approach**: For each x, average H(g, x) over all generators
g, then index by CRT coordinates (x mod 7, x mod 16).

**Preferred approach**: For each fixed generator g, construct H_g(r_A,
r_B) and compute Fourier transform per generator, then average energy
across generators.

**Impact**: The current approach may mix generator-dependent structure.
However, since the model sees multiple generators during training, the
learned representation should be approximately generator-invariant.
The final-state results are robust, but the temporal analysis should
be repeated per-generator for the strongest claims.

---

## Audit 27: Aggregation Method Comparison

**Wrong approach**: Average activations over g, then Fourier transform.
**Preferred approach**: Fourier transform per generator, convert to
energy, then average energy.

These are NOT equivalent because |F[E[H]]|² ≠ E[|F[H]|²] in general
(Jensen's inequality).

**Current implementation**: Uses the wrong approach (average then
transform). The preferred approach (transform then average) should
be used for the final paper.

**Impact on results**: The qualitative findings (RAW high E_AB, CRT
low E_AB) are robust to this choice because the difference is large
(0.65 vs 0.01). However, exact numerical values may change.

---

## Audit 28: Activation Location

**Current**: Hidden activations at the EOS position (last non-PAD
token), after the final transformer layer.

This is the prediction/readout position — the position where the
model produces the output logits. This is the correct location for
studying the representation that directly feeds the output classifier.

**Not analyzed**:
- BOS position
- Mean-pooled across positions
- Residual stream before/after MLP
- Before/after LayerNorm

**Recommendation**: For the paper, report results at the EOS position
after the final layer. Optionally include layer 0 (after first
transformer layer) for comparison.

---

## Audit 29: Mean Centering

**Current**: Fourier analysis is performed on raw (uncentered)
activations. The DC component (E_0) is reported separately.

**Primary metric**: R_int = E_AB / (E_A + E_B + E_AB)
This excludes E_0, so centering does not affect R_int.

**Secondary metric**: E_0 + E_A + E_B + E_AB = 1 (normalized)
This includes E_0, so centering matters.

**Recommendation**: Report both:
- Uncentered: E_0 + E_A + E_B + E_AB = 1
- Centered (excluding DC): E_A + E_B + E_AB = 1

The interaction ratio R_int is the primary metric and is robust to
centering.
