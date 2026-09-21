# PROBE AUDIT

## Audit 22: Probe Split Independence

**Status**: VERIFIED

Linear probes use a separate train/test split from the model's
training/test split:
- Model training: 30% of all (g, x) triples
- Model test: 70% of all (g, x) triples
- Probe training: 80% of model test set (hidden states from test examples)
- Probe test: 20% of model test set

The probe never sees hidden states from the model's training set.
The probe's own train/test split is independent.

---

## Audit 23: Cross-Generator Probes

**Status**: NOT YET IMPLEMENTED

Current probes mix generators: probe training and test sets contain
examples from all generators.

**Recommendation**: Split generators into G_probe-train and G_probe-test.
Train probes only on hidden states from training generators. Evaluate
on completely unseen generators.

This is a stronger test of whether the representation generalizes
across generator choice.

---

## Audit 24: Baseline-Corrected Probe Emergence

**Status**: ADDRESSED

The untrained network (epoch 0) has probe accuracies:
- A_x = 0.02 (chance)
- A_a = 0.21 (above chance for 7-way classification)
- A_b = 0.19 (above chance for 16-way classification)

The above-chance performance at epoch 0 is due to the embedding
initialization providing weak signal. The temporal analysis shows
probes rising from these baselines:
- ΔA_a(t) = A_a(t) - A_a(0)
- ΔA_b(t) = A_b(t) - A_b(0)

**Key finding**: A_a rises from 0.21 (epoch 0) to 0.47 (epoch 15K)
to 0.67 (epoch 50K). The emergence is genuine, not an artifact of
initialization.

---

## Audit 25: Non-Monotonic Probe Behavior

**Status**: INVESTIGATED

Some RAW checkpoints show non-monotonic probe performance:
- A_a at epoch 25K: 0.48
- A_a at epoch 25.5K: 0.36 (drops!)
- A_a at epoch 27K: 0.35
- A_a at epoch 30K: 0.36
- A_a at epoch 35K: 0.39 (recovers)

This is NOT a probe artifact:
- Same probe protocol across checkpoints
- Same probe train/test split
- Same regularization
- Same activation extraction location

**Interpretation**: The representation temporarily becomes less
linearly accessible around epoch 25K-30K, then recovers. This could
indicate a representational rotation or compression phase. The
interaction energy E_AB continues to rise during this period
(0.45 → 0.52), suggesting the network is restructuring.

**Recommendation**: Preserve this finding in the paper. A
representation that temporarily becomes less linearly accessible
and later recovers could itself be scientifically meaningful.
