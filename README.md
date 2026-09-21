# mi-dlp: Mechanistic Interpretability of Discrete Logarithm Grokking

Code and data for the manuscript **"Mechanistic Interpretability of Discrete Logarithm Grokking: How Representation Determines the Internal Algorithm"** by David Vesterlund.

## Overview

This repository contains the training code, analysis pipelines, audit data, and figure-generation scripts for a mechanistic interpretability study of grokking on the discrete logarithm problem (DLP).

Key results:
- **Different representations, different internal algorithms**: RAW models develop interaction-rich Fourier structure (E_AB ≈ 0.65); CRT models develop separable structure (E_AB ≈ 0.01).
- **Interaction structure forms before grokking**: E_AB rises during the memorization plateau, ~20K epochs before visible generalization.
- **SCRAMBLED is a vacuous control**: It is exactly conjugate to CRT at initialization (max logit diff = 0), preserving all algebraic structure.
- **Algebraic alignment is required for rapid generalization**: A hierarchy of information-preserving negative controls shows that representations preserving the true CRT factor partition grok (TRUE CRT, PERMUTED CRT, ROLE-DECOUPLED CRT), while representations destroying the partition fail to grok even after 100K epochs across 3 mapping seeds (Global Random Bijection, Order-Matched Random, Mixed Radix).
- **Causal ablation confirms Fourier modes are necessary**: Top-k ablation with 100 rank-matched and variance-matched random controls shows A-, B-, and AB-Fourier modes are causally necessary (ΔA_x < -0.89 for k ≥ 4, p < 0.01).

## Repository Structure

```
mi-dlp/
├── README.md
├── LICENSE                    # MIT License
├── src/                       # Training and analysis scripts
│   ├── train_controls.py      # Control training (R3, R3O, R4, R2, ROLE-DECOUPLED)
│   ├── topk_ablation.py       # Top-k causal ablation (100 random controls)
│   ├── causal_ablation.py     # Full-rank causal ablation
│   ├── crt_grokking_experiment.py  # CRT grokking experiment
│   ├── generate_paper_figures.py   # Figure generation
│   ├── mi_suite.py            # Mutual information suite
│   ├── comprehensive_audit.py  # CRT audit (invertibility, generators, targets)
│   ├── representation_controls.py  # Representation control construction
│   ├── cka_analysis.py        # CKA representation similarity
│   ├── probe_analysis.py      # Linear probe analysis
│   ├── crt_spectral_analysis.py   # CRT spectral analysis
│   ├── generate_crt_figures.py    # CRT figure generation
│   └── lumi_probe_pipeline.py     # LUMI probe pipeline
├── analysis/                  # Analysis outputs (CSVs, JSONs)
│   ├── topk_ablation/         # Top-k ablation results
│   ├── temporal/              # Temporal analysis data
│   ├── probes/                # Linear probe results
│   ├── mi/                     # Mutual information results
│   ├── ablation/              # Ablation results
│   └── lumi_probes/           # LUMI probe results
├── audit/                     # Algebraic structure audit data
│   ├── algebraic_structure_scores.csv
│   ├── scrambled_isomorphism.json
│   ├── crt_invertibility.json
│   ├── generator_verification.json
│   ├── target_uniqueness.json
│   └── token_embedding_audit.json
├── paper/                     # Manuscript
│   ├── aistats_paper1_dlp_crt_grokking.tex
│   ├── aistats_paper1_dlp_crt_grokking.pdf
│   ├── mlj_contribution_information_sheet.tex
│   ├── mlj_contribution_information_sheet.pdf
│   └── references.bib
├── figures/                   # Generated figures (PDF)
├── docs/                      # Documentation
├── CLAIM_LEDGER.md            # Scientific claim tracking
├── RUN_REGISTRY.md           # Experiment run registry (source of truth)
├── RESULTS_SUMMARY.md         # Final results summary
├── SUBMISSION_CHECKLIST.md    # Submission checklist
├── AUDIT_V2.md                # Complete methodological audit
├── SCRAMBLED_ISOMORPHISM_AUDIT.md  # SCRAMBLED conjugacy audit
├── TOKEN_EMBEDDING_AUDIT.md   # Tokenization audit
├── DATA_SPLIT_AUDIT.md        # Data split audit
├── GROKKING_METRIC_AUDIT.md   # Grokking metric audit
├── FOURIER_IMPLEMENTATION_AUDIT.md  # Fourier implementation audit
├── EQUIVARIANCE_AUDIT.md      # Equivariance audit
├── PROBE_AUDIT.md             # Probe audit
├── ABLATION_AUDIT.md          # Ablation audit
├── AISTATS_FREEZE.md          # Science freeze document
├── representation_properties.csv  # Representation property scores
└── negative_control_validation.json  # Negative control validation
```

## Key Findings

### Control Hierarchy (p=113)

| Representation | S_hom | Partition? | Grokked? | T_grok | Seeds |
|---------------|-------|-----------|----------|--------|-------|
| RAW | — | No | Yes (slow) | ~50K | 1 |
| TRUE CRT | 1.0 | Yes | Yes (fast) | ~20K | 1 |
| SCRAMBLED | 1.0 | Yes | Yes (fast) | ~16K | 1 |
| PERMUTED CRT (R2) | — | Yes | YES | ~8K | 1 |
| ROLE-DECOUPLED CRT | — | Yes | YES | ~26K | 1 |
| GLOBAL RANDOM (R3) | 0.008 | No | NO | >100K | 3 |
| ORDER-MATCHED (R3O) | 0.050 | No | NO | >50K | 1 |
| MIXED RADIX (R4) | 0.024 | No | NO | >50K | 1 |

### Top-k Causal Ablation (RAW, Layer 1, 100 random controls)

| Subspace | k | Δ (structured) | Random (rank-matched) |
|----------|---|---------------|----------------------|
| U_A | 4 | -0.894 | +0.010 ± 0.004 |
| U_B | 4 | -0.888 | +0.010 ± 0.004 |
| U_AB | 4 | -0.924 | +0.010 ± 0.004 |

All structured ablations at 0th percentile, p < 0.01.

## Relationship to the Companion Manuscript

A companion manuscript, "Grokking the Discrete Logarithm: Scaling Across Model Capacity and Mechanistic Analysis," is archived on Zenodo (DOI: [10.5281/zenodo.22813397](https://doi.org/10.5281/zenodo.22813397)) and submitted to JMLR. That work studies DLP grokking existence and scaling across a 12-model capacity sweep. The present work asks a different question: how does input representation determine the internal algorithm? The two manuscripts share the M04 anchor architecture and RAW baseline but have distinct hypotheses, interventions, and conclusions.

## Citation

```bibtex
@misc{vesterlund2026mi_dlp,
  title={Mechanistic Interpretability of Discrete Logarithm Grokking:
         How Representation Determines the Internal Algorithm},
  author={Vesterlund, David},
  year={2026},
  howpublished={Manuscript submitted to Machine Learning Journal (Springer)},
  url={https://github.com/VesterlundCoder/mi-dlp}
}
```

## License

MIT License. See `LICENSE` for details.
