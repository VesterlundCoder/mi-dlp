# Submission Checklist — Machine Learning Journal (Springer)

**Target**: Machine Learning Journal (Springer)
**Paper**: Mechanistic Interpretability of Discrete Logarithm Grokking: How Representation Determines the Internal Algorithm
**Author**: David Vesterlund, Industrial Research at Vesterlund Ventures Holding AB

## Science Freeze

- [x] R3 training COMPLETE: 100K epochs, 3 mapping seeds, all fail to grok
- [x] R2 (PERMUTED CRT) COMPLETE: groks at ~8K epochs
- [x] ROLE-DECOUPLED CRT COMPLETE: groks at ~26K epochs
- [x] R3O (ORDER-MATCHED) COMPLETE: 50K epochs, fails to grok
- [x] R4 (MIXED RADIX) COMPLETE: 50K epochs, fails to grok
- [x] Control hierarchy established: partition-preserving grok, partition-destroying fail
- [x] Top-k causal ablation COMPLETE: 100 rank-matched + 100 variance-matched controls, p < 0.01
- [x] SCRAMBLED audit COMPLETE: exactly conjugate to CRT (max logit diff = 0)
- [x] Temporal analysis COMPLETE: single RAW seed, E_AB forms before grokking
- [ ] Temporal replication on 3-5 RAW seeds (pending)
- [ ] Per-generator Fourier aggregation (pending)
- [ ] Top-k ablation on CRT-BOTH and SCRAMBLED (pending)

## Claim Corrections

- [x] "necessary" → "supports a requirement for rapid generalization"
- [x] "different algorithms" → "different internal representational organizations"
- [x] Removed incorrect |Δ|<0.02 claim from abstract
- [x] R3 described with 3 mapping seeds, 100K epochs (not single seed at 20K)
- [x] SCRAMBLED described as vacuous control (conjugate to CRT)
- [x] Control hierarchy reported in main text (not just appendix)

## Paper Format

- [x] Title updated: "How Representation Determines the Internal Algorithm"
- [x] Author: David Vesterlund (de-anonymized for journal submission)
- [x] 4-claim abstract
- [x] AI Use Statement before references
- [x] Figures: 4 main figures (learning curves, temporal E_AB, Fourier heatmaps, ablation dose-response)
- [x] Tables: moved to appendix
- [x] Related Work: expanded beyond Power/Nanda
- [x] Methods: full details (dataset, LR, batch, WD schedule, seeds, grokking criterion)
- [x] Reproducibility appendix
- [x] Compiled with Tectonic
- [x] PDF verified: 17 pages

## MLJ Contribution Information Sheet

- [x] Created as separate document: `mlj_contribution_information_sheet.tex`
- [x] 4 questions answered:
  1. Main claim and importance
  2. Evidence (4 forms: Fourier, temporal, causal, controls)
  3. Related work (Nanda, Chughtai, Stander, Furuta, McCracken, Nguyen)
  4. Previous publication disclosure (JMLR submission is distinct)
- [x] 2 pages
- [x] Compiled with Tectonic

## MLJ Submission Requirements

- [x] Main manuscript PDF
- [x] Contribution Information Sheet PDF
- [x] Author name and affiliation
- [x] AI Use Statement
- [x] References (references.bib)
- [x] Figures
- [x] Reproducibility information
- [ ] Submit via Springer Nature editorial system
- [ ] Verify no simultaneous review conflict with JMLR submission

## Files

- Main paper: `/Users/davidsvensson/Desktop/rd-lumi-z3/papers_for_publication/aistats_paper1_dlp_crt_grokking.tex`
- Main PDF: `/Users/davidsvensson/Desktop/rd-lumi-z3/papers_for_publication/aistats_paper1_dlp_crt_grokking.pdf`
- MLJ sheet: `/Users/davidsvensson/Desktop/rd-lumi-z3/papers_for_publication/mlj_contribution_information_sheet.tex`
- MLJ sheet PDF: `/Users/davidsvensson/Desktop/rd-lumi-z3/papers_for_publication/mlj_contribution_information_sheet.pdf`
- Bibliography: `/Users/davidsvensson/Desktop/rd-lumi-z3/papers_for_publication/references.bib`
- Figures: `/Users/davidsvensson/Desktop/rd-lumi-z3/papers_for_publication/figures/`

## Documentation

- [x] RUN_REGISTRY.md — reconciled, all final numbers
- [x] CLAIM_LEDGER.md — v4 FINAL, all claims updated
- [x] RESULTS_SUMMARY.md — final results
- [x] SUBMISSION_CHECKLIST.md — this file

## Pending (Non-Blocking for Submission)

1. Temporal E_AB replication on 3-5 RAW seeds
2. Per-generator Fourier aggregation
3. Top-k ablation on CRT-BOTH and SCRAMBLED
4. R3 additional training seeds (currently 3 mapping seeds, 1 training seed each)
5. Scaling to p=337 and p=601
