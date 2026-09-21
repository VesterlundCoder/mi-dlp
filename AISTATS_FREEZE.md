# AISTATS Freeze Document — Paper 1

## How Problem Factorization Changes Grokking: Discrete Logarithms through Chinese Remainder Decomposition

**Status:** IN PROGRESS — experiments running
**Date:** September 20, 2026
**Abstract deadline:** September 29, 2026
**Full paper deadline:** October 6, 2026

---

## 1. Title (LOCKED)

> How Problem Factorization Changes Grokking: Discrete Logarithms through Chinese Remainder Decomposition

## 2. Authors (LOCKED)

- David Vesterlund, Industrial Research at Vesterlund Ventures Holding AB, david@vesterlundventures.se

## 3. Mathematical Task Definition (LOCKED)

- DLP on $\mathbb{F}_p^*$ with $p = 113$, $N = p - 1 = 112 = 7 \times 16$
- CRT factors: $a = 7$, $b = 16$, $\gcd(7, 16) = 1$
- $\alpha_a = \log 7 / \log 112 \approx 0.41$
- $\alpha_b = \log 16 / \log 112 \approx 0.59$
- Input-derived projections: $\pi_a(y) = y^{16}$, $\pi_b(y) = y^7$
- Full problem space: $\varphi(112) \times 112 = 5376$ triples
- Train: 30% (1612), Test: 70% (3764)

## 4. Exact Meaning of CRT Information (LOCKED)

- **Input-derived**: $\pi_a(y) = y^{N/a}$, $\pi_b(y) = y^{N/b}$ — computed from input $y$ only
- **NOT target-derived**: no access to $x$ or $x \bmod a$ during encoding
- All CRT information is efficiently computable without solving DLP

## 5. Whether RAW Remains in HALF (LOCKED)

- **YES**: C2 (RAW+CRT-A) and C3 (RAW+CRT-B) retain the raw DLP input $[g, h]$
- The existing CRT-16 and CRT-7 experiments did NOT retain raw input — they are C5/C6, not C2/C3
- The new experiments (crt_grokking_experiment.py) implement the full intervention matrix with raw retained

## 6. Group Orders Tested

- Primary: $p = 113$, $N = 112 = 7 \times 16$
- Additional (planned): $p = 241$ ($N = 240 = 15 \times 16$), $p = 337$ ($N = 336 = 16 \times 21$)

## 7. Number of Seeds

- Current run: seed 42 only
- Planned: seeds 42, 123, 456 (3 seeds minimum)
- Protocol recommends 20 seeds for main configuration

## 8. Formal Grokking Criterion (LOCKED)

- $T_{\text{grok}}(\tau) = \min\{t : A_{\text{full}}(t) \geq \tau\}$
- Primary threshold: $\tau = 0.95$ ($T_{95}$)
- Sensitivity: $\tau \in \{0.50, 0.90, 0.95, 0.99\}$
- Memorization: $T_{\text{mem}} = \min\{t : A_{\text{train}}(t) \geq 0.995\}$
- Censoring: if not grokked by $T_{\max} = 100{,}000$, record $T_{\text{grok}} > T_{\max}$

## 9. Grokking Rate Per Condition (PENDING — experiments running)

| Condition | Seed 42 | Seed 123 | Seed 456 | Grok Rate |
|-----------|---------|----------|----------|-----------|
| C0: RAW | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| C1: RAW+CRT-BOTH | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| C2: RAW+CRT-A | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| C3: RAW+CRT-B | [PENDING] | [PENDING] | [PENDING] | [PENDING] |
| C4: CRT-BOTH | [PENDING] | — | — | [PENDING] |
| C5: CRT-A | [PENDING] | — | — | [PENDING] |
| C6: CRT-B | [PENDING] | — | — | [PENDING] |
| C7: RAW+NULL | [PENDING] | — | — | [PENDING] |
| C8: RAW+RAND-A | [PLANNED] | — | — | [PLANNED] |

## 10. Censored Time-to-Grok Statistics (PENDING)

- Restricted Mean Survival Time (RMST) up to $T_{\max} = 100{,}000$
- Median time to grok where defined
- Kaplan-Meier curves planned

## 11. FULL vs RAW Effect (PENDING)

- $R_{\text{grok}} = T_{90}^{\text{RAW}} / T_{90}^{\text{CRT}}$
- Existing data (from algebraic_pairs): $R \approx 19.7\times$
- New data: [PENDING]

## 12. HALF vs RAW Effect (PENDING — THE KEY RESULT)

- Does RAW+CRT-A fail while RAW groks?
- If yes: partial-structure trap confirmed
- If no: partial-structure trap not supported
- Smoke test (2000 epochs): A_a → 0.95, A_b → 0.00 — TRAP SIGNATURE PRESENT

## 13. Complementary CRT-Component Result (PENDING)

- A_a(t) and A_b(t) trajectories for all conditions
- Fiber mass M_a(t) and M_b(t)
- [PENDING — experiments running]

## 14. Strongest Random-Control Result (PLANNED)

- RAW+RAND-A vs RAW+CRT-A comparison
- [NOT YET RUN]

## 15. Strongest Mechanistic Result (PENDING)

- Component accuracy trajectories
- Fiber mass analysis
- [PENDING — experiments running]

## 16. Every Numerical Statement Permitted in Abstract (PENDING)

- [PENDING — will be filled from experiment results]

---

## Experiment Status

- **Running**: 7 conditions × 1 seed (42) × 100K epochs
- **Location**: experiments/crt_intervention/
- **Device**: Apple MPS
- **ETA**: ~5 hours from start (started ~18:54, expected ~23:54)
- **Next**: Run seeds 123, 456 for main conditions
- **Then**: Run random controls (RAW+RAND-A, RAW+RAND-B)
- **Then**: Generate figures, fill in paper, finalize abstract
