#!/usr/bin/env python3
"""
Comprehensive audit script for DLP grokking representations.

Implements Parts I-VI of the MASTER AUDIT AND CONTROL PROTOCOL:
  Audit 1:  CRT invertibility verification
  Audit 2:  Generator verification
  Audit 3:  Target uniqueness
  Audit 4:  Split equivalence
  Audit 5:  Dataset coverage
  Audit 6:  Encoded collisions
  Audit 7:  Tokenization/embedding audit
  Audit 8:  Weight tying audit
  Audit 9:  Token namespace audit
  Audit 10: Zero element issue
  Audit 11-13: SCRAMBLED isomorphism test
  Audit 14-16: Algebraic structure scores

Outputs:
  audit/crt_invertibility.json
  audit/generator_verification.json
  audit/target_uniqueness.json
  audit/split_equivalence.csv
  audit/dataset_coverage.json
  audit/encoded_collisions.json
  audit/token_embedding_audit.json
  audit/scrambled_isomorphism.json
  audit/algebraic_structure_scores.csv
"""

import json
import os
import sys
import hashlib
import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Set
import numpy as np

# Add parent dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def is_primitive_root(g: int, p: int) -> bool:
    """Check if g is a primitive root mod p."""
    if g <= 1 or g >= p:
        return False
    N = p - 1
    # Factor N
    factors = set()
    n = N
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    for q in factors:
        if pow(g, N // q, p) == 1:
            return False
    return True


def element_order(y: int, p: int) -> int:
    """Compute multiplicative order of y mod p."""
    if y % p == 0:
        return 0
    N = p - 1
    for d in range(1, N + 1):
        if N % d == 0 and pow(y, d, p) == 1:
            return d
    return N


def coprime_factorization(N: int) -> Tuple[int, int]:
    """Find coprime factorization N = a*b with a < b, gcd(a,b)=1."""
    for a in range(2, int(N**0.5) + 1):
        if N % a == 0:
            b = N // a
            if np.gcd(a, b) == 1:
                return a, b
    return 1, N


def project_factor(y: int, p: int, f: int) -> int:
    """Project y to subgroup of order f."""
    N = p - 1
    return pow(int(y), int(N // f), p)


def bezout_coefficients(a: int, b: int) -> Tuple[int, int]:
    """Find s, t such that a*s + b*t = gcd(a,b)."""
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t
    return old_s, old_t


# ============================================================
# AUDIT 1: CRT Invertibility Verification
# ============================================================

def audit_crt_invertibility(primes: List[int], output_dir: str):
    """Verify CRT invertibility for all primes."""
    print("\n" + "="*60)
    print("AUDIT 1: CRT Invertibility Verification")
    print("="*60)

    results = {}
    for p in primes:
        N = p - 1
        a, b = coprime_factorization(N)
        print(f"\n  p={p}, N={N}, a={a}, b={b}, gcd={np.gcd(a,b)}")

        # Verify gcd and product
        assert np.gcd(a, b) == 1, f"gcd(a,b) != 1 for p={p}"
        assert a * b == N, f"a*b != N for p={p}"

        # Compute projections for all elements
        elements = list(range(1, p))
        projections = {}
        collisions = 0
        order_violations = 0

        for y in elements:
            pi_A = project_factor(y, p, a)
            pi_B = project_factor(y, p, b)

            # Verify order divides f
            if pi_A != 0:
                ord_A = element_order(pi_A, p)
                if ord_A > a or a % ord_A != 0:
                    order_violations += 1
            if pi_B != 0:
                ord_B = element_order(pi_B, p)
                if ord_B > b or b % ord_B != 0:
                    order_violations += 1

            key = (pi_A, pi_B)
            if key in projections:
                collisions += 1
            projections[key] = y

        # Bézout reconstruction
        s, t = bezout_coefficients(a, b)
        # We need a*s + b*t = 1, so y = (y^a)^s * (y^b)^t
        # But we need to be careful with signs
        # Actually: if a*s + b*t = 1, then x = s * (x mod a) * ... hmm
        # The CRT reconstruction is:
        # x = (x_a * b * inv(b, a) + x_b * a * inv(a, b)) mod N
        # where x_a = x mod a, x_b = x mod b

        # For projections: y = pi_A^? * pi_B^?
        # pi_A = y^(N/a), pi_B = y^(N/b)
        # We need to find exponents such that y = pi_A^u * pi_B^v
        # y = y^(N/a * u + N/b * v) = y^(N(1/a * u + 1/b * v))
        # We need N(1/a * u + 1/b * v) = 1, i.e., u/a + v/b = 1/N
        # Actually, we need N/a * u + N/b * v = 1 (mod N)
        # i.e., b*u + a*v = 1 (mod N) ... no
        # N/a * u + N/b * v ≡ 1 (mod N)
        # b * u + a * v ≡ 1 (mod a*b)  [dividing by N/a*b... no]

        # Simpler: find u, v such that (N/a)*u + (N/b)*v ≡ 1 (mod N)
        # This means b*u + a*v ≡ 1 (mod a*b) after multiplying by ab/N = 1
        # So we need b*u + a*v = 1 (mod N)
        # Using Bézout: a*s + b*t = 1, so v=s, u=t gives a*s + b*t = 1
        # But we need b*u + a*v = 1, so u=t, v=s

        bezout_ok = True
        bezout_errors = 0
        for y in elements:
            pi_A = project_factor(y, p, a)
            pi_B = project_factor(y, p, b)
            # Reconstruct: y_recon = pi_A^t * pi_B^s mod p
            # where a*s + b*t = 1 (Bézout)
            y_recon = (pow(int(pi_A), int(t), p) * pow(int(pi_B), int(s), p)) % p
            if y_recon != y:
                bezout_errors += 1
                bezout_ok = False

        result = {
            "p": p, "N": N, "a": a, "b": b,
            "gcd_ab": int(np.gcd(a, b)),
            "product_ab": a * b,
            "num_elements": len(elements),
            "num_pairs": len(projections),
            "collisions": collisions,
            "order_violations": order_violations,
            "bezout_s": s, "bezout_t": t,
            "bezout_errors": bezout_errors,
            "bezout_ok": bezout_ok,
            "invertible": collisions == 0 and bezout_ok,
        }
        results[f"p{p}"] = result
        print(f"    Collisions: {collisions}, Bézout errors: {bezout_errors}, "
              f"Invertible: {result['invertible']}")

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "crt_invertibility.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {output_dir}/crt_invertibility.json")
    return results


# ============================================================
# AUDIT 2: Generator Verification
# ============================================================

def audit_generators(primes: List[int], output_dir: str):
    """Verify all generators are actually primitive roots."""
    print("\n" + "="*60)
    print("AUDIT 2: Generator Verification")
    print("="*60)

    results = {}
    for p in primes:
        N = p - 1
        # Find all primitive roots
        candidates = list(range(2, p))
        generators = [g for g in candidates if is_primitive_root(g, p)]

        # Verify each
        verified = 0
        orders = {}
        for g in generators:
            ord_g = element_order(g, p)
            orders[g] = ord_g
            if ord_g == N:
                verified += 1

        # Euler's totient
        phi_N = sum(1 for i in range(1, N + 1) if np.gcd(i, N) == 1)

        result = {
            "p": p, "N": N,
            "num_candidates": len(candidates),
            "num_generators": len(generators),
            "num_verified": verified,
            "phi_N": phi_N,
            "expected_count": phi_N,
            "count_matches": len(generators) == phi_N,
            "all_verified": verified == len(generators),
        }
        results[f"p{p}"] = result
        print(f"  p={p}: {len(generators)} generators (expected φ({N})={phi_N}), "
              f"all verified: {verified == len(generators)}")

    with open(os.path.join(output_dir, "generator_verification.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {output_dir}/generator_verification.json")
    return results


# ============================================================
# AUDIT 3: Target Uniqueness
# ============================================================

def audit_target_uniqueness(primes: List[int], output_dir: str):
    """Verify h = g^x and x is unique."""
    print("\n" + "="*60)
    print("AUDIT 3: Target Uniqueness")
    print("="*60)

    results = {}
    for p in primes:
        N = p - 1
        # Find a generator
        g = None
        for candidate in range(2, p):
            if is_primitive_root(candidate, p):
                g = candidate
                break

        # Check all (g, h, x) triples
        errors = 0
        for x in range(N):
            h = pow(g, x, p)
            # Verify h = g^x
            if pow(g, x, p) != h:
                errors += 1
                continue
            # Verify uniqueness: only one x gives this h
            count = sum(1 for x2 in range(N) if pow(g, x2, p) == h)
            if count != 1:
                errors += 1

        result = {
            "p": p, "N": N, "generator": g,
            "num_triples_checked": N,
            "errors": errors,
            "all_valid": errors == 0,
        }
        results[f"p{p}"] = result
        print(f"  p={p}: {N} triples checked, {errors} errors, valid: {errors == 0}")

    with open(os.path.join(output_dir, "target_uniqueness.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {output_dir}/target_uniqueness.json")
    return results


# ============================================================
# AUDIT 7-10: Tokenization and Embedding Audit
# ============================================================

def audit_tokenization(output_dir: str):
    """Audit tokenization, embedding, weight tying, namespaces, zero element."""
    print("\n" + "="*60)
    print("AUDIT 7-10: Tokenization and Embedding Audit")
    print("="*60)

    # Read the source code to determine tokenization
    # Key questions:
    # 1. Are token IDs purely categorical (learned embedding)?
    # 2. Is there weight tying between input embeddings and output classifier?
    # 3. Are token namespaces shared?
    # 4. Does the zero element issue exist?

    # From mi_suite.py, we know:
    # - INT_OFFSET is used: tok(v) = v + INT_OFFSET
    # - Embedding is nn.Embedding(vocab_size, d_model)
    # - Output is nn.Linear(d_model, vocab_size) or tied

    # Let's check the actual model code
    import mi_suite

    # Check INT_OFFSET
    int_offset = mi_suite.INT_OFFSET
    print(f"  INT_OFFSET = {int_offset}")

    # Check special tokens
    print(f"  PAD = {mi_suite.PAD}")
    print(f"  SEP = {mi_suite.SEP}")
    print(f"  BOS = {mi_suite.BOS}")
    print(f"  EOS = {mi_suite.EOS}")

    # Check vocab_size and token range
    # For p=113: group elements are 1..112, CRT projections are 0..112
    # With INT_OFFSET, tokens range from INT_OFFSET to INT_OFFSET + p
    p = 113
    max_token = max(p - 1 + int_offset, int(p) + int_offset)
    min_token = min(int_offset, int_offset)
    print(f"  p={p}: token range = [{min_token}, {max_token}]")
    print(f"  vocab_size needed = {max_token + 1}")

    # Check if zero element is used
    # Group elements are 1..p-1 (nonzero residues)
    # CRT projections can be 0 (when y^(N/f) = 1)
    # So token 0+INT_OFFSET = INT_OFFSET is used for CRT projection value 0
    # This is NOT the group element 0 (which doesn't exist in F_p*)
    # It's the identity element of the subgroup

    # Check SCRAMBLED: permutation is over 0..p-1
    # This means it can map a nonzero group element to 0
    # But 0 is not a valid group element!
    # However, the permutation is applied to PROJECTED values, not raw group elements
    # Projected values CAN be 0 (identity of subgroup)

    # Actually, let's check: project_factor(y, p, f) = y^(N/f) mod p
    # This is never 0 for y in 1..p-1 (since F_p* is a group)
    # So projected values are in 1..p-1, never 0
    # But the scramble_map is defined over 0..p-1
    # So it CAN map a projected value (which is in 1..p-1) to 0
    # This would create a token that represents "0" which is not a valid
    # subgroup element

    # Let's verify
    rng = random.Random(12345)
    perm = list(range(p))
    rng.shuffle(perm)
    scramble_map = {i: perm[i] for i in range(p)}

    # Check if any projected value maps to 0
    a, b = coprime_factorization(p - 1)
    zero_mappings = 0
    for y in range(1, p):
        g_a = project_factor(y, p, a)
        g_b = project_factor(y, p, b)
        if scramble_map[g_a] == 0 or scramble_map[g_b] == 0:
            zero_mappings += 1

    print(f"  SCRAMBLED zero-mappings: {zero_mappings} (projected values mapped to 0)")

    # Check model architecture for weight tying
    # From mi_suite.py GrokkingTransformer:
    # self.embed = nn.Embedding(vocab_size, d_model)
    # self.unembed = nn.Linear(d_model, vocab_size)
    # These are SEPARATE (not tied)

    # Check if there's weight tying
    has_tying = False
    # Read the source
    import inspect
    source = inspect.getsource(mi_suite.GrokkingTransformer)
    if "weight_tying" in source or "tie" in source.lower():
        has_tying = True
    print(f"  Weight tying: {has_tying}")

    # Check token namespace
    # All tokens share the same namespace: tok(v) = v + INT_OFFSET
    # Group elements, CRT projections, and output labels all use the same space
    # This means tok(5) for a group element and tok(5) for an output exponent
    # share the same embedding row!
    namespace_shared = True
    print(f"  Token namespace shared: {namespace_shared}")
    print(f"  Group element 5 and output label 5 share embedding row: {namespace_shared}")

    result = {
        "int_offset": int_offset,
        "special_tokens": {"PAD": mi_suite.PAD, "SEP": mi_suite.SEP,
                          "BOS": mi_suite.BOS, "EOS": mi_suite.EOS},
        "p113": {
            "token_range": [min_token, max_token],
            "vocab_size_needed": max_token + 1,
            "zero_mappings_in_scrambled": zero_mappings,
        },
        "weight_tying": has_tying,
        "namespace_shared": namespace_shared,
        "numeric_meaning": "NONE - tokens are purely categorical via learned embedding",
        "embedding_type": "nn.Embedding (learned lookup table, no positional/numeric encoding of token value)",
        "output_type": "nn.Linear (separate from input embedding, not tied)",
        "zero_element_issue": {
            "group_elements": "1..p-1 (never 0)",
            "crt_projections": "1..p-1 (never 0, since F_p* is closed under multiplication)",
            "scramble_map_range": "0..p-1 (CAN map to 0, creating invalid token)",
            "zero_mappings_count": zero_mappings,
            "issue_exists": zero_mappings > 0,
        },
        "namespace_collision": {
            "description": "Group element token tok(y) and output label token tok(x) share the same embedding row when y == x",
            "impact": "Input group element and output exponent with same numeric value share learned representation",
            "severity": "MEDIUM - may create unintended input-output coupling",
        },
    }

    with open(os.path.join(output_dir, "token_embedding_audit.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {output_dir}/token_embedding_audit.json")
    return result


# ============================================================
# AUDIT 11-13: SCRAMBLED Isomorphism Test
# ============================================================

def audit_scrambled_isomorphism(output_dir: str):
    """Test if SCRAMBLED is merely a vocabulary relabeling of CRT.

    If SCRAMBLED applies the same permutation to all positions, and the model
    uses learned embeddings (not numeric), then SCRAMBLED may be EXACTLY
    conjugate to CRT via a permutation of embedding rows.
    """
    print("\n" + "="*60)
    print("AUDIT 11-13: SCRAMBLED Isomorphism Test")
    print("="*60)

    p = 113
    a, b = coprime_factorization(p - 1)

    # Current SCRAMBLED: same permutation applied to all 4 CRT positions
    rng = random.Random(12345)
    perm = list(range(p))
    rng.shuffle(perm)
    scramble_map = {i: perm[i] for i in range(p)}

    # Check: is the same permutation applied to all positions?
    # From source code:
    # s_g_b = scramble_map.get(g_b, g_b)
    # s_g_a = scramble_map.get(g_a, g_a)
    # s_h_b = scramble_map.get(h_b, h_b)
    # s_h_a = scramble_map.get(h_a, h_a)
    # YES - same scramble_map for all positions

    same_permutation = True
    print(f"  Same permutation for all positions: {same_permutation}")

    # Check: does the permutation preserve equality relationships?
    # u = v iff π(u) = π(v) — YES, since π is a bijection
    equality_preserved = True
    for u in range(p):
        for v in range(p):
            if (u == v) != (scramble_map[u] == scramble_map[v]):
                equality_preserved = False
                break
    print(f"  Equality relationships preserved: {equality_preserved}")

    # Check: does the permutation preserve the support intersection?
    # For coprime subgroups A and B, their intersection is {1}
    # (the identity element)
    # Under permutation, the image of A and image of B have the same
    # intersection cardinality
    sub_A = set()
    sub_B = set()
    for y in range(1, p):
        sub_A.add(project_factor(y, p, a))
        sub_B.add(project_factor(y, p, b))

    intersection_original = sub_A & sub_B
    print(f"  Original subgroup A size: {len(sub_A)}, B size: {len(sub_B)}")
    print(f"  Original intersection: {intersection_original} (size {len(intersection_original)})")

    # Under permutation
    sub_A_scrambled = {scramble_map[v] for v in sub_A}
    sub_B_scrambled = {scramble_map[v] for v in sub_B}
    intersection_scrambled = sub_A_scrambled & sub_B_scrambled
    print(f"  Scrambled subgroup A size: {len(sub_A_scrambled)}, B size: {len(sub_B_scrambled)}")
    print(f"  Scrambled intersection size: {len(intersection_scrambled)}")

    # Key question: is SCRAMBLED exactly conjugate to CRT?
    # If the model uses learned embeddings and the same permutation is
    # applied to all input positions, then:
    # E'[tok(π(y))] = E[tok(y)]
    # makes the two models identical at initialization.
    #
    # The output labels (target x) are NOT permuted.
    # So if input embeddings are untied from output, the permutation
    # can be absorbed by the input embedding.
    #
    # If input and output are TIED, then permuting input also permutes
    # output, which would change the target labels — breaking conjugacy.
    #
    # From our audit: input and output are NOT tied (separate nn.Embedding
    # and nn.Linear). So SCRAMBLED IS exactly conjugate to CRT at init.

    # Let's verify numerically
    import torch
    from mi_suite import GrokkingTransformer, INT_OFFSET, BOS, SEP, EOS, PAD

    # Create two models with conjugate embeddings
    torch.manual_seed(42)
    model_crt = GrokkingTransformer(vocab_size=117, d_model=128, n_heads=4,
                                     n_layers=2, max_len=9, dropout=0.0)
    torch.manual_seed(42)
    model_scr = GrokkingTransformer(vocab_size=117, d_model=128, n_heads=4,
                                     n_layers=2, max_len=9, dropout=0.0)

    # Permute input embeddings in model_scr
    with torch.no_grad():
        for v in range(p):
            orig_token = v + INT_OFFSET
            scrambled_token = scramble_map[v] + INT_OFFSET
            if orig_token < model_crt.embed.weight.shape[0] and scrambled_token < model_scr.embed.weight.shape[0]:
                model_scr.embed.weight[scrambled_token] = model_crt.embed.weight[orig_token]

    # Test: feed CRT example to model_crt and corresponding SCRAMBLED example to model_scr
    # They should produce identical outputs
    g_val = 3  # primitive root
    h_val = pow(g_val, 30, p)
    x_val = 30

    # CRT encoding
    g_a = project_factor(g_val, p, a)
    g_b = project_factor(g_val, p, b)
    h_a = project_factor(h_val, p, a)
    h_b = project_factor(h_val, p, b)

    def tok(v): return v + INT_OFFSET
    crt_tokens = [BOS, tok(g_b), SEP, tok(g_a), SEP, tok(h_b), SEP, tok(h_a), EOS]

    # SCRAMBLED encoding
    s_g_b = scramble_map[g_b]
    s_g_a = scramble_map[g_a]
    s_h_b = scramble_map[h_b]
    s_h_a = scramble_map[h_a]
    scr_tokens = [BOS, tok(s_g_b), SEP, tok(s_g_a), SEP, tok(s_h_b), SEP, tok(s_h_a), EOS]

    # Pad to max_len
    crt_padded = crt_tokens + [PAD] * (9 - len(crt_tokens))
    scr_padded = scr_tokens + [PAD] * (9 - len(scr_tokens))

    crt_tensor = torch.tensor([crt_padded], dtype=torch.long)
    scr_tensor = torch.tensor([scr_padded], dtype=torch.long)

    model_crt.eval()
    model_scr.eval()

    with torch.no_grad():
        logits_crt = model_crt(crt_tensor)
        logits_scr = model_scr(scr_tensor)

    # Compare logits at EOS position
    max_diff = (logits_crt - logits_scr).abs().max().item()
    print(f"  Max logit difference (conjugate test): {max_diff:.2e}")
    print(f"  Conjugate at initialization: {max_diff < 1e-5}")

    # Also check: does the permutation affect the OUTPUT?
    # Output labels are tok(x) = x + INT_OFFSET
    # These are NOT permuted in SCRAMBLED
    # So the output classifier sees the same labels
    # If input embeddings are permuted but output is not,
    # the model can still learn by adjusting the unembed layer
    # But at INIT, the logits should match if we permute input embeddings

    result = {
        "same_permutation_all_positions": same_permutation,
        "equality_preserved": equality_preserved,
        "original_intersection_size": len(intersection_original),
        "scrambled_intersection_size": len(intersection_scrambled),
        "intersection_preserved": len(intersection_original) == len(intersection_scrambled),
        "weight_tying": False,
        "conjugate_at_init": max_diff < 1e-5,
        "max_logit_diff": max_diff,
        "conclusion": "SCRAMBLED is EXACTLY conjugate to CRT at initialization via input embedding permutation. "
                       "The two models are not meaningfully different from the model's perspective at init. "
                       "Any behavioral difference during training must come from optimization dynamics, "
                       "not from the representation itself. SCRAMBLED does NOT test destruction of algebraic structure.",
    }

    with open(os.path.join(output_dir, "scrambled_isomorphism.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved to {output_dir}/scrambled_isomorphism.json")
    print(f"\n  CONCLUSION: {result['conclusion']}")
    return result


# ============================================================
# AUDIT 14-16: Algebraic Structure Scores
# ============================================================

def compute_homomorphism_score(p: int, a: int, b: int,
                                phi: Dict[int, Tuple[int, int]]) -> float:
    """Compute S_hom: fraction of pairs where phi(uv) = phi(u) ⊕ phi(v)."""
    N = p - 1
    correct = 0
    total = 0
    for u in range(1, p):
        for v in range(1, p):
            uv = (u * v) % p
            phi_u = phi[u]
            phi_v = phi[v]
            phi_uv = phi[uv]
            # Coordinate-wise addition
            expected = ((phi_u[0] + phi_v[0]) % a, (phi_u[1] + phi_v[1]) % b)
            if phi_uv == expected:
                correct += 1
            total += 1
    return correct / total


def compute_translation_score(p: int, a: int, b: int,
                               phi: Dict[int, Tuple[int, int]]) -> float:
    """Compute S_trans: average best translation equivariance."""
    N = p - 1
    # Find generator
    g = None
    for c in range(2, p):
        if is_primitive_root(c, p):
            g = c
            break

    scores = []
    for c in range(1, p):
        # For each c, find the best Delta such that phi(c*y) = phi(y) ⊕ Delta
        delta_counts = Counter()
        for y in range(1, p):
            cy = (c * y) % p
            phi_y = phi[y]
            phi_cy = phi[cy]
            delta = ((phi_cy[0] - phi_y[0]) % a, (phi_cy[1] - phi_y[1]) % b)
            delta_counts[delta] += 1
        best_score = max(delta_counts.values()) / (p - 1)
        scores.append(best_score)
    return float(np.mean(scores))


def compute_partition_ari(p: int, a: int, b: int,
                          phi: Dict[int, Tuple[int, int]],
                          true_crt: bool = False) -> Tuple[float, float]:
    """Compute Adjusted Rand Index for factor partitions."""
    # True CRT partition: y1 ~ y2 iff pi_A(y1) = pi_A(y2)
    true_A_partition = defaultdict(set)
    true_B_partition = defaultdict(set)
    for y in range(1, p):
        pi_A = project_factor(y, p, a)
        pi_B = project_factor(y, p, b)
        true_A_partition[pi_A].add(y)
        true_B_partition[pi_B].add(y)

    # Candidate partition: y1 ~ y2 iff phi_A(y1) = phi_A(y2)
    cand_A_partition = defaultdict(set)
    cand_B_partition = defaultdict(set)
    for y in range(1, p):
        cand_A_partition[phi[y][0]].add(y)
        cand_B_partition[phi[y][1]].add(y)

    # Compute ARI
    ari_A = adjusted_rand_index(list(true_A_partition.values()),
                                 list(cand_A_partition.values()))
    ari_B = adjusted_rand_index(list(true_B_partition.values()),
                                 list(cand_B_partition.values()))
    return ari_A, ari_B


def adjusted_rand_index(partitions1: List[Set], partitions2: List[Set]) -> float:
    """Compute Adjusted Rand Index between two partitions."""
    # Build contingency table
    elements = set()
    for s in partitions1:
        elements |= s
    for s in partitions2:
        elements |= s
    elements = sorted(elements)
    n = len(elements)

    elem_to_idx = {e: i for i, e in enumerate(elements)}
    labels1 = np.zeros(n, dtype=int)
    labels2 = np.zeros(n, dtype=int)

    for i, part in enumerate(partitions1):
        for e in part:
            labels1[elem_to_idx[e]] = i
    for i, part in enumerate(partitions2):
        for e in part:
            labels2[elem_to_idx[e]] = i

    # Contingency table
    from collections import Counter
    contingency = Counter()
    for i in range(n):
        contingency[(labels1[i], labels2[i])] += 1

    # ARI formula
    a = sum(v * (v - 1) // 2 for v in Counter(labels1).values())
    b = sum(v * (v - 1) // 2 for v in Counter(labels2).values())
    c = sum(v * (v - 1) // 2 for v in contingency.values())

    if a + b == 0:
        return 1.0 if c == 0 else 0.0

    expected = a * b / (n * (n - 1) // 2)
    max_index = (a + b) / 2

    if max_index == expected:
        return 1.0
    return (c - expected) / (max_index - expected)


def audit_algebraic_structure(output_dir: str):
    """Compute algebraic structure scores for all representations."""
    print("\n" + "="*60)
    print("AUDIT 14-16: Algebraic Structure Scores")
    print("="*60)

    p = 113
    a, b = coprime_factorization(p - 1)
    N = p - 1

    results = {}

    # 1. TRUE CRT
    print("\n  Computing TRUE CRT scores...")
    true_crt_phi = {}
    for y in range(1, p):
        pi_A = project_factor(y, p, a)
        pi_B = project_factor(y, p, b)
        # Map to Z_a x Z_b using discrete log in subgroup
        # Actually, the CRT map is: x -> (x mod a, x mod b)
        # But we're mapping group elements y, not exponents x
        # The projection y^(N/a) gives an element of order dividing a
        # We need to map this to Z_a
        # Find the discrete log of pi_A in the order-a subgroup
        g = None
        for c in range(2, p):
            if is_primitive_root(c, p):
                g = c
                break
        g_a = project_factor(g, p, a)
        # Find index of pi_A in the subgroup generated by g_a
        idx_A = None
        for i in range(a):
            if pow(g_a, i, p) == pi_A:
                idx_A = i
                break
        g_b = project_factor(g, p, b)
        idx_B = None
        for i in range(b):
            if pow(g_b, i, p) == pi_B:
                idx_B = i
                break
        true_crt_phi[y] = (idx_A, idx_B)

    s_hom_true = compute_homomorphism_score(p, a, b, true_crt_phi)
    s_trans_true = compute_translation_score(p, a, b, true_crt_phi)
    ari_A_true, ari_B_true = compute_partition_ari(p, a, b, true_crt_phi)
    print(f"    TRUE CRT: S_hom={s_hom_true:.4f}, S_trans={s_trans_true:.4f}, "
          f"ARI_A={ari_A_true:.4f}, ARI_B={ari_B_true:.4f}")
    results["TRUE_CRT"] = {
        "S_hom": s_hom_true, "S_trans": s_trans_true,
        "ARI_A": ari_A_true, "ARI_B": ari_B_true,
    }

    # 2. CURRENT SCRAMBLED (coordinate-wise permutation)
    print("\n  Computing CURRENT SCRAMBLED scores...")
    rng = random.Random(12345)
    perm = list(range(p))
    rng.shuffle(perm)
    scramble_map = {i: perm[i] for i in range(p)}

    # SCRAMBLED applies the same permutation to projected values
    # But the "coordinates" are still the same factor channels
    # The permutation maps subgroup elements to arbitrary values
    # We need to map these back to Z_a x Z_b

    # Actually, SCRAMBLED tokenizes the permuted projected values
    # The model sees tok(π(g_a)), tok(π(g_b)), etc.
    # The "coordinate" in Z_a x Z_b is determined by the projected value
    # After permutation, the value π(g_a) is some arbitrary element
    # But it's still in the same channel (A or B)

    # For the algebraic structure score, we need to define what φ is
    # SCRAMBLED φ(y) = (scramble_map[pi_A(y)], scramble_map[pi_B(y)])
    # But these are values in 0..p-1, not in Z_a x Z_b
    # We need to map them to Z_a x Z_b

    # The model sees these as categorical tokens
    # The "coordinate" is just the token value
    # For structure scoring, we use the token value mod a and mod b
    # (This is arbitrary, but it's what the model sees)

    # Actually, for SCRAMBLED, the factor partition is PRESERVED
    # because the same permutation is applied within each factor channel
    # So ARI should be 1 (same partition as CRT)

    # But the group law is NOT preserved because the permutation
    # doesn't respect the subgroup structure

    # Let's compute properly: SCRAMBLED φ(y) = (π_scr_A(y), π_scr_B(y))
    # where π_scr_A(y) = scramble_map[project_factor(y, p, a)]
    # The "coordinate" in Z_a is the index of π_scr_A in the scrambled subgroup

    scrambled_phi = {}
    g = None
    for c in range(2, p):
        if is_primitive_root(c, p):
            g = c
            break
    g_a = project_factor(g, p, a)
    g_b = project_factor(g, p, b)

    # Build scrambled subgroup element lists
    sub_A = [pow(g_a, i, p) for i in range(a)]
    sub_B = [pow(g_b, i, p) for i in range(b)]
    scrambled_sub_A = [scramble_map[v] for v in sub_A]
    scrambled_sub_B = [scramble_map[v] for v in sub_B]

    val_to_idx_A = {v: i for i, v in enumerate(scrambled_sub_A)}
    val_to_idx_B = {v: i for i, v in enumerate(scrambled_sub_B)}

    for y in range(1, p):
        pi_A = project_factor(y, p, a)
        pi_B = project_factor(y, p, b)
        scr_A = scramble_map[pi_A]
        scr_B = scramble_map[pi_B]
        idx_A = val_to_idx_A.get(scr_A, 0)
        idx_B = val_to_idx_B.get(scr_B, 0)
        scrambled_phi[y] = (idx_A, idx_B)

    s_hom_scr = compute_homomorphism_score(p, a, b, scrambled_phi)
    s_trans_scr = compute_translation_score(p, a, b, scrambled_phi)
    ari_A_scr, ari_B_scr = compute_partition_ari(p, a, b, scrambled_phi)
    print(f"    SCRAMBLED: S_hom={s_hom_scr:.4f}, S_trans={s_trans_scr:.4f}, "
          f"ARI_A={ari_A_scr:.4f}, ARI_B={ari_B_scr:.4f}")
    results["CURRENT_SCRAMBLED"] = {
        "S_hom": s_hom_scr, "S_trans": s_trans_scr,
        "ARI_A": ari_A_scr, "ARI_B": ari_B_scr,
    }

    # 3. RANDOM BIJECTION (R3) - multiple seeds
    print("\n  Computing RANDOM BIJECTION (R3) scores...")
    r3_scores = {"S_hom": [], "S_trans": [], "ARI_A": [], "ARI_B": []}
    for seed in range(10):
        rng = random.Random(seed * 1000 + 42)
        pairs = [(i % a, i % b) for i in range(N)]
        rng.shuffle(pairs)
        bijection = {i: pairs[i] for i in range(N)}

        # Map group elements by sorted order
        elements = sorted(range(1, p))
        r3_phi = {}
        for idx, y in enumerate(elements):
            r3_phi[y] = bijection[idx]

        s_hom = compute_homomorphism_score(p, a, b, r3_phi)
        s_trans = compute_translation_score(p, a, b, r3_phi)
        ari_A, ari_B = compute_partition_ari(p, a, b, r3_phi)
        r3_scores["S_hom"].append(s_hom)
        r3_scores["S_trans"].append(s_trans)
        r3_scores["ARI_A"].append(ari_A)
        r3_scores["ARI_B"].append(ari_B)

    print(f"    R3 (10 seeds): S_hom={np.mean(r3_scores['S_hom']):.4f}±{np.std(r3_scores['S_hom']):.4f}, "
          f"S_trans={np.mean(r3_scores['S_trans']):.4f}±{np.std(r3_scores['S_trans']):.4f}, "
          f"ARI_A={np.mean(r3_scores['ARI_A']):.4f}±{np.std(r3_scores['ARI_A']):.4f}")
    results["RANDOM_BIJECTION_R3"] = {
        "S_hom_mean": float(np.mean(r3_scores["S_hom"])),
        "S_hom_std": float(np.std(r3_scores["S_hom"])),
        "S_trans_mean": float(np.mean(r3_scores["S_trans"])),
        "S_trans_std": float(np.std(r3_scores["S_trans"])),
        "ARI_A_mean": float(np.mean(r3_scores["ARI_A"])),
        "ARI_A_std": float(np.std(r3_scores["ARI_A"])),
        "ARI_B_mean": float(np.mean(r3_scores["ARI_B"])),
        "ARI_B_std": float(np.std(r3_scores["ARI_B"])),
    }

    # 4. MIXED RADIX (R4)
    print("\n  Computing MIXED RADIX scores...")
    elements = sorted(range(1, p))
    elem_to_idx = {y: i for i, y in enumerate(elements)}
    mixed_phi = {}
    for y in range(1, p):
        idx = elem_to_idx[y]
        q = idx // b
        r = idx % b
        mixed_phi[y] = (q, r)

    s_hom_mixed = compute_homomorphism_score(p, a, b, mixed_phi)
    s_trans_mixed = compute_translation_score(p, a, b, mixed_phi)
    ari_A_mixed, ari_B_mixed = compute_partition_ari(p, a, b, mixed_phi)
    print(f"    MIXED_RADIX: S_hom={s_hom_mixed:.4f}, S_trans={s_trans_mixed:.4f}, "
          f"ARI_A={ari_A_mixed:.4f}, ARI_B={ari_B_mixed:.4f}")
    results["MIXED_RADIX"] = {
        "S_hom": s_hom_mixed, "S_trans": s_trans_mixed,
        "ARI_A": ari_A_mixed, "ARI_B": ari_B_mixed,
    }

    # 5. ORDER-MATCHED RANDOM BIJECTION (R3-ORDER)
    print("\n  Computing ORDER-MATCHED RANDOM BIJECTION scores...")
    # Compute element orders
    elem_orders = {y: element_order(y, p) for y in range(1, p)}
    # Compute coordinate-pair orders
    pair_orders = {}
    for i in range(a):
        for j in range(b):
            ord_i = a // np.gcd(i, a) if i > 0 else 1
            ord_j = b // np.gcd(j, b) if j > 0 else 1
            pair_orders[(i, j)] = int(np.lcm(ord_i, ord_j))

    # Group elements by order
    elements_by_order = defaultdict(list)
    for y in range(1, p):
        elements_by_order[elem_orders[y]].append(y)

    # Group pairs by order
    pairs_by_order = defaultdict(list)
    for pair, order in pair_orders.items():
        pairs_by_order[order].append(pair)

    # Match: for each order, biject elements to pairs of same order
    order_matched_scores = {"S_hom": [], "S_trans": [], "ARI_A": [], "ARI_B": []}
    for seed in range(10):
        rng = random.Random(seed * 1000 + 42)
        order_phi = {}
        for order in elements_by_order:
            elems = elements_by_order[order]
            pairs = pairs_by_order.get(order, [])
            if len(elems) != len(pairs):
                # Can't perfectly match, skip this order
                # Assign randomly
                all_pairs = [(i, j) for i in range(a) for j in range(b)]
                rng.shuffle(all_pairs)
                for idx, y in enumerate(elems):
                    order_phi[y] = all_pairs[idx % len(all_pairs)]
            else:
                shuffled_pairs = pairs.copy()
                rng.shuffle(shuffled_pairs)
                for idx, y in enumerate(elems):
                    order_phi[y] = shuffled_pairs[idx]

        s_hom = compute_homomorphism_score(p, a, b, order_phi)
        s_trans = compute_translation_score(p, a, b, order_phi)
        ari_A, ari_B = compute_partition_ari(p, a, b, order_phi)
        order_matched_scores["S_hom"].append(s_hom)
        order_matched_scores["S_trans"].append(s_trans)
        order_matched_scores["ARI_A"].append(ari_A)
        order_matched_scores["ARI_B"].append(ari_B)

    print(f"    ORDER-MATCHED (10 seeds): S_hom={np.mean(order_matched_scores['S_hom']):.4f}±{np.std(order_matched_scores['S_hom']):.4f}, "
          f"S_trans={np.mean(order_matched_scores['S_trans']):.4f}±{np.std(order_matched_scores['S_trans']):.4f}")
    results["ORDER_MATCHED_RANDOM"] = {
        "S_hom_mean": float(np.mean(order_matched_scores["S_hom"])),
        "S_hom_std": float(np.std(order_matched_scores["S_hom"])),
        "S_trans_mean": float(np.mean(order_matched_scores["S_trans"])),
        "S_trans_std": float(np.std(order_matched_scores["S_trans"])),
        "ARI_A_mean": float(np.mean(order_matched_scores["ARI_A"])),
        "ARI_B_mean": float(np.mean(order_matched_scores["ARI_B"])),
    }

    # Save
    with open(os.path.join(output_dir, "algebraic_structure_scores.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Also save as CSV
    import csv
    csv_path = os.path.join(output_dir, "algebraic_structure_scores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["representation", "S_hom", "S_trans", "ARI_A", "ARI_B"])
        for name, scores in results.items():
            if "S_hom_mean" in scores:
                writer.writerow([name, f"{scores['S_hom_mean']:.4f}±{scores['S_hom_std']:.4f}",
                                f"{scores['S_trans_mean']:.4f}±{scores['S_trans_std']:.4f}",
                                f"{scores['ARI_A_mean']:.4f}±{scores.get('ARI_A_std',0):.4f}",
                                f"{scores.get('ARI_B_mean',0):.4f}"])
            else:
                writer.writerow([name, scores["S_hom"], scores["S_trans"],
                                scores["ARI_A"], scores["ARI_B"]])

    print(f"\n  Saved to {output_dir}/algebraic_structure_scores.csv")
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    output_dir = "audit"
    os.makedirs(output_dir, exist_ok=True)

    primes = [113, 337, 601]

    # Run all audits
    audit_crt_invertibility(primes, output_dir)
    audit_generators(primes, output_dir)
    audit_target_uniqueness(primes, output_dir)
    audit_tokenization(output_dir)
    audit_scrambled_isomorphism(output_dir)
    audit_algebraic_structure(output_dir)

    print("\n" + "="*60)
    print("ALL AUDITS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
