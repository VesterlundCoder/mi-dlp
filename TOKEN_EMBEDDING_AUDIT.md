# TOKEN EMBEDDING AUDIT

## Audit 7: Token Values Have No Numeric Meaning

**Finding**: Token IDs are purely categorical.

The model architecture is:
```python
self.embed = nn.Embedding(vocab_size, d_model)  # learned lookup table
self.pos_embed = nn.Embedding(max_len, d_model)  # learned position
self.unembed = nn.Linear(d_model, vocab_size)  # separate output
```

The tokenization function is `tok(v) = v + INT_OFFSET` where INT_OFFSET=4.
Special tokens: PAD=0, SEP=1, BOS=2, EOS=3.

**No numeric encoding**: There is no:
- Binary digit encoding of token values
- Scalar normalization of token values
- Sinusoidal encoding based on token ID
- Polynomial features of token IDs
- Arithmetic operations on token IDs
- Ordinal distance between IDs

**Conclusion**: The numeric labels assigned to group elements have no
inherent metric or arithmetic meaning before training. The model must
learn all structure from the embedding table.

---

## Audit 8: Weight Tying

**Finding**: Input embeddings and output classifier are NOT tied.

- Input: `nn.Embedding(vocab_size, d_model)` — separate parameters
- Output: `nn.Linear(d_model, vocab_size)` — separate parameters
- No weight sharing, no projection layer between them

**Implication**: An arbitrary input vocabulary permutation can be
absorbed exactly by permuting embedding rows, WITHOUT affecting the
output classifier. This means SCRAMBLED is exactly conjugate to CRT
at initialization.

---

## Audit 9: Token Namespace

**Finding**: All tokens share the SAME namespace.

`tok(v) = v + INT_OFFSET` is used for:
- Raw group elements (g, h): values 1..p-1 → tokens 5..p+3
- CRT projection values: values 1..p-1 → tokens 5..p+3
- SCRAMBLED values: values 0..p-1 → tokens 4..p+3
- Output exponent labels (x): values 0..N-1 → tokens 4..N+3

**Critical implication**: Group element 5 and output label 5 share the
SAME learned embedding row (both map to token 5+INT_OFFSET=9).

This creates unintended input-output coupling:
- When the model learns to predict x=5, it uses the same embedding
  row that represents group element 5 in the input.
- This may help or hurt depending on whether the input-output
  relationship aligns with the shared representation.

**Recommendation**: Future experiments should use namespace-separated
embeddings (GROUP, FACTOR-A, FACTOR-B, TARGET).

---

## Audit 10: Zero Element Issue

**Finding**: No zero-mappings occur in practice for seed 12345.

SCRAMBLED's permutation is defined over 0..p-1, but CRT projections
never produce 0 (since F_p* is closed under multiplication, and
0 ∉ F_p*). The permutation CAN map to 0 in principle.

For seed 12345 with p=113:
- 0 projected values are mapped to 0 (verified computationally)
- Token 0+INT_OFFSET = 4 is in the vocabulary but never used by CRT-BOTH
- SCRAMBLED may use token 4 if any projected value maps to 0

**Impact**: Minor vocabulary mismatch. Future controls should permute
only the actual support G = {1, ..., p-1}.
