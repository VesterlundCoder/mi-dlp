# GROKKING METRIC AUDIT

## Audit 20: Define Grokking Time Rigorously

**Status**: ADDRESSED

The LUMI trainer uses early stopping with:
- Threshold: test_acc > 0.99
- Patience: 200 consecutive evaluations above threshold
- Evaluation interval: 50 epochs

**Definitions**:
- T_mem: First epoch where train_acc > 0.99 (memorization)
- T_grok: First epoch where test_acc > 0.99 sustained for 200 evaluations
- T_grok,95: First epoch where test_acc ≥ 0.95 (stable)
- Total epochs: May be much larger than T_grok if early stopping doesn't trigger

**Important distinction**:
- "total_epochs" in config = configured maximum epochs
- Actual stopping epoch may be much earlier (early stopping)
- T_grok is the scientifically meaningful quantity

**Reconstruction from logs**: The metrics.jsonl files contain per-evaluation
train_acc, test_acc, and loss, allowing exact reconstruction of T_mem
and T_grok.

---

## Audit 21: Multiple Compute Axes

**Status**: PARTIALLY ADDRESSED

Different representations have different sequence lengths:
- RAW: 5 tokens
- CRT-BOTH: 9 tokens
- RAW+CRT: 13 tokens

This means the same number of epochs corresponds to different numbers
of tokens processed.

**Required metrics**:
- T_grok^steps: Number of optimizer steps (= epochs, since full-batch)
- T_grok^examples: Training examples consumed = T_grok^steps × train_size
- T_grok^tokens: Input tokens consumed = T_grok^examples × seq_len

For p=113, train_size=1612:
| Representation | Seq Len | T_grok (steps) | T_grok (tokens) |
|---------------|---------|----------------|-----------------|
| RAW | 5 | ~50,000 | ~403M |
| CRT-BOTH | 9 | ~20,000 | ~290M |
| SCRAMBLED | 9 | ~16,000 | ~232M |
| R3 | 9 | >8,500 (ongoing) | >123M |

**Note**: Even in token-normalized terms, CRT is faster than RAW
(290M vs 403M tokens). The token normalization does NOT eliminate
the difference.

**Wall-clock**: Hardware-dependent, not scientifically meaningful for
cross-platform comparison.

---

## Early Stopping Behavior

The LUMI trainer saves:
- metrics.jsonl: Per-evaluation metrics
- final.pt: Final checkpoint
- summary.json: Total epochs, best_test_acc, mem_epoch

**CRT-BOTH_p113_s42**: Stopped at epoch 20,400 (early stopped)
**SCRAMBLED_p113_s42**: Stopped at epoch 16,050 (early stopped)
**RAW_p113_s42 (LUMI)**: Ran to 499,999 (did NOT early stop, did not grok)
**M04_p113_s42 (standard)**: Ran to configured max (grokked at ~50K)

**Key insight**: The LUMI RAW models (type="crt", variant="RAW") did NOT
grok despite running for 500K epochs. The M04 standard models (type="standard")
DID grok. The difference may be due to max_len (13 vs 5) or other config
differences. This needs investigation.
