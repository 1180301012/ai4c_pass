"""
Shared Triton kernel for fused causal attention mask computation.

Given input `in_0` of shape [1, N] dtype=int64:
  - causal mask (above diagonal) is already -3.4028234663852886e+38
  - diagonal and below are 0
  - add in_0[j] (0 or 1) per column j:
      combined[j] = causal[j-i] + in_0[j]
        = -inf if j>i
        = in_0[j] if j<=i
  - where combined[j] == 0 (i.e. j<=i AND in_0[j]==0), write -inf
  - final output:
      out[i,j] = -inf  if (j>i) OR (in_0[i]==0)
      out[i,j] = in_0[j]  otherwise  (where j<=i AND in_0[j]==1)

For each row i: mask_out[i] = 1 if ALL N cols in row i are -inf else 0
Then: out[i,j] = 0 when j < N and mask_out[i]==0 (entire row zeros),
      out[i,j] = out[i,j] as above when mask_out[i]==1
"""
import torch
import triton
import triton.language as tl


@triton.jit
def fuse_causal_attn_mask_kernel(
    in_ptr,   # int64 ptr, values from in_0 treated as flat [N]
    out_ptr,  # float32 ptr, output [N*N] elements
    N: tl.constexpr,
):
    # each program handles one row i
    i = tl.program_id(0)
    j_offs = tl.arange(0, N)   # [0 .. N-1]

    # Load causal row i: above diagonal  → -inf, below/below → 0
    causal = tl.where(j_offs > i, -3.4028234663852886e+38, 0.0)

    # Load in_0 values (int64)
    in0 = tl.load(in_ptr + j_offs)   # [N] int64
    combined = causal.to(tl.int64) + in0  # [N] int64

    # Masked positions: where (j>i) OR (in_0[j]==0) → -inf
    masked = j_offs > i | (in0 == 0)
    out_val = tl.where(masked, -3.4028234663852886e+38, in0.to(tl.float32))

    # Row-check mask: True iff all j in [0, N-1] are masked → full row is -inf
    full_masked = tl.sum(tl.where(j_offs < N, tl.cast(masked & (in0 != 0), tl.int32), 0), axis=0) == N

    # Zero out entire rows that have no valid non-mask position
    effective = tl.where(full_masked, 0.0, 1.0)
    out_val = out_val * effective.to(tl.float32)

    tl.store(out_ptr + i * N + j_offs, out_val)