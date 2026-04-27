import torch
import triton
import triton.language as tl


@triton.jit
def fuse_attn_mask_kernel_flat(
    in_ptr,          # (1, N) int64  — attention mask: 1=valid, 0=masked
    out_ptr,         # (1, 1, N, N) float32 — output combined mask
    N: tl.constexpr,
    N2: tl.constexpr,  # N * N
    BLOCK: tl.constexpr,
):
    """
    Single-program flat kernel.
    For each position (i,j) at flat index offs = i*N + j:
      out[offs] = NEG_INF  if (j > i)  [causal: future token]
                           or (in_0[0,j] == 0)  [attention: masked token]
               = 0.0       otherwise
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N2

    i = offs // N    # row
    j = offs % N     # col

    # in_0[0, j] = in_ptr[j]  (shape (1,N) contiguous)
    attn = tl.load(in_ptr + j, mask=mask, other=1)   # int64; default 1 = valid

    causal_invalid = j > i
    attn_invalid   = attn == 0
    invalid        = causal_invalid | attn_invalid

    NEG_INF = -3.4028234663852886e+38
    val = tl.where(invalid, NEG_INF, 0.0)

    tl.store(out_ptr + offs, val.to(tl.float32), mask=mask)