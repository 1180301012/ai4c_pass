import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: matches the entire subgraph in model.py
#   cumsum  →  mul(input)  →  sub(1)  →  long()  →  slice[:,0:]  →  add(2)
# ──────────────────────────────────────────────────────────────────────────────
def pattern(in_0):
    tmp_1 = torch.cumsum(in_0, dim=1)
    tmp_2 = tmp_1 * in_0
    tmp_3 = tmp_2 - 1
    tmp_4 = tmp_3.long()
    tmp_5 = tmp_4[slice(None, None, None), slice(0, None, None)]
    tmp_6 = tmp_5 + 2
    return tmp_6


def replacement_args(in_0):
    return (in_0,)


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel: fused  cumsum * x + 1  (all in one pass, int64)
#   For each row:
#     cumsum[j] = sum(x[0..j])
#     out[j]    = cumsum[j] * x[j] + 1
#
# BLOCK_N must be a power-of-2 (required by tl.cumsum).
# For the reference shape [1, 13] we use BLOCK_N=16.
# ──────────────────────────────────────────────────────────────────────────────
@triton.jit
def fused_cumsum_mul_kernel(
    in_ptr,
    out_ptr,
    B,          # number of rows
    N,          # number of columns  (runtime value, used for mask)
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N

    # Load one full row (padding masked positions with 0 so cumsum is correct)
    x = tl.load(in_ptr + row * N + offsets, mask=mask, other=0)

    # Inclusive prefix-sum along the block (requires power-of-2 BLOCK_N)
    cumsum = tl.cumsum(x, axis=0)

    # Fused arithmetic:  (cumsum * x - 1) + 2  =  cumsum * x + 1
    out = cumsum * x + 1

    tl.store(out_ptr + row * N + offsets, out, mask=mask)


@torch.fx.wrap
def fused_cumsum_mul(in_0):
    B = in_0.shape[0]
    N = in_0.shape[1]

    out = torch.empty((B, N), dtype=torch.int64, device=in_0.device)

    # BLOCK_N must be a compile-time power-of-2; compute it in Python
    BLOCK_N = 1
    while BLOCK_N < N:
        BLOCK_N <<= 1
    # Minimum block size of 16 to avoid Triton edge-cases on tiny inputs
    BLOCK_N = max(BLOCK_N, 16)

    fused_cumsum_mul_kernel[(B,)](
        in_0,
        out,
        B,
        N,
        BLOCK_N=BLOCK_N,
    )
    return out


def replacement_func():
    return fused_cumsum_mul