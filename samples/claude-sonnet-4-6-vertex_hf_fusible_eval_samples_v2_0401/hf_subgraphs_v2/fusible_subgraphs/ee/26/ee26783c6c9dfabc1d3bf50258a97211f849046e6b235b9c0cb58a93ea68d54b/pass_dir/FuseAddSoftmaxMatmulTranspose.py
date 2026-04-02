import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: (a + b).softmax(dim=-1) @ v  then  .transpose(-1, -2)
# Matches all four graph variants (N=256 or N=64, float16 or bfloat16).
# ──────────────────────────────────────────────────────────────────────────────

def pattern(a, b, v):
    x = a + b
    s = x.softmax(dim=-1)
    m = s @ v
    t = m.transpose(-1, -2)
    return t


def replacement_args(a, b, v):
    return (a, b, v)


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel: fused element-wise add + numerically-stable row softmax.
#
#   Input : A [rows, N],  B [rows, N]   (contiguous)
#   Output: softmax(A + B, dim=-1)      [rows, N]
#
#   Grid  : [rows]  — one program per row.
#   BLOCK_N autotuned; promotes fp32 for numerical stability.
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _add_softmax_kernel(
    A_ptr, B_ptr, Out_ptr,
    N,            # row length  (runtime)
    stride_row,   # row stride  (= N for contiguous tensors)
    BLOCK_N: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    base = row * stride_row

    a = tl.load(A_ptr + base + offs, mask=mask, other=float('-inf'))
    b = tl.load(B_ptr + base + offs, mask=mask, other=float('-inf'))

    x     = a.to(tl.float32) + b.to(tl.float32)
    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    out   = (x_exp / x_sum).to(a.dtype)

    tl.store(Out_ptr + base + offs, out, mask=mask)


# ──────────────────────────────────────────────────────────────────────────────
# Python wrapper
# ──────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def fused_add_softmax_matmul_T(a, b, v):
    """
    Computes (a + b).softmax(dim=-1) @ v, then transposes the last two dims.

    Stage 1: Triton fused add+softmax  — one kernel instead of two native ops,
             saves one [B,M,N] intermediate materialization.
    Stage 2: native @ operator         — dispatches to cuBLAS batched GEMM.
    Stage 3: .transpose(-1, -2)        — free non-contiguous view.

    a, b : [B, M, N]  (float16 or bfloat16)
    v    : [B, N, D]
    returns [B, D, M]  (non-contiguous view)
    """
    a = a.contiguous()
    b = b.contiguous()
    v = v.contiguous()

    B, M, N = a.shape

    soft = torch.empty_like(a)
    # Choose BLOCK_N: smallest power of 2 >= N to avoid excess wasted lanes.
    # For N=256 → BLOCK_N=256 (8 warps); for N=64 → BLOCK_N=64 (4 warps).
    if N <= 64:
        _add_softmax_kernel[(B * M,)](a, b, soft, N=N, stride_row=N,
                                      BLOCK_N=64, num_warps=4)
    else:
        _add_softmax_kernel[(B * M,)](a, b, soft, N=N, stride_row=N,
                                      BLOCK_N=256, num_warps=8)

    result = soft @ v
    return result.transpose(-1, -2)


def replacement_func():
    return fused_add_softmax_matmul_T