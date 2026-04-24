import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matmul(in_2, in_1) * in_0  — matmul + scale
# Shapes: in_2=[M,K], in_1=[K,1], in_0=scalar  → out=[M,1]
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_2, in_1)
    tmp_1 = matmul * in_0
    return tmp_1


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel: single-CTA 2D fused GEMV + scale
#
# Uses a single CTA that handles all M rows at once via 2D tile loading.
# Benefits:
#   • Load y (in_1) only ONCE instead of M times
#   • Single kernel launch (less overhead than 2 separate launches)
#   • Fully coalesced memory accesses
#
# Memory layout (all row-major, contiguous):
#   in_2 [M, K]  → in2_ptr + row*K + k
#   in_1 [K, 1]  → in1_ptr + k           (stride-1 along k)
#   in_0 scalar  → in0_ptr
#   out  [M, 1]  → out_ptr + row         (element [row,0])
#
# BLOCK_M = M (2), BLOCK_K = K (1024).  No masking needed.
# Grid: (1,)  — exactly one CTA.
# ---------------------------------------------------------------------------

@triton.jit
def _fused_gemv_scale_kernel(
    in0_ptr,
    in1_ptr,
    in2_ptr,
    out_ptr,
    M, K,
    BLOCK_M: tl.constexpr,   # = M = 2
    BLOCK_K: tl.constexpr,   # = K = 1024
):
    # ---- load scalar scale once ----
    scale = tl.load(in0_ptr)

    # ---- 2D tile: [BLOCK_M, BLOCK_K] ----
    row_off = tl.arange(0, BLOCK_M)[:, None]   # [M, 1]
    col_off = tl.arange(0, BLOCK_K)[None, :]   # [1, K]

    x = tl.load(in2_ptr + row_off * K + col_off)   # [M, K]  – coalesced rows
    y = tl.load(in1_ptr + col_off)                  # [1, K]  – one shot

    # ---- compute M dot products in parallel ----
    dot = tl.sum(x.to(tl.float32) * y.to(tl.float32), axis=1)  # [M]
    result = dot * scale                                          # [M]

    # ---- store all results ----
    tl.store(out_ptr + tl.arange(0, BLOCK_M), result)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_matmul_scale(in_0, in_1, in_2):
    M, K = in_2.shape
    N = in_1.shape[1]   # always 1

    out = torch.empty((M, N), dtype=in_2.dtype, device=in_2.device)

    # Single CTA covers all M rows; BLOCK_K covers all K cols in one pass
    _fused_gemv_scale_kernel[(1,)](
        in0_ptr=in_0,
        in1_ptr=in_1,
        in2_ptr=in_2,
        out_ptr=out,
        M=M, K=K,
        BLOCK_M=M,
        BLOCK_K=K,
        num_warps=4,
    )

    return out


def replacement_func():
    return fused_matmul_scale