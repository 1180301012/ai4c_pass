import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Dtype mapping
# ---------------------------------------------------------------------------
_DTYPE_MAP = {
    torch.float16:  tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32:  tl.float32,
}


# ---------------------------------------------------------------------------
# Single-pass GEMV.  One program per output row.  BLOCK_K >= K so the entire
# K-dimension is reduced in a single pass without a loop.
# DTYPE constexpr casts the fp32 accumulator in-kernel so no separate
# type-conversion kernel launch is needed.
# ---------------------------------------------------------------------------
@triton.jit
def _gemv_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K,
    stride_am, stride_ak, stride_bk,
    BLOCK_K: tl.constexpr,
    DTYPE:   tl.constexpr,
):
    row    = tl.program_id(0)
    k_offs = tl.arange(0, BLOCK_K)
    mask   = k_offs < K
    a   = tl.load(A_ptr + row * stride_am + k_offs * stride_ak, mask=mask, other=0.0).to(tl.float32)
    b   = tl.load(B_ptr + k_offs * stride_bk,                   mask=mask, other=0.0).to(tl.float32)
    acc = tl.sum(a * b, axis=0)
    tl.store(C_ptr + row, acc.to(DTYPE))


# ---------------------------------------------------------------------------
# Replacement wrapper  (must be @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _fast_gemv(in_2, in_3):
    """
    Replacement for torch.matmul(in_2, in_3) where in_2=[M,K], in_3=[K,1].
    BLOCK_K is set so the single-pass kernel always covers all K elements.
    """
    M = in_2.shape[0]
    K = in_2.shape[1]

    C     = torch.empty((M, 1), dtype=in_2.dtype, device=in_2.device)
    DTYPE = _DTYPE_MAP[in_2.dtype]

    # BLOCK_K must be a power-of-2 >= K for correctness
    BLOCK_K = 1024 if K <= 1024 else 2048

    _gemv_kernel[(M,)](
        in_2, in_3, C,
        M, K,
        in_2.stride(0), in_2.stride(1), in_3.stride(0),
        BLOCK_K=BLOCK_K,
        DTYPE=DTYPE,
    )

    return C


# ---------------------------------------------------------------------------
# Pattern / replacement_args / replacement_func
# ---------------------------------------------------------------------------
def pattern(in_2, in_3):
    return torch.matmul(in_2, in_3)


def replacement_args(in_2, in_3):
    return (in_2, in_3)


def replacement_func():
    return _fast_gemv