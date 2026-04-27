"""
Shared dispatch module for CacheToCuda and FuseMatmulToDevice passes.

Both passes return this same _shared_dispatch function object from their
replacement_func(). The route string appended by replacement_args() directs
each call to the appropriate implementation.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
_cuda_cache = {}   # CPU-tensor id -> CUDA copy

_DTYPE_MAP = {
    torch.float16:  tl.float16,
    torch.bfloat16: tl.bfloat16,
    torch.float32:  tl.float32,
}


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------
@triton.jit
def _gemv_m2_kernel(
    A_ptr, B_ptr, C_ptr,
    K, stride_am, stride_bk,
    BLOCK_K: tl.constexpr,
    DTYPE:   tl.constexpr,
):
    """Single-program GEMV for exactly M=2 rows; loads B once for both."""
    k_offs = tl.arange(0, BLOCK_K)
    mask   = k_offs < K
    b  = tl.load(B_ptr + k_offs * stride_bk, mask=mask, other=0.0).to(tl.float32)
    a0 = tl.load(A_ptr + k_offs,             mask=mask, other=0.0).to(tl.float32)
    a1 = tl.load(A_ptr + stride_am + k_offs, mask=mask, other=0.0).to(tl.float32)
    acc0 = tl.sum(a0 * b, axis=0)
    acc1 = tl.sum(a1 * b, axis=0)
    tl.store(C_ptr + 0, acc0.to(DTYPE))
    tl.store(C_ptr + 1, acc1.to(DTYPE))


@triton.jit
def _gemv_general_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K,
    stride_am, stride_ak, stride_bk,
    BLOCK_K: tl.constexpr,
    DTYPE:   tl.constexpr,
):
    """General GEMV: one program per row."""
    row    = tl.program_id(0)
    k_offs = tl.arange(0, BLOCK_K)
    mask   = k_offs < K
    a   = tl.load(A_ptr + row * stride_am + k_offs * stride_ak, mask=mask, other=0.0).to(tl.float32)
    b   = tl.load(B_ptr + k_offs * stride_bk,                   mask=mask, other=0.0).to(tl.float32)
    acc = tl.sum(a * b, axis=0)
    tl.store(C_ptr + row, acc.to(DTYPE))


# ---------------------------------------------------------------------------
# Single shared dispatcher  (returned by replacement_func in BOTH passes)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _shared_dispatch(*args):
    """
    Route 'cache': args = (cpu_tensor, "cache")
        Returns a cached CUDA copy of cpu_tensor.

    Route 'gemv':  args = (in_2, in_3, "gemv")
        Returns the result of in_2 @ in_3 via a Triton GEMV kernel.
    """
    route = args[-1]

    if route == "cache":
        x   = args[0]
        key = id(x)
        if key not in _cuda_cache:
            _cuda_cache[key] = torch.as_tensor(x, device='cuda')
        return _cuda_cache[key]

    # route == "gemv"
    in_2, in_3 = args[0], args[1]
    M = in_2.shape[0]
    K = in_2.shape[1]

    C     = torch.empty((M, 1), dtype=in_2.dtype, device=in_2.device)
    DTYPE = _DTYPE_MAP[in_2.dtype]
    BLOCK_K = 1024 if K <= 1024 else 2048

    if M == 2:
        _gemv_m2_kernel[(1,)](
            in_2, in_3, C,
            K, in_2.stride(0), in_3.stride(0),
            BLOCK_K=BLOCK_K, DTYPE=DTYPE,
        )
    else:
        _gemv_general_kernel[(M,)](
            in_2, in_3, C,
            M, K,
            in_2.stride(0), in_2.stride(1), in_3.stride(0),
            BLOCK_K=BLOCK_K, DTYPE=DTYPE,
        )
    return C