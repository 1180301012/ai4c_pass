"""
Shared Triton GEMM+bias kernel and dispatch wrapper for all linear fusion passes.

All patterns share this single replacement_func to stay within replacement_func_limit.
Routing is done via a route-str argument appended by each pass's replacement_args().

Shapes handled:
  BigBird: A=[1,17,768], B=[3072,768], bias=[3072]  →  C=[1,17,3072]
  RECT_L:  A=[128,128],  B=[128,128],  bias=[128]   →  C=[128,128]

C = A @ B^T + bias
A stored row-major  [M, K] with strides (stride_am, stride_ak)
B stored row-major  [N, K] with strides (stride_bn, stride_bk)  (= weight^T view)
bias: [N]
C stored row-major  [M, N] with strides (N, 1)
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Configs for BigBird (M=17, N=3072, K=768) — small M, large N/K
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 128, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32,  'num_stages': 4, 'num_warps': 8}),
        # Configs for RECT_L (M=128, N=128, K=128) — tiny square GEMM
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 16, 'num_stages': 3, 'num_warps': 2}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _linear_gemm_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    IS_FP16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C = A @ B^T + bias
    A: [M, K], B: [N, K] (weight rows), C: [M, N], bias: [N]
    Standard layout: load B as [BLOCK_N, BLOCK_K] rows (contiguous) then transpose for tl.dot.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K]  — row-major, contiguous rows
        a_ptrs = a_ptr + m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak
        a = tl.load(a_ptrs,
                    mask=(m_offs[:, None] < M) & (k_offs[None, :] < K),
                    other=0.0)

        # B loaded as [BLOCK_N, BLOCK_K] — weight rows (contiguous in K)
        # tl.trans(b) turns it into [BLOCK_K, BLOCK_N] for tl.dot
        b_ptrs = b_ptr + n_offs[:, None] * stride_bn + k_offs[None, :] * stride_bk
        b = tl.load(b_ptrs,
                    mask=(n_offs[:, None] < N) & (k_offs[None, :] < K),
                    other=0.0)

        # a [BLOCK_M, BLOCK_K] @ trans(b) [BLOCK_K, BLOCK_N] → [BLOCK_M, BLOCK_N]
        acc += tl.dot(a, tl.trans(b), out_dtype=tl.float32)

    # Fused bias add
    bias = tl.load(bias_ptr + n_offs, mask=n_offs < N, other=0.0)
    acc += bias[None, :].to(tl.float32)

    # Store with optional dtype conversion
    c_ptrs = c_ptr + m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn
    if IS_FP16:
        tl.store(c_ptrs, acc.to(tl.float16),
                 mask=(m_offs[:, None] < M) & (n_offs[None, :] < N))
    else:
        tl.store(c_ptrs, acc.to(tl.bfloat16),
                 mask=(m_offs[:, None] < M) & (n_offs[None, :] < N))


@torch.fx.wrap
def _dispatch_linear(in_0, in_1, in_2, route):
    """
    Shared dispatch wrapper for all linear fusion passes.
    in_0 : bias   [N]
    in_1 : weight [N, K]
    in_2 : input  [*, K]  (any number of leading dims)
    route: constant string selecting IS_FP16 code-path
    """
    # K is the last-dimension size
    K = in_2.shape[-1]
    # M = total rows in the 2-D logical view
    M = in_2.numel() // K
    N = in_1.shape[0]

    # Strides for the 2-D logical view of the input (assume contiguous)
    stride_am = in_2.stride(-2)   # = K for contiguous
    stride_ak = in_2.stride(-1)   # = 1 for contiguous
    stride_bn = in_1.stride(0)    # = K for contiguous weight
    stride_bk = in_1.stride(1)    # = 1 for contiguous weight

    # Pre-convert weight/bias to same device & dtype as input
    # (no-op if already there; guards against CPU→GPU mismatches)
    dev = in_2.device
    dt  = in_2.dtype
    b    = in_1.to(device=dev, dtype=dt)
    bias = in_0.to(device=dev, dtype=dt)

    # Allocate output [*, N]
    out_shape = list(in_2.shape[:-1]) + [N]
    c = torch.empty(out_shape, dtype=dt, device=dev)

    # Contiguous output strides
    stride_cm = c.stride(-2)   # = N
    stride_cn = c.stride(-1)   # = 1

    # Fixed config: optimal for M~17-128, N~128-3072, K~128-768 on A30
    # (autotune will select the best from its own configs)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),
                         triton.cdiv(N, meta['BLOCK_N']))

    if route == "bf16":
        _linear_gemm_bias_kernel[grid](
            in_2, b, bias, c,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
            IS_FP16=False,
        )
    elif route == "fp16":
        _linear_gemm_bias_kernel[grid](
            in_2, b, bias, c,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
            IS_FP16=True,
        )
    elif route == "rect_fp16":
        _linear_gemm_bias_kernel[grid](
            in_2, b, bias, c,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
            IS_FP16=True,
        )
    else:  # "rect_bf16"
        _linear_gemm_bias_kernel[grid](
            in_2, b, bias, c,
            M, N, K,
            stride_am, stride_ak,
            stride_bn, stride_bk,
            stride_cm, stride_cn,
            IS_FP16=False,
        )

    return c