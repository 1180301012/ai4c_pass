"""
Shared Triton kernel for fused linear + elementwise scale.
Computes: output = linear(input, weight) * scale
where linear uses torch.nn.functional.linear semantics (input @ weight.T).
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32,  'BLOCK_K': 16}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_scale_fused_kernel(
    A_ptr, B_ptr, scale_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    OUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Computes C = (A @ B^T) * scale
    A: [M, K], B: [N, K] (weight matrix stored as [out_features, in_features]),
    scale: [N], C: [M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulate in float32 for precision
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        # Load A[offs_m, offs_k] – shape [BLOCK_M, BLOCK_K]
        a = tl.load(
            A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )

        # Load B[offs_n, offs_k] transposed → shape [BLOCK_K, BLOCK_N]
        b_T = tl.load(
            B_ptr + offs_k[:, None] * stride_bn + offs_n[None, :] * stride_bk,
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0,
        )

        # Accumulate: acc += A @ B^T (both loaded as original dtype)
        acc += tl.dot(a, b_T)

    # Load scale[offs_n] – shape [BLOCK_N]
    scale = tl.load(scale_ptr + offs_n, mask=offs_n < N, other=1.0)

    # Apply elementwise scale (broadcast acc[BLOCK_M, BLOCK_N] over M)
    acc = acc * scale[None, :]

    # Cast to output dtype and store
    if OUT_DTYPE == tl.float16:
        c = acc.to(tl.float16)
    elif OUT_DTYPE == tl.bfloat16:
        c = acc.to(tl.bfloat16)
    else:
        c = acc  # float32

    tl.store(
        C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
        c,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@torch.fx.wrap
def linear_scale_fused(weight, scale, x):
    """
    weight: [N, K]
    scale:  [N]
    x:      [..., K]  (arbitrary leading batch dimensions)
    returns: [..., N]  (same leading dims as x, last dim N)

    Computes: x @ weight.T * scale  (element-wise, scale broadcasts over batch)
    """
    leading = x.shape[:-1]
    K = x.shape[-1]
    M = x.numel() // K
    N = weight.shape[0]

    A = x.reshape(M, K)          # [M, K]
    C = torch.empty((M, N), dtype=x.dtype, device=x.device)

    if x.dtype == torch.float16:
        OUT_DTYPE = tl.float16
    elif x.dtype == torch.bfloat16:
        OUT_DTYPE = tl.bfloat16
    else:
        OUT_DTYPE = tl.float32

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    linear_scale_fused_kernel[grid](
        A, weight, scale, C,
        M, N, K,
        A.stride(0), A.stride(1),
        weight.stride(0), weight.stride(1),
        C.stride(0), C.stride(1),
        OUT_DTYPE=OUT_DTYPE,
    )

    return C.reshape(leading + (N,))