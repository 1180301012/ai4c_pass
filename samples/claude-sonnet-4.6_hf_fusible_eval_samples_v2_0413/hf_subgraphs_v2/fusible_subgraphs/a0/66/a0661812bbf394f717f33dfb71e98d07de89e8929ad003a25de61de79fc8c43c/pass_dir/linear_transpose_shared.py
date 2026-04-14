import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Medium shapes: M~249, N~768-1024, K=512
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        # Small N=16 case (M~1248, N=16, K=32)
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16,  'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 16,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _gemm_bias_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xb, stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ob,  stride_om,  stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Batched GEMM+bias: out[b,m,n] = sum_k x[b,m,k]*w[n,k] + bias[n]

    Loads w as [BLOCK_N, BLOCK_K] (coalesced on K dim) then transposes for tl.dot.
    """
    pid_b = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Pre-compute base pointers to reduce address recomputation per iteration
    x_base = x_ptr + pid_b * stride_xb + offs_m[:, None] * stride_xm
    w_base = w_ptr + offs_n[:, None] * stride_wn

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K

        # x: [BLOCK_M, BLOCK_K] — coalesced on K dim
        x = tl.load(x_base + offs_k[None, :] * stride_xk,
                    mask=(offs_m[:, None] < M) & k_mask[None, :],
                    other=0.0)

        # w: [BLOCK_N, BLOCK_K] — coalesced on K dim (stride_wk=1)
        w = tl.load(w_base + offs_k[None, :] * stride_wk,
                    mask=(offs_n[:, None] < N) & k_mask[None, :],
                    other=0.0)

        # Use accumulator form for better software-pipeline interaction
        acc = tl.dot(x, tl.trans(w), acc)

    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += b[None, :].to(tl.float32)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = (out_ptr
                + pid_b * stride_ob
                + offs_m[:, None] * stride_om
                + offs_n[None, :] * stride_on)
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=mask)


@torch.fx.wrap
def fused_linear(bias, weight, x):
    """
    Fused linear: out = x @ weight.T + bias, using autotuned Triton GEMM.
    Returns a single tensor [B, M, N].  The transpose(1,2) op stays in graph.
    """
    B, M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((B, M, N), dtype=x.dtype, device=x.device)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
        B,
    )

    _gemm_bias_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0),      x.stride(1),      x.stride(2),
        weight.stride(0), weight.stride(1),
        out.stride(0),    out.stride(1),    out.stride(2),
    )
    return out