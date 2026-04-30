import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 8, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 8, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_to_flat_view_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    HEADS,
    M,
    N,
    K,
    stride_a0,
    stride_a1,
    stride_a2,
    stride_a3,
    stride_b0,
    stride_b1,
    stride_b2,
    stride_b3,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    OUT_DTYPE: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    pid_bh = tl.program_id(axis=2)

    b = pid_bh // HEADS
    h = pid_bh % HEADS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_base = a_ptr + b * stride_a0 + h * stride_a1
    b_base = b_ptr + b * stride_b0 + h * stride_b1

    a_ptrs = a_base + offs_m[:, None] * stride_a2 + offs_k[None, :] * stride_a3
    b_ptrs = b_base + offs_k[:, None] * stride_b2 + offs_n[None, :] * stride_b3

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_idx = k_start * BLOCK_K + offs_k
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k_idx[None, :] < K), other=0.0)
        b_mat = tl.load(b_ptrs, mask=(k_idx[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b_mat, acc)
        a_ptrs += BLOCK_K * stride_a3
        b_ptrs += BLOCK_K * stride_b2

    out_vals = acc.to(OUT_DTYPE)
    flat_base = pid_bh * M * N
    flat_offsets = flat_base + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptr + flat_offsets, out_vals, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@torch.fx.wrap
def triton_batched_matmul_view(in_0, in_1, out_shape):
    bsz = in_0.shape[0]
    heads = in_0.shape[1]
    k_dim = in_0.shape[2]
    n_dim = in_0.shape[3]
    m_dim = in_1.shape[2]

    out = torch.empty(out_shape, device=in_0.device, dtype=in_0.dtype)

    out_dtype = tl.float32
    if out.dtype == torch.float16:
        out_dtype = tl.float16
    elif out.dtype == torch.bfloat16:
        out_dtype = tl.bfloat16

    grid = lambda META: (
        triton.cdiv(m_dim, META['BLOCK_M']),
        triton.cdiv(n_dim, META['BLOCK_N']),
        bsz * heads,
    )

    _matmul_to_flat_view_kernel[grid](
        in_1,
        in_0,
        out,
        heads,
        m_dim,
        n_dim,
        k_dim,
        in_1.stride(0),
        in_1.stride(1),
        in_1.stride(2),
        in_1.stride(3),
        in_0.stride(0),
        in_0.stride(1),
        in_0.stride(2),
        in_0.stride(3),
        OUT_DTYPE=out_dtype,
    )
    return out