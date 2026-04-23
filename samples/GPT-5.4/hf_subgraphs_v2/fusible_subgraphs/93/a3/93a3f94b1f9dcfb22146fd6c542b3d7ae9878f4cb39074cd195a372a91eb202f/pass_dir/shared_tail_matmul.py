import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'num_warps': 8, 'num_stages': 2}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_warps': 8, 'num_stages': 3}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_tail_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_az,
    stride_ah,
    stride_am,
    stride_ak,
    stride_bz,
    stride_bh,
    stride_bk,
    stride_bn,
    stride_oz,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_hn = tl.program_id(1)
    pid_z = tl.program_id(2)

    num_n_blocks = tl.cdiv(N, BLOCK_N)
    h = pid_hn // num_n_blocks
    pid_n = pid_hn % num_n_blocks

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_head_ptr = a_ptr + pid_z * stride_az + h * stride_ah
    b_head_ptr = b_ptr + pid_z * stride_bz + h * stride_bh

    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        a_ptrs = a_head_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak
        b_ptrs = b_head_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b).to(tl.float32)

    out_col = h * N + offs_n
    out_ptrs = out_ptr + pid_z * stride_oz + offs_m[:, None] * stride_om + out_col[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@torch.fx.wrap
def fused_matmul_tail(attn, value):
    batch = attn.shape[0]
    heads = attn.shape[1]
    m = attn.shape[2]
    k = attn.shape[3]
    n = value.shape[3]

    out = torch.empty((batch, m, heads * n), device=attn.device, dtype=attn.dtype)
    grid = lambda META: (
        triton.cdiv(m, META['BLOCK_M']),
        heads * triton.cdiv(n, META['BLOCK_N']),
        batch,
    )
    _matmul_tail_kernel[grid](
        attn,
        value,
        out,
        m,
        n,
        k,
        attn.stride(0),
        attn.stride(1),
        attn.stride(2),
        attn.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )
    return out


def replacement_func():
    return fused_matmul_tail