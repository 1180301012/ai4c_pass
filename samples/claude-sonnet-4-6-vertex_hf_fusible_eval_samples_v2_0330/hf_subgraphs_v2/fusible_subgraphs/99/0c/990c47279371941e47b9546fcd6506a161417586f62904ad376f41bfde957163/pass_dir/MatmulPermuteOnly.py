"""
Optimized pass: fused matmul + permute(0,2,1).
Matches:  torch.matmul(softmax_out, in_1).permute(0, 2, 1)
Input `a` = softmax output, `b` = value tensor.

Shapes: a[B, 8192, 19], b[B, 19, 256], out[B, 256, 8192]
"""
import torch
import triton
import triton.language as tl


def pattern(a, b):
    result = torch.matmul(a, b)
    out = result.permute(0, 2, 1)
    return out


def replacement_args(a, b):
    return (a, b)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128}, num_warps=4, num_stages=2),
    ],
    key=['B', 'M', 'K', 'N'],
)
@triton.jit
def matmul_permute_kernel(
    a_ptr, b_ptr, out_ptr,
    B, M, K, N,
    stride_a_b, stride_a_m, stride_a_k,
    stride_b_b, stride_b_k, stride_b_n,
    stride_out_b, stride_out_n, stride_out_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_PADDED: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    k_offs = tl.arange(0, K_PADDED)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offs < M
    k_mask = k_offs < K
    n_mask = n_offs < N

    a_offs = (pid_b * stride_a_b
              + m_offs[:, None] * stride_a_m
              + k_offs[None, :] * stride_a_k)
    a = tl.load(a_ptr + a_offs, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

    b_offs = (pid_b * stride_b_b
              + k_offs[:, None] * stride_b_k
              + n_offs[None, :] * stride_b_n)
    b = tl.load(b_ptr + b_offs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)

    # Native dtype for fp16/bf16 tensor core path; fp32 accumulation
    result = tl.dot(a, b, out_dtype=tl.float32, allow_tf32=True)

    result_t = tl.trans(result)
    out_offs = (pid_b * stride_out_b
                + n_offs[:, None] * stride_out_n
                + m_offs[None, :] * stride_out_m)
    tl.store(out_ptr + out_offs, result_t, mask=n_mask[:, None] & m_mask[None, :])


@torch.fx.wrap
def matmul_permute(a, b):
    B, M, K = a.shape
    _,  _, N = b.shape
    K_PADDED = max(16, 1 << (max(K, 1) - 1).bit_length())
    out = torch.empty((B, N, M), dtype=a.dtype, device=a.device)

    def grid(meta):
        return (B, triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    matmul_permute_kernel[grid](
        a, b, out,
        B, M, K, N,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        K_PADDED=K_PADDED,
    )
    return out


def replacement_func():
    return matmul_permute