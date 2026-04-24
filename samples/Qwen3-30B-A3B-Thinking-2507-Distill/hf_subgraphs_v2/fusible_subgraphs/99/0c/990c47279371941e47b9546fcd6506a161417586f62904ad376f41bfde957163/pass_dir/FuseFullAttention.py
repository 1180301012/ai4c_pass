import torch
import triton
import triton.language as tl


# Full 4-op fusion: scale + softmax + matmul + permute
def pattern(in_0, in_1):
    tmp_0 = 0.0625 * in_0
    tmp_1 = tmp_0.softmax(dim=-1)
    matmul = torch.matmul(tmp_1, in_1)
    tmp_3 = matmul.permute(0, 2, 1)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 16}, num_warps=4),
        triton.Config({'BLOCK_N': 32}, num_warps=4),
        triton.Config({'BLOCK_N': 32}, num_warps=8),
        triton.Config({'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_N': 64}, num_warps=8),
        triton.Config({'BLOCK_N': 128}, num_warps=8),
    ],
    key=['B', 'N', 'K', 'C'],
)
@triton.jit
def full_fusion_kernel(
    in0_ptr, in1_ptr, out_ptr,
    B, N, K, C,
    stride_in0_b, stride_in0_n, stride_in0_k,
    stride_in1_b, stride_in1_k, stride_in1_c,
    stride_out_b, stride_out_c, stride_out_n,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    k_offs = tl.arange(0, BLOCK_K)
    c_offs = tl.arange(0, BLOCK_C)

    # Load in_0[b, n_offs, :K] and apply scale
    x = tl.load(
        in0_ptr + pid_b * stride_in0_b + n_offs[:, None] * stride_in0_n + k_offs[None, :],
        mask=(n_mask[:, None] & (k_offs[None, :] < K)),
        other=0.0,
    ).to(tl.float32)
    x = x * 0.0625

    # Softmax: max-subtract + exp + normalize
    x_max = tl.max(x, axis=1)[:, None]
    x = x - x_max
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=1)[:, None]
    sm = exp_x / sum_exp

    # Load in_1[b, :K, :C]
    in1 = tl.load(
        in1_ptr + pid_b * stride_in1_b + k_offs[:, None] * stride_in1_k + c_offs[None, :] * stride_in1_c,
        mask=(k_offs[:, None] < K) & (c_offs[None, :] < C),
        other=0.0,
    ).to(tl.float32)

    # Matmul + store transposed
    acc = tl.dot(sm, in1, out_dtype=tl.float32, allow_tf32=True)

    out_ptrs = (
        out_ptr
        + pid_b * stride_out_b
        + c_offs[:, None] * stride_out_c
        + n_offs[None, :] * stride_out_n
    )
    tl.store(
        out_ptrs,
        acc.to(out_ptr.dtype.element_ty).T,
        mask=(c_offs[:, None] < C) & (n_offs[None, :] < N),
    )


@torch.fx.wrap
def fused_scale_softmax_matmul_permute(in_0, in_1):
    B, N, K = in_0.shape
    _, K2, C = in_1.shape
    out = torch.empty((B, C, N), dtype=in_0.dtype, device=in_0.device)
    BLOCK_K = triton.next_power_of_2(max(K, 16))

    def grid(meta):
        return (triton.cdiv(N, meta['BLOCK_N']), B)

    full_fusion_kernel[grid](
        in_0, in_1, out,
        B, N, K, C,
        in_0.stride(0), in_0.stride(1), in_0.stride(2),
        in_1.stride(0), in_1.stride(1), in_1.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_K=BLOCK_K,
        BLOCK_C=256,
    )
    return out


def replacement_func():
    return fused_scale_softmax_matmul_permute