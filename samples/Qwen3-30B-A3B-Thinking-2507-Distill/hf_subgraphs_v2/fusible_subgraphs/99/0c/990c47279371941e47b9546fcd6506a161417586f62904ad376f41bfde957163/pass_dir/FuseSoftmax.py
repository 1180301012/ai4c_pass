import torch
import triton
import triton.language as tl


# Try softmax as a method call (call_method style, matching permute approach)
def pattern(x):
    return x.softmax(dim=-1)


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 128}, num_warps=8),
        triton.Config({'BLOCK_N': 256}, num_warps=8),
    ],
    key=['N', 'K'],
)
@triton.jit
def softmax_kernel(
    in_ptr, out_ptr,
    B, N, K,
    stride_b, stride_n, stride_k,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    n_start = pid_n * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < N
    k_offs = tl.arange(0, BLOCK_K)

    # Load row tile: in[b, n_offs, :K]
    x = tl.load(
        in_ptr + pid_b * stride_b + n_offs[:, None] * stride_n + k_offs[None, :],
        mask=(n_mask[:, None] & (k_offs[None, :] < K)),
        other=-float('inf'),
    ).to(tl.float32)

    # Numerically stable softmax
    x_max = tl.max(x, axis=1)[:, None]
    x = x - x_max
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=1)[:, None]
    out = exp_x / sum_exp

    # Store (cast back to original dtype)
    tl.store(
        out_ptr + pid_b * stride_b + n_offs[:, None] * stride_n + k_offs[None, :],
        out.to(out_ptr.dtype.element_ty),
        mask=(n_mask[:, None] & (k_offs[None, :] < K)),
    )


@torch.fx.wrap
def fused_softmax(x):
    B, N, K = x.shape
    out = torch.empty_like(x)
    BLOCK_K = triton.next_power_of_2(max(K, 16))

    def grid(meta):
        return (triton.cdiv(N, meta['BLOCK_N']), B)

    softmax_kernel[grid](
        x, out,
        B, N, K,
        x.stride(0), x.stride(1), x.stride(2),
        BLOCK_K=BLOCK_K,
    )
    return out


def replacement_func():
    return fused_softmax