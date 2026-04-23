import torch
import triton
import triton.language as tl

# Pattern matching function - must mirror model.py exactly (without cleanup statements)
def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.reshape(linear, [-1, 9, 1])
    tmp_4 = torch.softmax(tmp_3, dim=1)
    return (tmp_4,)

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# Fused Triton kernel: linear + reshape + softmax
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 32}, num_warps=2),
        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 64}, num_warps=2),
        triton.Config({'BLOCK_M': 2, 'BLOCK_K': 32}, num_warps=2),
        triton.Config({'BLOCK_M': 2, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 4, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 4, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 19, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 19, 'BLOCK_K': 64}, num_warps=4),
    ],
    key=['M', 'K'],
)
@triton.jit
def fused_linear_reshape_softmax_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, K,
    x_stride_m, x_stride_k,
    w_stride_n, w_stride_k,
    b_stride_n,
    out_stride_0, out_stride_1,
    SOFTMAX_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    N: tl.constexpr = NUM_HEADS * SOFTMAX_SIZE

    pid = tl.program_id(0)
    row_off = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_off < M

    n_idx = tl.arange(0, N)

    # ---- Linear: accumulate x @ w^T + bias ----
    acc = tl.zeros((BLOCK_M, N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_off < K
        x_ptrs = x_ptr + row_off[:, None] * x_stride_m + k_off[None, :] * x_stride_k
        x_vals = tl.load(x_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        w_ptrs = w_ptr + n_idx[:, None] * w_stride_n + k_off[None, :] * w_stride_k
        w_vals = tl.load(w_ptrs, mask=k_mask[None, :], other=0.0)
        acc += tl.dot(x_vals, tl.trans(w_vals), allow_tf32=False)

    if b_ptr is not None:
        b_vals = tl.load(b_ptr + n_idx * b_stride_n)
        acc += b_vals[None, :]

    # ---- Softmax over SOFTMAX_SIZE ----
    head_idx = tl.arange(0, NUM_HEADS)

    max_vals = tl.full((BLOCK_M, NUM_HEADS), float('-inf'), dtype=tl.float32)
    for pos in range(SOFTMAX_SIZE):
        col = head_idx * SOFTMAX_SIZE + pos
        val = acc[:, col]
        max_vals = tl.maximum(max_vals, val)

    sum_vals = tl.zeros((BLOCK_M, NUM_HEADS), dtype=tl.float32)
    exp_cache = []
    for pos in range(SOFTMAX_SIZE):
        col = head_idx * SOFTMAX_SIZE + pos
        val = acc[:, col]
        e = tl.exp(val - max_vals)
        sum_vals += e
        exp_cache.append(e)

    for pos in range(SOFTMAX_SIZE):
        softmax_val = exp_cache[pos] / sum_vals
        out_row_idx = row_off[:, None] * NUM_HEADS + head_idx[None, :]
        out_ptrs = out_ptr + out_row_idx * out_stride_0 + pos * out_stride_1
        tl.store(out_ptrs, softmax_val, mask=row_mask[:, None])


@torch.fx.wrap
def fused_linear_reshape_softmax(bias, weight, x):
    x_2d = x.reshape(-1, x.shape[-1])
    M = x_2d.shape[0]
    K = x_2d.shape[1]
    N = weight.shape[0]

    SOFTMAX_SIZE = 9
    NUM_HEADS = N // SOFTMAX_SIZE

    out = torch.empty((M * NUM_HEADS, SOFTMAX_SIZE, 1), dtype=x.dtype, device=x.device)

    grid = ((M + 19 - 1) // 19,)

    fused_linear_reshape_softmax_kernel[grid](
        x_ptr=x_2d, w_ptr=weight, b_ptr=bias, out_ptr=out,
        M=M, K=K,
        x_stride_m=x_2d.stride(0), x_stride_k=x_2d.stride(1),
        w_stride_n=weight.stride(0), w_stride_k=weight.stride(1),
        b_stride_n=bias.stride(0),
        out_stride_0=out.stride(0), out_stride_1=out.stride(1),
        SOFTMAX_SIZE=SOFTMAX_SIZE,
        NUM_HEADS=NUM_HEADS,
    )

    return out


def replacement_func():
    return fused_linear_reshape_softmax