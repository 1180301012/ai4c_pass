import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 64}, num_warps=8, num_stages=2),
    ],
    key=['H'],
)
@triton.jit
def _swin_bias_softmax_kernel(
    linear_ptr,
    index_ptr,
    scores_ptr,
    mask_ptr,
    out_ptr,
    H,
    S,
    N_ROWS,
    stride_linear_tok,
    stride_linear_h,
    stride_index_m,
    stride_index_n,
    stride_scores_b,
    stride_scores_h,
    stride_scores_m,
    stride_scores_n,
    stride_mask_b,
    stride_mask_m,
    stride_mask_n,
    stride_out_b,
    stride_out_h,
    stride_out_m,
    stride_out_n,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    row = pid // H
    h = pid % H

    offs_n = tl.arange(0, BLOCK_N)

    b = row // S
    m = row % S

    idx_ptrs = index_ptr + m * stride_index_m + offs_n * stride_index_n
    idx = tl.load(idx_ptrs, mask=offs_n < S, other=0).to(tl.int32)

    gathered_ptrs = linear_ptr + idx * stride_linear_tok + h * stride_linear_h
    gathered = tl.load(gathered_ptrs, mask=offs_n < S, other=0.0)
    bias = tl.sigmoid(gathered) * 16.0

    score_ptrs = scores_ptr + b * stride_scores_b + h * stride_scores_h + m * stride_scores_m + offs_n * stride_scores_n
    mask_ptrs = mask_ptr + b * stride_mask_b + m * stride_mask_m + offs_n * stride_mask_n

    vals = tl.load(score_ptrs, mask=offs_n < S, other=-float('inf'))
    mask_vals = tl.load(mask_ptrs, mask=offs_n < S, other=0.0)
    vals = vals + bias + mask_vals + mask_vals

    vals_fp32 = vals.to(tl.float32)
    vals_masked = tl.where(offs_n < S, vals_fp32, -float('inf'))
    max_val = tl.max(vals_masked, axis=0)
    exp_vals = tl.exp(vals_masked - max_val)
    denom = tl.sum(tl.where(offs_n < S, exp_vals, 0.0), axis=0)
    out = exp_vals / denom

    out_ptrs = out_ptr + b * stride_out_b + h * stride_out_h + m * stride_out_m + offs_n * stride_out_n
    tl.store(out_ptrs, out.to(vals.dtype), mask=offs_n < S)


@torch.fx.wrap
def swin_bias_softmax_dispatch(in_0, in_1, in_2, in_3, in_4, route):
    H = 12 if route == 'h12' else 24
    S = 64

    linear = in_4 @ in_1.t()
    linear = linear.reshape(-1, H)

    out = torch.empty_like(in_2)
    n_rows = in_2.shape[0] * S
    grid = (n_rows * H,)

    index = in_0.view(S, S)

    _swin_bias_softmax_kernel[grid](
        linear,
        index,
        in_2,
        in_3,
        out,
        H,
        S,
        n_rows,
        linear.stride(0),
        linear.stride(1),
        index.stride(0),
        index.stride(1),
        in_2.stride(0),
        in_2.stride(1),
        in_2.stride(2),
        in_2.stride(3),
        in_3.stride(0),
        in_3.stride(1),
        in_3.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
    )
    return (out,)