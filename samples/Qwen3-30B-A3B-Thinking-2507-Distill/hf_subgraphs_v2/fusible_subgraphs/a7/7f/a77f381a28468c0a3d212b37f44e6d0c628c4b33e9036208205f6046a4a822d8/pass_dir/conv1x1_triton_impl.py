"""
Shared Triton kernel: 1x1 convolution as GEMM.
Input: [B, C_in, H_in, W_in]
Weight: [C_out, C_in, 1, 1]  ->  treated as [C_out, C_in]
Output: [B, C_out, H_out, W_out]
where H_out = H_in (stride=1) or H_in//2 (stride=2)
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Large blocks for large M (B=128, high arithmetic intensity)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        # Medium blocks
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
        # Small blocks for small M (B=1) — higher thread block count
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _conv1x1_gemm_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    B, C_in, H_in, W_in,
    C_out, H_out, W_out,
    stride_h, stride_w,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    output[b, c_out, h_out, w_out] =
        sum_k  input[b, k, h_out*stride_h, w_out*stride_w] * weight[c_out, k]
    Grid: (ceil(M/BLOCK_M), ceil(N/BLOCK_N))
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decompose m -> (b, h_out, w_out)
    hw_out    = H_out * W_out
    b_idx     = offs_m // hw_out
    hw_flat   = offs_m  % hw_out
    h_out_idx = hw_flat // W_out
    w_out_idx = hw_flat  % W_out

    # Accumulator in float32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Input tile [BLOCK_M, BLOCK_K]
        # input[b, k, h_out*stride_h, w_out*stride_w]
        h_in = h_out_idx[:, None] * stride_h   # [BLOCK_M, 1]
        w_in = w_out_idx[:, None] * stride_w   # [BLOCK_M, 1]
        in_idx  = (b_idx[:, None] * (C_in * H_in * W_in) +
                   offs_k[None, :] * (H_in * W_in) +
                   h_in * W_in +
                   w_in)
        in_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(input_ptr + in_idx, mask=in_mask, other=0.0)

        # Weight tile [BLOCK_N, BLOCK_K]
        w_idx  = offs_n[:, None] * K + offs_k[None, :]
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        b_tile = tl.load(weight_ptr + w_idx, mask=w_mask, other=0.0)

        # Accumulate in float32; handles fp16/bf16 inputs natively on Ampere
        acc += tl.dot(a, tl.trans(b_tile), out_dtype=tl.float32)

    # Store output
    out_idx  = (b_idx[:, None] * (C_out * H_out * W_out) +
                offs_n[None, :] * (H_out * W_out) +
                h_out_idx[:, None] * W_out +
                w_out_idx[:, None])
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptr + out_idx, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


def conv1x1_triton(weight, inp, stride_h, stride_w):
    """
    1x1 convolution using Triton GEMM.
    weight : [C_out, C_in, 1, 1]
    inp    : [B, C_in, H_in, W_in]
    returns: [B, C_out, H_out, W_out]
    """
    B, C_in, H_in, W_in = inp.shape
    C_out = weight.shape[0]

    H_out = H_in // stride_h
    W_out = W_in // stride_w

    M = B * H_out * W_out
    N = C_out
    K = C_in

    output = torch.empty((B, C_out, H_out, W_out),
                         dtype=inp.dtype, device=inp.device)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    _conv1x1_gemm_kernel[grid](
        inp, weight, output,
        B, C_in, H_in, W_in,
        C_out, H_out, W_out,
        stride_h, stride_w,
        M, N, K,
    )
    return output


@torch.fx.wrap
def triton_conv1x1_dispatch(in_0, in_1, route):
    """
    Shared dispatch wrapper for all 1x1-conv passes.
    in_0: weight [C_out, C_in, 1, 1]
    in_1: input  [B, C_in, H_in, W_in]
    route: string selecting stride configuration
    """
    B, C_in, H_in, W_in = in_1.shape
    if route == "stride2":
        H_out = H_in // 2
        W_out = W_in // 2
        output = conv1x1_triton(in_0, in_1, 2, 2)
    else:
        # stride=1 routes: spatial dims stay the same
        output = conv1x1_triton(in_0, in_1, 1, 1)
    return output