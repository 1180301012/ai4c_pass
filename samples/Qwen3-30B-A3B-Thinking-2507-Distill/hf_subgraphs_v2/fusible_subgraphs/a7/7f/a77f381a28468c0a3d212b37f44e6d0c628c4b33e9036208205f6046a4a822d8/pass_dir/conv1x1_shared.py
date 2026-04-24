"""
Shared Triton kernel for 1x1 convolution (as GEMM) + slice.
1x1 conv is equivalent to: input_reshaped @ weight.T
  input  : [B, C_in,  H, W]
  weight : [C_out, C_in, 1, 1]  (squeezed to [C_out, C_in])
  output : [B, C_out, H_out, W_out]
where H_out = H (stride=1) or H//2 (stride=2), same for W.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _conv1x1_gemm_kernel(
    input_ptr,   # [B, C_in, H_in, W_in]
    weight_ptr,  # [C_out, C_in]  (1x1 kernel squeezed)
    output_ptr,  # [B, C_out, H_out, W_out]
    B, C_in, H_in, W_in,
    C_out, H_out, W_out,
    stride_h, stride_w,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes: output[b, c_out, h_out, w_out] =
      sum_k input[b, k, h_out*stride_h, w_out*stride_w] * weight[c_out, k]
    Grid: (ceil(M/BLOCK_M), ceil(N/BLOCK_N))
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decompose m -> (b, h_out, w_out)
    hw_out = H_out * W_out
    b_idx     = offs_m // hw_out
    hw_flat   = offs_m  % hw_out
    h_out_idx = hw_flat // W_out
    w_out_idx = hw_flat  % W_out

    # Accumulator (mixed precision)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load input tile [BLOCK_M, BLOCK_K]
        # input[b, k, h_out*stride_h, w_out*stride_w]
        h_in = h_out_idx[:, None] * stride_h  # [BLOCK_M, 1]
        w_in = w_out_idx[:, None] * W_out     # [BLOCK_M, 1]  (reuse W_out for stride_w=2)
        in_idx = (b_idx[:, None] * (C_in * H_in * W_in) +
                  offs_k[None, :] * (H_in * W_in) +
                  h_in * W_in +
                  w_in)
        in_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(input_ptr + in_idx, mask=in_mask, other=0.0)

        # Load weight tile [BLOCK_N, BLOCK_K]
        # weight[c_out, k] = weight_ptr + c_out * K + k
        w_idx  = offs_n[:, None] * K + offs_k[None, :]
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        b_tile = tl.load(weight_ptr + w_idx, mask=w_mask, other=0.0)

        acc += tl.dot(a.to(tl.float16), tl.trans(b_tile.to(tl.float16)))

    # Store output [BLOCK_M, BLOCK_N]
    out_idx  = (b_idx[:, None] * (C_out * H_out * W_out) +
                offs_n[None, :] * (H_out * W_out) +
                h_out_idx[:, None] * W_out +
                w_out_idx[:, None])
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptr + out_idx, acc.to(output_ptr.dtype.element_ty), mask=out_mask)


def conv1x1_triton(in_0, in_1, stride_h, stride_w, H_out, W_out):
    """
    Computes 1x1 conv of in_1 using weights in_0.
    in_0 : [C_out, C_in, 1, 1]  (weight)
    in_1 : [B,    C_in, H_in, W_in]  (input)
    stride_h, stride_w : stride in H and W directions (1 or 2)
    H_out, W_out : output spatial dimensions
    """
    B, C_in, H_in, W_in = in_1.shape
    C_out = in_0.shape[0]

    M = B * H_out * W_out
    N = C_out
    K = C_in

    output = torch.empty((B, C_out, H_out, W_out),
                         dtype=in_1.dtype, device=in_1.device)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    _conv1x1_gemm_kernel[grid](
        in_1, in_0, output,
        B, C_in, H_in, W_in,
        C_out, H_out, W_out,
        stride_h, stride_w,
        M, N, K,
    )

    return output