"""
Shared Triton kernel for fused 1x1 conv + cat.

A 1x1 conv on NCHW input [N, C_in, H, W] with weight [C_out, C_in, 1, 1]
and bias [C_out] is equivalent to a batched GEMM:

  For each spatial position m = n*HW + hw:
    out[m, c_out] = sum_{k=0}^{C_in-1} input[n, k, h, w] * weight[c_out, k] + bias[c_out]

In NCHW memory: input offset for (m, k) = n*C_in*HW + k*HW + hw
                                          = (n*C_in + k)*HW + hw
                                          = input_base[m] + k*HW
  where input_base[m] = n*C_in*HW + hw_idx  (with n=m//HW, hw_idx=m%HW)

The output is written directly into the first C_out channels of a pre-allocated
[N, C_total, H, W] tensor, then the cat tensor is copied into channels [C_out:].
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Large tiles — high throughput for large M (GROUP_SIZE_M=16 for better L2 reuse)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 16}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 16}, num_stages=4, num_warps=8),
        # GROUP_SIZE_M=8 configs (previous best)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        # Small M tiles for low-latency
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
    ],
    key=['M', 'C_out', 'C_in'],
)
@triton.jit
def conv1x1_gemm_nchw_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, C_in, C_out, HW, C_total,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    GEMM for 1x1 conv on NCHW input.
    Input [N,C_in,H,W] accessed as [M,C_in] with stride HW in C_in direction.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(C_out, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    n_batch    = m_offsets // HW
    hw_idx     = m_offsets % HW
    input_base = n_batch * C_in * HW + hw_idx

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(C_in, BLOCK_K)):
        k_offsets = k * BLOCK_K + tl.arange(0, BLOCK_K)

        # Input [BLOCK_M, BLOCK_K]: stride HW in K direction
        input_offs = input_base[:, None] + k_offsets[None, :] * HW
        input_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < C_in)
        a = tl.load(input_ptr + input_offs, mask=input_mask, other=0.0)

        # Weight [BLOCK_K, BLOCK_N]: weight[c_out, c_in] at c_out*C_in + c_in
        weight_offs = n_offsets[None, :] * C_in + k_offsets[:, None]
        weight_mask = (k_offsets[:, None] < C_in) & (n_offsets[None, :] < C_out)
        b = tl.load(weight_ptr + weight_offs, mask=weight_mask, other=0.0)

        acc += tl.dot(a, b, out_dtype=tl.float32)

    bias = tl.load(bias_ptr + n_offsets, mask=n_offsets < C_out, other=0.0)
    acc += bias[None, :].to(tl.float32)

    # Write result to first C_out channels of output [N, C_total, H, W]
    out_base = n_batch * C_total * HW + hw_idx
    out_offs = out_base[:, None] + n_offsets[None, :] * HW
    out_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < C_out)
    tl.store(output_ptr + out_offs,
             acc.to(output_ptr.dtype.element_ty),
             mask=out_mask)


@triton.jit
def copy_cat_to_output_kernel(
    src_ptr, dst_ptr,
    N, C_src, HW, C_offset, C_total,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Copies src [N, C_src, H, W] into dst [N, C_total, H, W] at channel offset C_offset.
    Reads src in flat row-major order and remaps to the correct dst location.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C_src * HW
    mask = offsets < total

    # Decompose flat src index: n * C_src * HW + c * HW + hw
    n   = offsets // (C_src * HW)
    rem = offsets % (C_src * HW)
    c   = rem // HW
    hw  = rem % HW

    val = tl.load(src_ptr + offsets, mask=mask)
    dst_offset = n * C_total * HW + (C_offset + c) * HW + hw
    tl.store(dst_ptr + dst_offset, val, mask=mask)


@torch.fx.wrap
def fused_conv1x1_cat(bias, weight, conv_in, cat_in):
    """
    Fused 1x1 conv + cat along channel dimension.

    Args:
        bias:    [C_out]
        weight:  [C_out, C_in, 1, 1]
        conv_in: [N, C_in, H, W]
        cat_in:  [N, C_cat, H, W]

    Returns:
        [N, C_out + C_cat, H, W]
    """
    N, C_in, H, W = conv_in.shape
    C_out   = weight.shape[0]
    C_cat   = cat_in.shape[1]
    C_total = C_out + C_cat
    HW      = H * W
    M       = N * HW

    # Allocate output tensor for the concatenated result
    output = torch.empty((N, C_total, H, W), dtype=conv_in.dtype, device=conv_in.device)

    # --- Stage 1: 1x1 conv GEMM → first C_out channels of output ---
    def _gemm_grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(C_out, meta['BLOCK_N']),)

    conv1x1_gemm_nchw_kernel[_gemm_grid](
        conv_in, weight, bias, output,
        M, C_in, C_out, HW, C_total,
    )

    # --- Stage 2: copy cat_in → channels [C_out : C_total] of output ---
    total_cat  = N * C_cat * HW
    BLOCK_SIZE = 1024
    num_blocks = triton.cdiv(total_cat, BLOCK_SIZE)
    copy_cat_to_output_kernel[(num_blocks,)](
        cat_in, output,
        N, C_cat, HW, C_out, C_total,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output