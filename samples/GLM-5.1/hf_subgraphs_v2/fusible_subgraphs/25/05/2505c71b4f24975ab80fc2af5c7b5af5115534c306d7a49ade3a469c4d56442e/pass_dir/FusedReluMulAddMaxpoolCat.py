import torch
import triton
import triton.language as tl
import math


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    tmp_5 = torch.nn.functional.max_pool2d(in_3, 2, 1, 0, 1, ceil_mode=True, return_indices=False)
    tmp_6 = torch.cat([tmp_5, tmp_4], dim=1)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def maxpool_branch_kernel(
    in_3_ptr, out_ptr,
    N, C_in, H_in3, W_in3, H_out, W_out,
    stride_n3, stride_c3, stride_h3, stride_w3,
    stride_n_out, stride_c_out, stride_h_out, stride_w_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    total = N * C_in * H_out * W_out
    mask = offsets < total

    # Decode (n, c, h_out, w_out) from flat offset
    n = offsets // (C_in * H_out * W_out)
    c = (offsets // (H_out * W_out)) % C_in
    h_out = (offsets // W_out) % H_out
    w_out = offsets % W_out

    # Max pool 2x2 with stride=1, padding=0, dilation=1, ceil_mode=True
    # Window: (h_out, w_out), (h_out, w_out+1), (h_out+1, w_out), (h_out+1, w_out+1)
    base = n * stride_n3 + c * stride_c3

    valid00 = mask & (h_out < H_in3) & (w_out < W_in3)
    valid01 = mask & (h_out < H_in3) & (w_out + 1 < W_in3)
    valid10 = mask & (h_out + 1 < H_in3) & (w_out < W_in3)
    valid11 = mask & (h_out + 1 < H_in3) & (w_out + 1 < W_in3)

    NEG_INF = float('-inf')
    v00 = tl.load(in_3_ptr + base + h_out * stride_h3 + w_out * stride_w3, mask=valid00, other=NEG_INF)
    v01 = tl.load(in_3_ptr + base + h_out * stride_h3 + (w_out + 1) * stride_w3, mask=valid01, other=NEG_INF)
    v10 = tl.load(in_3_ptr + base + (h_out + 1) * stride_h3 + w_out * stride_w3, mask=valid10, other=NEG_INF)
    v11 = tl.load(in_3_ptr + base + (h_out + 1) * stride_h3 + (w_out + 1) * stride_w3, mask=valid11, other=NEG_INF)

    pool_val = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))

    # Store to output (channels 0 to C_in-1)
    out_offset = n * stride_n_out + c * stride_c_out + h_out * stride_h_out + w_out * stride_w_out
    tl.store(out_ptr + out_offset, pool_val, mask=mask)


@triton.jit
def affine_branch_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, out_ptr,
    N, C_in, H_out, W_out,
    stride_n2, stride_c2, stride_h2, stride_w2,
    stride_n_out, stride_c_out, stride_h_out, stride_w_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    total = N * C_in * H_out * W_out
    mask = offsets < total

    # Decode (n, c, h_out, w_out) from flat offset
    n = offsets // (C_in * H_out * W_out)
    c = (offsets // (H_out * W_out)) % C_in
    h_out = (offsets // W_out) % H_out
    w_out = offsets % W_out

    # Load scalar bias and scale
    bias = tl.load(in_0_ptr)
    scale = tl.load(in_1_ptr)

    # Load from in_2 and compute relu * scale + bias
    in_2_offset = n * stride_n2 + c * stride_c2 + h_out * stride_h2 + w_out * stride_w2
    v = tl.load(in_2_ptr + in_2_offset, mask=mask, other=0.0)
    result = tl.maximum(v, 0.0) * scale + bias

    # Store to output (channels C_in to 2*C_in-1)
    out_offset = n * stride_n_out + (c + C_in) * stride_c_out + h_out * stride_h_out + w_out * stride_w_out
    tl.store(out_ptr + out_offset, result, mask=mask)


@torch.fx.wrap
def fused_relu_scale_bias_maxpool_cat(in_0, in_1, in_2, in_3):
    N = in_2.shape[0]
    C_in = in_2.shape[1]
    H_in2 = in_2.shape[2]
    W_in2 = in_2.shape[3]
    H_in3 = in_3.shape[2]
    W_in3 = in_3.shape[3]

    # Compute output spatial dimensions
    # max_pool2d with kernel_size=2, stride=1, padding=0, dilation=1, ceil_mode=True
    # Output size: ceil((H_in - 2) / 1 + 1) = H_in - 1
    H_out = H_in3 - 1
    W_out = W_in3 - 1

    # Allocate output: [N, 2*C_in, H_out, W_out]
    out = torch.empty((N, 2 * C_in, H_out, W_out), dtype=in_2.dtype, device=in_2.device)

    pool_total = N * C_in * H_out * W_out
    BLOCK_SIZE = 1024
    grid_size = (pool_total + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch max_pool kernel (writes to channels 0:C_in)
    maxpool_branch_kernel[grid_size](
        in_3, out,
        N, C_in, H_in3, W_in3, H_out, W_out,
        in_3.stride(0), in_3.stride(1), in_3.stride(2), in_3.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Launch affine kernel (writes to channels C_in:2*C_in)
    affine_branch_kernel[grid_size](
        in_0, in_1, in_2, out,
        N, C_in, H_out, W_out,
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return fused_relu_scale_bias_maxpool_cat