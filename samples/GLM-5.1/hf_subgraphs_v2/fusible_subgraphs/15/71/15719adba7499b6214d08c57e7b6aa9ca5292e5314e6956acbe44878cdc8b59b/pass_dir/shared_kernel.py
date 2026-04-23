import torch
import triton
import triton.language as tl
from graph_net_bench.torch.posion_dispatch_tensor import unwrap_tensor


@triton.jit
def fused_ln_kernel(
    conv_out_ptr,
    norm_weight_ptr,
    norm_bias_ptr,
    out1_ptr,
    N,  # Total spatial positions (H * W)
    C,  # Channel dimension
    H,  # Height of conv2d output
    W,  # Width of conv2d output
    eps,  # Layer norm epsilon
    conv_stride_c,  # Stride for channel dim in conv2d output
    conv_stride_h,  # Stride for height dim in conv2d output
    conv_stride_w,  # Stride for width dim in conv2d output
    GROUP_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles GROUP_SIZE spatial positions
    pid = tl.program_id(0)
    pos_start = pid * GROUP_SIZE
    pos_offsets = pos_start + tl.arange(0, GROUP_SIZE)
    pos_mask = pos_offsets < N

    # Decompose spatial position into (h, w)
    h_idx = pos_offsets // W
    w_idx = pos_offsets % W

    # First pass: compute sum and sum of squares for mean/variance
    sum_x = tl.zeros([GROUP_SIZE], dtype=tl.float32)
    sum_x2 = tl.zeros([GROUP_SIZE], dtype=tl.float32)

    c_block_start = 0
    while c_block_start < C:
        c_offsets = c_block_start + tl.arange(0, BLOCK_SIZE)
        c_mask = c_offsets < C

        # Load channels for this block of spatial positions
        # conv2d output layout: [1, C, H, W]
        # Element at (c, h, w): c * conv_stride_c + h * conv_stride_h + w * conv_stride_w
        conv_offsets = c_offsets[None, :] * conv_stride_c + h_idx[:, None] * conv_stride_h + w_idx[:, None] * conv_stride_w
        x_block = tl.load(conv_out_ptr + conv_offsets, mask=pos_mask[:, None] & c_mask[None, :], other=0.0)
        x_block_f32 = x_block.to(tl.float32)

        # Accumulate
        sum_x += tl.sum(x_block_f32, axis=1)
        sum_x2 += tl.sum(x_block_f32 * x_block_f32, axis=1)

        c_block_start += BLOCK_SIZE

    # Compute mean and variance
    mean = sum_x / C
    variance = sum_x2 / C - mean * mean

    # Compute reciprocal of sqrt(variance + eps)
    rstd = 1.0 / tl.sqrt(variance + eps)

    # Second pass: normalize, apply weight/bias, and store to output
    c_block_start = 0
    while c_block_start < C:
        c_offsets = c_block_start + tl.arange(0, BLOCK_SIZE)
        c_mask = c_offsets < C

        # Load channels again
        conv_offsets = c_offsets[None, :] * conv_stride_c + h_idx[:, None] * conv_stride_h + w_idx[:, None] * conv_stride_w
        x_block = tl.load(conv_out_ptr + conv_offsets, mask=pos_mask[:, None] & c_mask[None, :], other=0.0)
        x_block_f32 = x_block.to(tl.float32)

        # Normalize
        x_norm = (x_block_f32 - mean[:, None]) * rstd[:, None]

        # Load weight and bias
        w_val = tl.load(norm_weight_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)
        b_val = tl.load(norm_bias_ptr + c_offsets, mask=c_mask, other=0.0).to(tl.float32)

        # Apply weight and bias
        result = x_norm * w_val[None, :] + b_val[None, :]

        # Store output1: [1, N, C] contiguous layout
        # Element at (pos, c): pos * C + c
        out1_offsets = pos_offsets[:, None] * C + c_offsets[None, :]
        tl.store(out1_ptr + out1_offsets, result, mask=pos_mask[:, None] & c_mask[None, :])

        c_block_start += BLOCK_SIZE


@torch.fx.wrap
def fused_ln_dispatch(*args):
    # args: (conv_out, norm_weight, norm_bias, route_string)
    conv_out_raw = args[0]
    norm_weight_raw = args[1]
    norm_bias_raw = args[2]
    route = args[3]

    # Unwrap PosionDispatchTensor if needed
    conv_out = unwrap_tensor(conv_out_raw)
    norm_weight = unwrap_tensor(norm_weight_raw)
    norm_bias = unwrap_tensor(norm_bias_raw)

    # Get metadata from conv2d output
    C = conv_out.size(1)
    H = conv_out.size(2)
    W = conv_out.size(3)
    N = H * W
    dtype = conv_out.dtype
    device = conv_out.device

    eps = 1e-05

    # Get strides for conv2d output [1, C, H, W]
    conv_stride_c = conv_out.stride(1)
    conv_stride_h = conv_out.stride(2)
    conv_stride_w = conv_out.stride(3)

    # Allocate output: [1, N, C] = [1, H*W, C]
    out1 = torch.empty(1, N, C, dtype=dtype, device=device)

    # Choose kernel configuration based on C
    if C <= 32:
        BLOCK_SIZE = 32
        GROUP_SIZE = 32
    elif C <= 64:
        BLOCK_SIZE = 64
        GROUP_SIZE = 64
    else:
        BLOCK_SIZE = 128
        GROUP_SIZE = 128

    # Ensure BLOCK_SIZE >= C
    while BLOCK_SIZE < C:
        BLOCK_SIZE *= 2

    grid = ((N + GROUP_SIZE - 1) // GROUP_SIZE,)

    fused_ln_kernel[grid](
        conv_out, norm_weight, norm_bias, out1,
        N, C, H, W, eps,
        conv_stride_c, conv_stride_h, conv_stride_w,
        GROUP_SIZE=GROUP_SIZE, BLOCK_SIZE=BLOCK_SIZE,
    )

    return out1


def replacement_func():
    return fused_ln_dispatch