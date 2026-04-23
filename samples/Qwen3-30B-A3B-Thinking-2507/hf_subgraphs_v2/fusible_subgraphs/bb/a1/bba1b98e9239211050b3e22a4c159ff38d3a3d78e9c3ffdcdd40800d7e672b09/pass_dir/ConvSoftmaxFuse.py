import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_2, in_1, in_0):
    conv = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    view = conv.view(conv.size(0), 1, -1)
    softmax = view.softmax(dim=-1)
    return softmax

# Argument extraction function
def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

# Triton kernel for fused convolution and softmax
def conv_softmax_fused_kernel(
    in_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    batch_size,
    in_channels,
    height,
    width,
    block_size_h,
    block_size_w
):
    # Each block processes one batch and the entire spatial dimension
    batch_idx = tl.program_id(0)

    # Thread indices for spatial dimensions
    h = tl.thread_id(0)
    w = tl.thread_id(1)
    # Ensure we stay within bounds
    if h >= height or w >= width:
        return

    # Load input tensor for this (h, w) position
    # input: [batch, channels, height, width]
    input_offset = batch_idx * (in_channels * height * width) + h * width + w
    # Convolution calculation: sum_c (input[batch, c, h, w] * weight[c]) + bias
    acc = tl.load(bias_ptr, cache=tl.L2)
    for c in range(in_channels):
        input_val = tl.load(in_ptr + input_offset + c * height * width, cache=tl.L2)
        weight_val = tl.load(weight_ptr + c, cache=tl.L2)
        acc += input_val * weight_val

    # Store result in out tensor (for spatial position)
    out_offset = batch_idx * (1 * height * width) + h * width + w
    tl.store(out_ptr + out_offset, acc, cache=tl.L2)

    # Synchronize to ensure all spatial positions are computed
    tl.sync.warp()

    # Now compute softmax over spatial dimension
    # Find max value for numerical stability
    max_val = -1e30
    for y in range(height):
        for x in range(width):
            val = tl.load(out_ptr + batch_idx * (1 * height * width) + y * width + x, cache=tl.L2)
            max_val = tl.max(max_val, val)
    # Compute exp and sum
    exp_sum = 0.0
    for y in range(height):
        for x in range(width):
            val = tl.load(out_ptr + batch_idx * (1 * height * width) + y * width + x, cache=tl.L2)
            exp_val = tl.exp(val - max_val)
            exp_sum += exp_val
            tl.store(out_ptr + batch_idx * (1 * height * width) + y * width + x, exp_val, cache=tl.L2)
    # Normalize
    for y in range(height):
        for x in range(width):
            exp_val = tl.load(out_ptr + batch_idx * (1 * height * width) + y * width + x, cache=tl.L2)
            normalized = exp_val / exp_sum
            tl.store(out_ptr + batch_idx * (1 * height * width) + y * width + x, normalized, cache=tl.L2)

# Kernel wrapper
@torch.fx.wrap
def conv_softmax_wrapper(in_2, in_1, in_0):
    # Extract shapes
    batch_size = in_2.shape[0]
    height = in_2.shape[2]
    width = in_2.shape[3]
    in_channels = in_2.shape[1]
    
    # Weight shape: [1, 512, 1, 1] → reshape to [512]
    weight = in_1[0, :, 0, 0]
    bias = in_0[0]
    
    # Allocate output
    out = torch.empty((batch_size, 1, height, width), dtype=in_2.dtype, device=in_2.device)
    
    # Launch kernel
    grid = (batch_size, 1)
    conv_softmax_fused_kernel[grid](
        in_2, weight, bias, out,
        batch_size,
        in_channels,
        height,
        width,
        64, 64  # Block size
    )
    
    # Final reshape to match original: [batch, 1, -1]
    return out.view(batch_size, 1, -1)

# Replacement function
def replacement_func():
    return conv_softmax_wrapper