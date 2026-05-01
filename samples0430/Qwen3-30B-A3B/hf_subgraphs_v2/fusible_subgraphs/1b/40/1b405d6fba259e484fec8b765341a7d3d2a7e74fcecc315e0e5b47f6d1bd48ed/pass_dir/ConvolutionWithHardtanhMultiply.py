import torch
import triton
import triton.language as tl

# Pattern matching function (exact match to model.py structure)
def pattern(in_2, in_1, in_0, in_3):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    return tmp_3 * conv2d

# Argument extraction

def replacement_args(in_2, in_1, in_0, in_3):
    return (in_2, in_1, in_0, in_3)

# Triton kernel
@triton.jit
def conv2d_hm_kernel(
    input_ptr, weight_ptr, bias_ptr, in3_ptr, output_ptr,
    batch, in_channels, out_channels, H, W,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    # Grid dimensions: batch, out_channels, H, W
    batch_idx = tl.program_id(0)
    out_channel_idx = tl.program_id(1)
    h_idx = tl.program_id(2)
    w_idx = tl.program_id(3)

    # Calculate output element offset
    output_offset = batch_idx * out_channels * H * W + \
                   out_channel_idx * H * W + \
                   h_idx * W + w_idx

    # Load bias
    bias = tl.load(bias_ptr + out_channel_idx)
    
    # Convolution sum (input channels reduction)
    acc = tl.zeros((1,), dtype=tl.float32)
    for k in range(0, in_channels, BLOCK_K):
        k_end = min(k + BLOCK_K, in_channels)
        
        # Load input channel data
        input_offset = batch_idx * in_channels * H * W + \
                      k * H * W + h_idx * W + w_idx
        input_data = tl.load(input_ptr + input_offset)
        
        # Load weight data
        weight_offset = out_channel_idx * in_channels * 1 * 1 + k
        weight_data = tl.load(weight_ptr + weight_offset)
        
        acc += input_data * weight_data

    # Apply convolution + bias
    conv_result = acc + bias

    # Load and clamp in3 value
    in3_offset = batch_idx * out_channels * H * W + \
                out_channel_idx * H * W + \
                h_idx * W + w_idx
    in3_val = tl.load(in3_ptr + in3_offset)
    clamped_in3 = tl.clamp(in3_val, 0.0, 6.0)

    # Final multiplication
    result = conv_result * clamped_in3
    tl.store(output_ptr + output_offset, result)

# Kernel wrapper
@torch.fx.wrap
def conv2d_hm(in_2, in_1, in_0, in_3):
    batch, in_channels, H, W = in_2.shape
    out_channels = in_1.shape[0]
    
    # Block configuration
    BLOCK_M = 32  # Output channels block size
    BLOCK_N = 32  # Spatial block size (H, W)
    BLOCK_K = 8   # Input channels block size
    
    # Calculate grid dimensions
    grid_m = (out_channels + BLOCK_M - 1) // BLOCK_M
    grid_n = (H + BLOCK_N - 1) // BLOCK_N
    grid_p = (W + BLOCK_N - 1) // BLOCK_N
    
    # Output tensor
    output = torch.empty_like(in_2, dtype=in_2.dtype)
    
    # Launch kernel
    conv2d_hm_kernel[
        (batch, grid_m, grid_n, grid_p),
    ](
        in_2, in_1, in_0, in_3,
        batch, in_channels, out_channels, H, W,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return conv2d_hm