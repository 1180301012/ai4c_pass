import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_0):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_2, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    return tmp_2

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

@triton.jit
def conv2d_silu_kernel(
    input_ptr, weight_ptr, bias_ptr, 
    output_ptr,
    N, C_out, C_in, H, W,
    stride_h, stride_w,
    pad_h, pad_w,
    dil_h, dil_w,
    groups,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr
):
    # Get program IDs 
    spatial_idx = tl.program_id(0)  # spatial position (0 to N-1)
    output_ch_block = tl.program_id(1)  # output channel block
    
    # Calculate offsets
    spatial_offset = spatial_idx
    output_ch_offset = output_ch_block * BLOCK_SIZE_M
    
    # Load bias once per output channel (cache in registers for reuse)
    bias_val = tl.load(bias_ptr + output_ch_offset)
    
    # Load input value for this spatial position (reuse for all output channels)
    input_offset = spatial_offset * C_in
    
    # Initialize accumulator with bias for better arithmetic scheduling
    accumulator = bias_val
    
    # Unroll the input channel loop for better performance
    for k in range(0, C_in, BLOCK_SIZE_K):
        # Vectorized load of input values
        input_ptrs = input_ptr + input_offset + k + tl.arange(0, BLOCK_SIZE_K)
        input_vals = tl.load(input_ptrs, mask=input_offset + k + tl.arange(0, BLOCK_SIZE_K) < N * C_in, other=0.0)
        
        # Vectorized load of weights for this output channel
        weight_offset = output_ch_offset * C_in
        weight_ptrs = weight_ptr + weight_offset + k + tl.arange(0, BLOCK_SIZE_K)
        weight_vals = tl.load(weight_ptrs, mask=weight_offset + k + tl.arange(0, BLOCK_SIZE_K) < C_out * C_in, other=0.0)
        
        # Vectorized dot product with better arithmetic ordering
        chunk_sum = tl.sum(input_vals * weight_vals)
        accumulator += chunk_sum
    
    # Apply SiLU activation (bias already accumulated)
    output_val = accumulator * tl.sigmoid(accumulator)
    
    # Store result
    if output_ch_offset < C_out and spatial_offset < N:
        output_ptr_local = output_ptr + spatial_offset * C_out + output_ch_offset
        tl.store(output_ptr_local, output_val)

@torch.fx.wrap
def fused_conv2d_silu(input, weight, bias):
    # Get input dimensions
    batch, in_channels, in_height, in_width = input.shape
    out_channels, in_channels_w, kernel_h, kernel_w = weight.shape
    
    # Output spatial dimensions (same as input for 1x1 conv with stride 1, padding 0)
    out_height = in_height
    out_width = in_width
    
    # Optimize for 1x1 convolution
    assert in_channels == in_channels_w and kernel_h == 1 and kernel_w == 1, "Only optimized for 1x1 conv"
    assert batch == 1, "Only optimized for batch size 1"
    
    # For 1x1 conv, this is essentially a linear operation: output = input @ weight.T + bias
    # We can optimize this by processing it as a matrix multiplication
    
    # Reshape input for efficient matrix multiplication: [1, C_in, H, W] -> [H*W, C_in]
    input_reshaped = input.transpose(1, 2).transpose(2, 3).reshape(in_height * in_width, in_channels)
    
    # Weight is [C_out, C_in, 1, 1] -> [C_out, C_in]
    weight_reshaped = weight.reshape(out_channels, in_channels)
    
    # Create output tensor and reshape for the kernel
    output_spatial = out_height * out_width
    output = torch.empty((output_spatial, out_channels), dtype=input.dtype, device=input.device)
    
    # Fine-tuned block sizes for optimal performance
    BLOCK_SIZE_M = 64   # Output channels per block (better for 256 output channels)
    BLOCK_SIZE_N = 32   # Spatial positions per block (better warp occupancy)  
    BLOCK_SIZE_K = 32   # Input channels per iteration (better vectorization for 128 input channels)
    
    grid = (
        (output_spatial + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N,
        (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M,
    )
    
    # Launch optimized kernel
    conv2d_silu_kernel[grid](
        input_reshaped, weight_reshaped, bias, output,
        output_spatial, out_channels, in_channels, 1, 1,
        1, 1, 0, 0, 1, 1, 1,
        BLOCK_SIZE_M, BLOCK_SIZE_N, 1
    )
    
    # Reshape back to original format [1, C_out, H, W]  
    return output.reshape(out_height, out_width, out_channels).permute(2, 0, 1).unsqueeze(0)

def replacement_func():
    return fused_conv2d_silu