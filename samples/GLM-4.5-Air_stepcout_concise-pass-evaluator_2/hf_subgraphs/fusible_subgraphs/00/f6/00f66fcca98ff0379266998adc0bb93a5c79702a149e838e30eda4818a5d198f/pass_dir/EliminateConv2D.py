import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor):
    # Match just the conv2d operation to optimize it with a better kernel
    # Original: tmp_2 = torch.conv2d(in_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    conv_out = torch.conv2d(input_tensor, weight_tensor, bias_tensor, (1, 1), (0, 0), (1, 1), 1)
    return conv_out

def replacement_args(input_tensor, weight_tensor, bias_tensor):
    return (input_tensor, weight_tensor, bias_tensor)

@triton.jit
def optimized_conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Calculate ranges for blocks
    m_offset = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    n_offset = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Load bias once per output channel
    bias_vals = tl.load(bias_ptr + n_offset, mask=n_offset < out_channels, other=0.0)
    
    # For 1x1 convolution, this simplifies to a channel-wise linear transformation
    # We'll implement a simpler version that directly computes the output
    if pid_b >= batch_size:
        return
    if pid_m >= height:
        return  
    if pid_n >= (out_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N:
        return
    
    # Process this block
    start_channel = pid_n * BLOCK_SIZE_N
    end_channel = min(start_channel + BLOCK_SIZE_N, out_channels)
    
    # For each channel in this block
    for c_out in range(start_channel, end_channel):
        # Compute output for this channel across all spatial locations
        for h in range(pid_m * BLOCK_SIZE_M, min((pid_m + 1) * BLOCK_SIZE_M, height)):
            for w in range(width):
                # Accumulate result for this spatial position and channel
                result = bias_vals[c_out - start_channel]
                
                # Sum over input channels weighted by the convolution weights
                for c_in in range(in_channels):
                    # Load input value
                    input_idx = pid_b * in_channels * height * width + c_in * height * width + h * width + w
                    input_val = tl.load(input_ptr + input_idx)
                    
                    # Load weight value  
                    weight_idx = c_out * in_channels + c_in
                    weight_val = tl.load(weight_ptr + weight_idx)
                    
                    # Apply weight and accumulate
                    result += input_val * weight_val
                
                # Store result
                output_idx = pid_b * out_channels * height * width + c_out * height * width + h * width + w
                tl.store(output_ptr + output_idx, result)

@torch.fx.wrap
def optimized_conv2d_triton(input_tensor, weight_tensor, bias_tensor):
    # Get input dimensions  
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = bias_tensor.shape[0]
    
    # Tune block sizes for optimal performance
    BLOCK_SIZE_M = 16  # Spatial tiling
    BLOCK_SIZE_N = 256  # Channel tiling
    
    # Calculate grid dimensions
    grid_m = (height + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (out_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_size = (grid_m, grid_n, batch_size)
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, height, width), 
                        dtype=input_tensor.dtype, 
                        device=input_tensor.device)
    
    # Launch optimized kernel
    optimized_conv2d_kernel[grid_size](
        input_ptr=input_tensor,
        weight_ptr=weight_tensor, 
        bias_ptr=bias_tensor,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=height,
        width=width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

def replacement_func():
    return optimized_conv2d_triton