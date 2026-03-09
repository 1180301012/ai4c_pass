import torch
import triton
import triton.language as tl

# Pattern matching function for Conv2D operation - same as test pass but with conv2d
def pattern(x, weight):
    # Try to match conv2d with bias as None (2 arguments)
    conv_output = torch.conv2d(x, weight, None, 
                              stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
    return conv_output

# Argument extraction function
def replacement_args(x, weight):
    return (x, weight)

# Optimized kernel for 1x1 convolution
@triton.jit
def conv2d_1x1_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels, height, width,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    # Program id along M dimension (batch x output_channels)
    pid_m = tl.program_id(0)
    # Program id along N dimension (spatial)
    pid_n = tl.program_id(1)
    
    # Number of programs needed for M dimension
    num_programs_m = tl.cdiv(batch_size * out_channels, BLOCK_SIZE_M)
    pid_m_in_batch = pid_m % num_programs_m
    batch_idx = pid_m_in_batch // (tl.cdiv(out_channels, BLOCK_SIZE_M))
    out_channel_idx = (pid_m_in_batch % (tl.cdiv(out_channels, BLOCK_SIZE_M))) * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    
    # Spatial indices
    h_idx = pid_n % height
    w_idx = pid_n // height
    
    # Bias vector
    bias = tl.load(bias_ptr + out_channel_idx, mask=out_channel_idx < out_channels, other=0.0)
    
    # Initialize output
    acc = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32) + bias[:, None]
    
    # Loop over input channels
    for k in range(0, in_channels, BLOCK_SIZE_K):
        k_block = min(BLOCK_SIZE_K, in_channels - k)
        
        # Load the weight for this K block (different for each output channel)
        weight_offset = (out_channel_idx[:, None] * in_channels + k + tl.arange(0, k_block)[None, :]) * 1 * 1
        weight = tl.load(weight_ptr + weight_offset, 
                        mask=(out_channel_idx[:, None] < out_channels)[:, None] & (k + tl.arange(0, k_block)[None, :] < in_channels),
                        other=0.0)
        
        # Load input for this batch, spatial location and K block
        input_offset = ((batch_idx * in_channels + k) * height + h_idx) * width + w_idx
        input_vals = tl.load(input_ptr + input_offset, 
                           mask=(batch_idx < batch_size) & (k + tl.arange(0, k_block) < in_channels),
                           other=0.0)
        
        # Matrix multiplication: acc += input @ weight.T
        acc += tl.sum(input_vals[:, None] * weight, axis=1)
    
    # Store output
    output_offset = ((batch_idx * out_channels + out_channel_idx) * height + h_idx) * width + w_idx
    tl.store(output_ptr + output_offset, acc, 
             mask=(batch_idx < batch_size) & (out_channel_idx < out_channels) & (h_idx < height) & (w_idx < width))

@torch.fx.wrap
def optimized_conv2d_1x1(x, weight):
    # Get tensor dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels = weight.shape[0]
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, height, width), dtype=x.dtype, device=x.device)
    
    # Create bias tensor (set to zeros since we're matching None bias in pattern)
    bias = torch.zeros(out_channels, dtype=x.dtype, device=x.device)
    
    # Optimize block sizes based on tensor sizes
    if batch_size * out_channels >= 1024:
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 8
        BLOCK_SIZE_K = 32
    else:
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 16  
        BLOCK_SIZE_K = 128
    
    # Calculate grid dimensions
    num_programs_m = (batch_size * out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    num_programs_n = height * width
    grid = (num_programs_m, num_programs_n)
    
    # Launch kernel
    conv2d_1x1_kernel[grid](
        x, weight, bias, output,
        batch_size, in_channels, out_channels, height, width,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return output

def replacement_func():
    return optimized_conv2d_1x1