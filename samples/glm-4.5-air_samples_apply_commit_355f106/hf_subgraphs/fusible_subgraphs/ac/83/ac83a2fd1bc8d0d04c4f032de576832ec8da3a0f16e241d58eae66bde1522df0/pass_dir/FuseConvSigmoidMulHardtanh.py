import torch
import triton
import triton.language as tl
import math

# Pattern matching function
def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.conv2d(in_3, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = tmp_2.sigmoid()
    tmp_4 = in_2 * tmp_3
    tmp_5 = torch.nn.functional.hardtanh(tmp_4, 0.0, 6.0, False)
    return tmp_5

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Optimized fused kernel with manual tune instead of autotune
# Using manually tuned parameters for better performance
@triton.jit
def fused_conv_sigmoid_mul_hardtanh_kernel(
    bias_ptr,  # [output_channels]
    weight_ptr,  # [output_channels, input_channels, 1, 1]
    x2_ptr,  # [batch_size, channels, height, width]
    x3_ptr,  # [batch_size, input_channels, 1, 1]
    out_ptr,  # [batch_size, channels, height, width]
    batch_size,
    channels,
    height,
    width,
    output_channels,
    input_channels,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Program ID for parallel execution  
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute program bounds with padding to avoid divergence
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    
    # Use vectorized loads for better performance
    mask_m = m_offset < batch_size
    mask_n = n_offset < output_channels
    if not (mask_m and mask_n):
        return
    
    # Calculate offsets with vectorization
    batch_idx = m_offset
    channel_idx = n_offset
    
    # Load bias with vectorization opportunities
    bias = tl.load(bias_ptr + channel_idx, mask=channel_idx < output_channels, other=0.0)
    
    # Optimized convolution with reduced memory accesses
    conv_sum = bias
    for ic in tl.range(0, input_channels):
        weight_offset = channel_idx * input_channels + ic
        weight_val = tl.load(weight_ptr + weight_offset, mask=(channel_idx * input_channels + ic) < (output_channels * input_channels), other=0.0)
        
        x3_offset = batch_idx * input_channels + ic
        x3_val = tl.load(x3_ptr + x3_offset, mask=batch_idx < batch_size, other=0.0)
        
        # Use fused multiply-add
        conv_sum += x3_val * weight_val
    
    # Optimized sigmoid approximation using fast exp
    sigmoid_val = tl.sigmoid(conv_sum)
    
    # Process spatial locations with loop unrolling benefits
    for h in range(height):
        for w in range(width):
            # Calculate linearized position
            x2_offset = (batch_idx * channels + channel_idx) * height * width + h * width + w
            
            # Vectorized load and compute
            x2_val = tl.load(x2_ptr + x2_offset, mask=x2_offset < (batch_size * channels * height * width), other=0.0)
            
            # Fused element-wise operations
            result = tl.clamp(x2_val * sigmoid_val, 0.0, 6.0)
            
            # Store with proper masking
            tl.store(out_ptr + x2_offset, result, mask=x2_offset < (batch_size * channels * height * width))

# Kernel wrapper
@torch.fx.wrap
def fused_conv_sigmoid_mul_hardtanh(in_0, in_1, in_2, in_3):
    # Get shapes
    batch_size, channels, height, width = in_2.shape
    input_channels = in_3.shape[1]
    output_channels = in_1.shape[0]  # weight is [output_channels, input_channels, 1, 1]
    
    # Move weights to CUDA
    weight = in_1.cuda()
    bias = in_0.cuda()
    
    # Create output tensor
    out = torch.empty_like(in_2)
    
    # Use optimized block sizes based on tensor shapes
    BLOCK_SIZE_M = 8   # Process batch dimension in blocks of 8
    BLOCK_SIZE_N = 32  # Process output channels in blocks of 32
    
    # Calculate grid size
    grid_m = (batch_size + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (output_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    # Launch kernel with required block size arguments
    fused_conv_sigmoid_mul_hardtanh_kernel[(grid_m, grid_n)](
        bias_ptr=bias,
        weight_ptr=weight,
        x2_ptr=in_2,
        x3_ptr=in_3,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        output_channels=output_channels,
        input_channels=input_channels,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

# Replacement function
def replacement_func():
    return fused_conv_sigmoid_mul_hardtanh