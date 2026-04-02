import torch
import triton
import triton.language as tl

def pattern(conv_input, weight, bias, multiplier):
    """
    Pattern matches: conv2d + 1.0 -> / 2.0 -> clamp(0.0, 1.0) -> * multiplier
    
    This matches the computation pattern found in all the target graphs:
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d + 1.0
    tmp_4 = tmp_3 / 2.0
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    tmp_6 = in_2 * tmp_5
    """
    conv_result = torch.conv2d(conv_input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv_result + 1.0
    tmp_4 = tmp_3 / 2.0
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    result = multiplier * tmp_5
    return result

def replacement_args(conv_input, weight, bias, multiplier):
    """Extract arguments needed for the optimized kernel"""
    return (conv_input, weight, bias, multiplier)

@triton.jit
def fused_conv_activation_kernel(
    # Input/output pointers
    conv_input_ptr, 
    weight_ptr, 
    bias_ptr,
    multiplier_ptr,
    output_ptr,
    
    # Convolution metadata
    batch_size,
    output_channels,
    input_channels,
    input_height,
    input_width,
    output_height,  # Added: spatial dimensions of multiplier tensor
    output_width,
    
    # Triton-specific constants
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    """
    Optimized Triton kernel that fuses:
    - conv2d with stride=(1,1), padding=(0,0), dilation=(1,1), groups=1
    - + 1.0 operation  
    - / 2.0 operation
    - clamp(0.0, 1.0) operation
    - broadcast multiplication with multiplier tensor
    
    Improved version with better parallelism and memory access
    """
    # Get program IDs - process multiple output elements per program
    pid = tl.program_id(axis=0)
    
    # Each program handles a block of output channels and spatial positions
    output_block_idx = pid // ((output_height * output_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    spatial_block_idx = pid % ((output_height * output_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N)
    
    # Process one batch per program to simplify memory access
    batch_idx = output_block_idx % batch_size
    channel_start = (output_block_idx // batch_size) * BLOCK_SIZE_M
    
    # Process a block of channels
    for mi in range(BLOCK_SIZE_M):
        output_channel = channel_start + mi
        # Skip if out of bounds
        if output_channel >= output_channels or batch_idx >= batch_size:
            # Process spatial positions but do nothing
            for ni in range(BLOCK_SIZE_N):
                spatial_idx = spatial_block_idx * BLOCK_SIZE_N + ni
                if spatial_idx < output_height * output_width:
                    # Just do nothing for out-of-bound elements
                    pass
        else:
            # Process a block of spatial positions
            for ni in range(BLOCK_SIZE_N):
                spatial_idx = spatial_block_idx * BLOCK_SIZE_N + ni
                if spatial_idx < output_height * output_width:
                    # Compute spatial coordinates
                    h = spatial_idx // output_width
                    w = spatial_idx % output_width
                    
                    # Compute 1x1 convolution result using vectorized operations
                    conv_result = 0.0
                    for in_ch in range(input_channels):
                        # Load input value (for 1x1 conv, only spatial position [0,0] matters)
                        input_offset = batch_idx * input_channels + in_ch
                        input_mask = in_ch < input_channels
                        input_val = tl.load(conv_input_ptr + input_offset, mask=input_mask, other=0.0).to(tl.float32)
                        
                        # Load weight value
                        weight_offset = output_channel * input_channels + in_ch
                        weight_mask = in_ch < input_channels
                        weight_val = tl.load(weight_ptr + weight_offset, mask=weight_mask, other=0.0).to(tl.float32)
                        
                        conv_result += input_val * weight_val
                    
                    # Load bias
                    bias_mask = output_channel < output_channels
                    bias_val = tl.load(bias_ptr + output_channel, mask=bias_mask, other=0.0).to(tl.float32)
                    
                    # Apply fused arithmetic operations
                    conv_with_bias = conv_result + bias_val
                    intermediate = conv_with_bias / 2.0 + 0.5
                    clamped = tl.maximum(tl.minimum(intermediate, 1.0), 0.0)
                    
                    # Load multiplier at current spatial position
                    multiplier_offset = batch_idx * output_channels * output_height * output_width + \
                                       output_channel * output_height * output_width + \
                                       h * output_width + w
                    # Create mask for spatial bounds checking
                    spatial_mask = (h < output_height) & (w < output_width)
                    multiplier_val = tl.load(multiplier_ptr + multiplier_offset, mask=spatial_mask, other=0.0).to(tl.float32)
                    
                    # Final multiplication and store
                    result = clamped * multiplier_val
                    output_offset = batch_idx * output_channels * output_height * output_width + \
                                   output_channel * output_height * output_width + \
                                   h * output_width + w
                    tl.store(output_ptr + output_offset, result, mask=spatial_mask)

@torch.fx.wrap  
def fused_conv_activation_function(conv_input, weight, bias, multiplier):
    """
    Wrapper function for the fused conv2d + activation kernel - OPTIMIZED VERSION
    """
    # Get tensor shapes and properties
    batch_size, input_channels, input_height, input_width = conv_input.shape
    output_channels, in_ch, kernel_height, kernel_width = weight.shape
    
    # Validate shapes
    assert input_channels == in_ch, f"Input channels mismatch: {input_channels} vs {in_ch}"
    assert bias.shape[0] == output_channels, f"Bias channels mismatch: {bias.shape[0]} vs {output_channels}"
    assert multiplier.shape[0] == batch_size, f"Multiplier batch mismatch: {multiplier.shape[0]} vs {batch_size}"
    assert multiplier.shape[1] == output_channels, f"Multiplier channels mismatch: {multiplier.shape[1]} vs {output_channels}"
    
    # Use multiplier's spatial dimensions for output
    output_height, output_width = multiplier.shape[2], multiplier.shape[3]
    
    output = torch.empty((batch_size, output_channels, output_height, output_width), 
                        dtype=conv_input.dtype, device=conv_input.device)
    
    # Optimized kernel configuration - use larger blocks for better GPU utilization
    BLOCK_SIZE_M = 16   # Process 16 output channels per program
    BLOCK_SIZE_N = 32   # Process 32 spatial positions per program
    
    # Calculate grid dimensions
    grid_m = (batch_size * output_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (output_height * output_width + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_size = grid_m * grid_n
    
    # Launch optimized kernel
    fused_conv_activation_kernel[(grid_size,)](
        # Pointers
        conv_input_ptr=conv_input,
        weight_ptr=weight,
        bias_ptr=bias,
        multiplier_ptr=multiplier,
        output_ptr=output,
        
        # Convolution metadata
        batch_size=batch_size,
        output_channels=output_channels,
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        output_height=output_height,
        output_width=output_width,
        
        # Triton constants
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output

def replacement_func():
    """Returns the optimized fused function"""
    return fused_conv_activation_function