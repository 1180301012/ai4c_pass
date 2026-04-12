import torch
import triton
import triton.language as tl

# Pattern matching function - matches Conv2D + Dropout + Add pattern
def pattern(conv_input, conv_weight, conv_bias, residual_input):
    # The computation pattern from model.py:
    # Conv2D operation with specific parameters
    conv_output = torch.conv2d(conv_input, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    
    # Dropout with keep_prob=0.0 (no-op operation)
    dropout_output = torch.nn.functional.dropout(conv_output, 0.0, False, False)
    
    # Addition with residual
    final_output = dropout_output + residual_input
    
    return final_output

# Argument extraction function
def replacement_args(conv_input, conv_weight, conv_bias, residual_input):
    return (conv_input, conv_weight, conv_bias, residual_input)

# Optimized kernel with better memory access pattern
@triton.jit
def conv2d_add_fusion_kernel(
    input_ptr,      # [1, 256, 4, 256] - input tensor
    weight_ptr,     # [128, 256, 1, 1] - weight tensor  
    bias_ptr,       # [128] - bias tensor
    residual_ptr,   # [1, 128, 4, 256] - residual tensor
    output_ptr,     # [1, 128, 4, 256] - output tensor
    
    input_channels, # 256
    spatial_size,   # 4 * 256 = 1024
    output_channels, # 128
):
    # Program indices - each thread handles multiple output channels, one spatial position
    output_block_idx = tl.program_id(0)
    spatial_idx = tl.program_id(1)
    
    # Range of output channels this thread handles
    output_start = output_block_idx * 8  # 8 output channels per thread
    output_end = min(output_start + 8, output_channels)
    
    # Check bounds for spatial position
    if spatial_idx >= spatial_size or output_start >= output_channels:
        return
    
    # Process multiple output channels for this single spatial position
    for output_channel_idx in range(output_start, output_end):
        # Initialize result with bias for this output channel
        bias_val = tl.load(bias_ptr + output_channel_idx)
        result = bias_val  # Use direct assignment for better precision
        
        # Matrix multiply: accumulate input * weight for all input channels
        for c_in in range(input_channels):
            # Load weight for this input->output channel pair
            weight_offset = output_channel_idx * input_channels + c_in
            weight = tl.load(weight_ptr + weight_offset)
            
            # Load input value for this channel and spatial position  
            input_offset = c_in * spatial_size + spatial_idx
            input_val = tl.load(input_ptr + input_offset)
            
            # Compute weighted input and accumulate
            weighted_input = input_val * weight
            result += weighted_input
        
        # Load residual value and add to result
        residual_offset = output_channel_idx * spatial_size + spatial_idx
        residual_val = tl.load(residual_ptr + residual_offset)
        final_result = result + residual_val
        
        # Store result with precise assignment
        output_offset = output_channel_idx * spatial_size + spatial_idx
        tl.store(output_ptr + output_offset, final_result)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def conv2d_add_fusion(input, weight, bias, residual):
    # Get tensor dimensions
    input_shape = input.shape
    weight_shape = weight.shape
    
    batch_size, input_channels, input_height, input_width = input_shape
    output_channels = weight_shape[0]
    
    spatial_size = input_height * input_width
    
    # Create output tensor
    output_shape = (batch_size, output_channels, input_height, input_width) 
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)
    
    # Launch kernel with highly efficient thread configuration  
    # Optimize for both performance and numerical accuracy
    output_channels_per_thread = 8  # Each thread handles 8 output channels
    
    # Calculate grid dimensions
    num_output_threads = (output_channels + output_channels_per_thread - 1) // output_channels_per_thread
    num_spatial_threads = 1  # Each thread handles one spatial position for better precision
    
    # Launch kernel with 2D grid - optimized for accuracy
    # Total threads: num_output_threads * num_spatial_threads = 128 threads
    conv2d_add_fusion_kernel[(num_output_threads, num_spatial_threads)](
        input_ptr=input,
        weight_ptr=weight,
        bias_ptr=bias,
        residual_ptr=residual,
        output_ptr=output,
        input_channels=input_channels,
        spatial_size=spatial_size,
        output_channels=output_channels,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return conv2d_add_fusion