import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor, bias_tensor):
    """Match Conv2D followed by Flatten pattern"""
    conv2d = torch.conv2d(input_tensor, weight_tensor, bias_tensor, (1, 1), (0, 0), (1, 1), 1)
    flattened = torch.flatten(conv2d, 2)
    return flattened

def replacement_args(input_tensor, weight_tensor, bias_tensor):
    """Extract arguments needed for the fused kernel"""
    return (input_tensor, weight_tensor, bias_tensor)

@triton.jit
def conv2d_flatten_fused_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    input_batch, input_channels, input_height, input_width,
    output_channels, spatial_size,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    """Fused Conv2D + Flatten kernel for 1x1 convolution
    
    Each program computes one element of the output [batch, output_channels * spatial_size].
    The computation: y[b, c] = sum_{k} (x[b, k, s] * w[c, k]) + bias[c]
    where:
    - x is [batch, input_channels, height, width] -> flattened spatially
    - w is [output_channels, input_channels]  
    - bias is [output_channels]
    - output is [batch * output_channels * spatial_size]
    """
    # Get program IDs  
    pid = tl.program_id(0)
    
    # Calculate which output element this program computes
    # Output shape: [batch_size, output_channels, spatial_size]
    batch_idx = pid // (output_channels * spatial_size)
    channel_idx = (pid // spatial_size) % output_channels
    spatial_pos = pid % spatial_size
    
    # Convert linear spatial position to (height, width) coordinates  
    spatial_height = spatial_pos // input_width
    spatial_width_idx = spatial_pos % input_width
    
    # Create scalar masks for bounds checking
    batch_valid = batch_idx < input_batch
    channel_valid = channel_idx < output_channels
    k_offset = tl.arange(0, BLOCK_SIZE_K)
    k_mask = k_offset < input_channels
    
    # Initialize accumulator
    accumulator = 0.0
    
    # Load bias if available and channel is valid
    if channel_valid and bias_ptr is not None:
        bias_value = tl.load(bias_ptr + channel_idx)
        accumulator += bias_value
    
    # Main computation loop over input channels
    for k_start in range(0, input_channels, BLOCK_SIZE_K):
        # Calculate current channel range
        current_k = k_start + k_offset
        k_mask_current = current_k < input_channels
        
        # Load input data for current channel range
        if batch_valid:
            # Calculate input indices: [batch_idx, current_k, spatial_height, spatial_width_idx]
            input_indices = (batch_idx * input_channels + current_k) * input_height * input_width + spatial_height * input_width + spatial_width_idx
            input_values = tl.load(input_ptr + input_indices, mask=k_mask_current, other=0.0)
            
            # Load weights for current channel range  
            if channel_valid:
                # Calculate weight indices: [channel_idx, current_k]
                weight_indices = channel_idx * input_channels + current_k
                weight_values = tl.load(weight_ptr + weight_indices, mask=k_mask_current, other=0.0)
                
                # Multiply and accumulate
                accumulator += tl.sum(input_values * weight_values)
    
    # Store result if in bounds
    if batch_valid & channel_valid:
        # Calculate 3D output tensor index: [batch_idx, channel_idx, spatial_pos]
        output_index = batch_idx * output_channels * spatial_size + channel_idx * spatial_size + spatial_pos
        tl.store(output_ptr + output_index, accumulator.to(tl.float32 if output_ptr.dtype.element_ty == 0x6 else tl.float16))

@torch.fx.wrap
def conv2d_flatten_fused(input_tensor, weight_tensor, bias_tensor):
    """Wrapper for fused Conv2D + Flatten operation"""
    input_shape = input_tensor.shape
    batch_size, input_channels, input_height, input_width = input_shape
    output_channels = weight_tensor.shape[0]
    spatial_size = input_height * input_width
    
    # Final output shape after flatten: [batch, output_channels, spatial_size]
    output_tensor = torch.empty((batch_size, output_channels, spatial_size), dtype=input_tensor.dtype, device=input_tensor.device)
    output_total_size = output_tensor.numel()  # Total number of elements for grid size
    
    # Optimal tile sizes for vectorization
    BLOCK_SIZE_M = 1      # Each program computes one output element
    BLOCK_SIZE_N = 1      # One-dimensional kernel
    BLOCK_SIZE_K = 32     # Process multiple channels per iteration
    
    # Launch kernel - grid size is total number of output elements
    conv2d_flatten_fused_kernel[(output_total_size,)](
        input_tensor,
        weight_tensor,
        bias_tensor,
        output_tensor,
        batch_size, input_channels, input_height, input_width,
        output_channels, spatial_size,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return output_tensor

def replacement_func():
    """Return the fused function"""
    return conv2d_flatten_fused