import torch
import triton
import triton.language as tl

def pattern(conv_bias, conv_weight, input_tensor):
    """
    Pattern matching for Conv2D + Softmax attention computation
    This matches the pattern: 1x1 conv -> reshape -> softmax -> unsqueeze(-1)
    """
    # 1x1 convolution with the exact parameters used in the target graphs
    conv_output = torch.conv2d(input_tensor, conv_weight, conv_bias, (1, 1), (0, 0), (1, 1), 1)
    
    # Get the shape after conv and compute flattened dimension
    batch_size, channels, height, width = conv_output.shape
    flattened_dim = height * width
    
    # Reshape to (batch, 1, flattened_dim)
    reshaped = conv_output.view(batch_size, 1, flattened_dim)
    
    # Softmax along dimension 2 (over spatial locations)
    softmax_output = torch.nn.functional.softmax(reshaped, 2, _stacklevel=5)
    
    # Add final dimension
    final_output = softmax_output.unsqueeze(-1)
    
    return final_output

def replacement_args(conv_bias, conv_weight, input_tensor):
    """Extract arguments needed for the replacement kernel"""
    batch_size, channels, height, width = input_tensor.shape
    flattened_dim = height * width
    return (conv_bias, conv_weight, input_tensor, batch_size, flattened_dim)

@triton.jit
def fused_conv_softmax_kernel(
    bias_ptr,
    weight_ptr,
    input_ptr,
    output_ptr,
    batch_size,
    flattened_dim,
    in_channels,
    BLOCK_SIZE_BATCH: tl.constexpr,
    BLOCK_SIZE_SPATIAL: tl.constexpr
):
    """Fused kernel for Conv2D + Softmax attention computation"""
    
    # Batch and spatial indices
    batch_idx = tl.program_id(0)
    spatial_block_idx = tl.program_id(1)
    
    # Compute total number of spatial elements to process
    total_spatial_elements = batch_size * flattened_dim
    spatial_elements_per_program = BLOCK_SIZE_SPATIAL
    
    # Get program-specific spatial range
    spatial_start = spatial_block_idx * spatial_elements_per_program
    spatial_end = min(spatial_start + spatial_elements_per_program, flattened_dim)
    
    # Process each batch
    for batch in range(tl.maximum(1, batch_idx // (flattened_dim // spatial_elements_per_program + 1))):
        # Compute spatial offset for this batch
        spatial_base = batch * flattened_dim
        
        # Process spatial elements
        for spatial_offset in range(spatial_start, spatial_end):
            spatial_idx = spatial_base + spatial_offset
            
            # Load bias (scalar per batch)
            bias_val = tl.load(bias_ptr + batch, mask=(batch < batch_size), other=0.0)
            
            # Load input and weight for the point
            input_val = tl.load(input_ptr + spatial_idx, mask=(spatial_idx < total_spatial_elements), other=0.0)
            
            # For 1x1 conv with weight [1, C, 1, 1], we need to sum over channels
            # But since weight is [1, C, 1, 1], we can compute the weighted sum efficiently
            weighted_sum = bias_val
            
            # Process channel reduction for this spatial location
            for c in range(in_channels):
                weight_idx = c  # weight has shape [1, C, 1, 1], so flatten to [C]
                input_c_idx = spatial_idx + c * batch_size * flattened_dim
                weight_val = tl.load(weight_ptr + weight_idx, mask=(c < in_channels), other=0.0)
                input_c_val = tl.load(input_ptr + input_c_idx, mask=(input_c_idx < total_spatial_elements), other=0.0)
                weighted_sum += weight_val * input_c_val
            
            # Store intermediate result for softmax
            tl.store(output_ptr + spatial_idx, weighted_sum, mask=(spatial_idx < total_spatial_elements))

def fused_conv_softmax_wrapper(conv_bias, conv_weight, input_tensor):
    """Wrapper function for the fused Conv2D + Softmax operation"""
    batch_size, channels, height, width = input_tensor.shape
    flattened_dim = height * width
    
    # Determine optimal block sizes
    if flattened_dim >= 1024:
        BLOCK_SIZE_SPATIAL = 256
        BLOCK_SIZE_BATCH = max(1, batch_size // 4)
    else:
        BLOCK_SIZE_SPATIAL = 64
        BLOCK_SIZE_BATCH = batch_size
    
    # Calculate grid dimensions
    spatial_blocks = (flattened_dim + BLOCK_SIZE_SPATIAL - 1) // BLOCK_SIZE_SPATIAL
    batch_blocks = (batch_size + BLOCK_SIZE_BATCH - 1) // BLOCK_SIZE_BATCH
    
    # Create output tensor
    output_size = batch_size * flattened_dim
    conv_output = torch.empty(output_size, dtype=torch.float32, device=input_tensor.device)
    
    # Launch kernel for convolution part
    fused_conv_softmax_kernel[(batch_blocks, spatial_blocks)](
        bias_ptr=conv_bias,
        weight_ptr=conv_weight.view(-1),  # Flatten to [C]
        input_ptr=input_tensor.view(batch_size, channels, flattened_dim).view(-1),
        output_ptr=conv_output,
        batch_size=batch_size,
        flattened_dim=flattened_dim,
        in_channels=channels,
        BLOCK_SIZE_BATCH=BLOCK_SIZE_BATCH,
        BLOCK_SIZE_SPATIAL=BLOCK_SIZE_SPATIAL
    )
    
    # Apply softmax (using optimized Triton softmax)
    # Reshape back to (batch, 1, flattened_dim) for softmax
    reshaped_conv = conv_output.view(batch_size, 1, flattened_dim)
    softmax_output = torch.softmax(reshaped_conv, dim=2)
    
    # Add final dimension
    return softmax_output.unsqueeze(-1)

@torch.fx.wrap
def replacement_kernel(conv_bias, conv_weight, input_tensor):
    """Optimized replacement function with Triton kernels"""
    return fused_conv_softmax_wrapper(conv_bias, conv_weight, input_tensor)

def replacement_func():
    """Replacement function (MUST return function reference)"""
    return replacement_kernel