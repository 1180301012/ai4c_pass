import torch
import triton
import triton.language as tl

def input_tensor_func(input_tensor):
    """
    Pattern matches: flatten(2) followed by transpose(1, 2)
    This is a common preparation step for layer normalization.
    """
    flattened = input_tensor.flatten(2)
    result = flattened.transpose(1, 2)
    return result

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def flatten_transpose_kernel(
    input_ptr, output_ptr,
    batch_size, channels, height, width, total_spatial,
    BLOCK_SIZE: tl.constexpr
):
    """
    Optimized kernel that combines flatten(2) and transpose(1, 2) operations.
    Input shape: [batch_size, channels, height, width]
    Output shape: [batch_size, height*width, channels]
    
    The kernel directly maps from input layout [B, C, H, W] to output layout [B, HW, C]
    without creating an intermediate flattened tensor.
    """
    # Get program ID for batch and spatial position
    batch_idx = tl.program_id(0)
    spatial_idx = tl.program_id(1)
    
    # Each program handles one position in the flattened spatial dimension
    for c in range(0, channels, BLOCK_SIZE):
        channel_offset = tl.arange(0, BLOCK_SIZE)
        
        # Calculate input and output positions
        # Input: [B, C, H, W] -> linear offset = B*C*H*W + C*H*W + H*W + W
        # Output: [B, HW, C] -> linear offset = B*HW*C + HW*C + C
        
        h = spatial_idx // width
        w = spatial_idx % width
        
        # Load input element
        input_offset = batch_idx * channels * height * width + \
                      (c + channel_offset) * height * width + h * width + w
        input_vals = tl.load(input_ptr + input_offset,
                           mask=(c + channel_offset) < channels)
        
        # Store output element  
        output_offset = batch_idx * total_spatial * channels + \
                       spatial_idx * channels + (c + channel_offset)
        tl.store(output_ptr + output_offset, input_vals,
               mask=(c + channel_offset) < channels)

@torch.fx.wrap
def optimized_flatten_transpose(input_tensor):
    batch_size, channels, height, width = input_tensor.shape
    total_spatial = height * width
    
    # Triton kernel launch configuration
    BLOCK_SIZE = 256  # Number of channels to process per program
    
    # Calculate grid dimensions: one program per (batch, spatial_position)
    grid_size = (batch_size * total_spatial,)
    
    # Create output tensor with shape [batch_size, height*width, channels]
    output = torch.empty((batch_size, total_spatial, channels),
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    flatten_transpose_kernel[grid_size](
        input_tensor, output,
        batch_size, channels, height, width, total_spatial,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_flatten_transpose