import torch
import triton
import triton.language as tl

def pattern(x):
    # This matches the pattern: flatten(2) followed by transpose(1, 2)
    # The original calls:
    #   tmp_7 = conv3d.flatten(2);  conv3d = None
    #   tmp_8 = tmp_7.transpose(1, 2);  tmp_7 = None
    flattened = x.flatten(2)
    transposed = flattened.transpose(1, 2)
    return transposed

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_flatten_transpose_kernel(
    input_ptr, output_ptr,
    batch_size, channel_dim, d1, d2,
    BLOCK_SIZE: tl.constexpr
):
    """Optimized kernel that combines flatten(2) and transpose(1, 2) operations"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Total elements in the output tensor
    output_elements = batch_size * d1 * channel_dim
    mask = offsets < output_elements
    
    if mask.any():
        # Flatten and transpose logic:
        # Original: [batch, channels, d1, d2] -> flatten(2) -> [batch, channels, d1*d2] -> transpose(1,2) -> [batch, d1*d2, channels]
        # We can directly compute the mapping from input to output
        
        # Calculate output coordinates
        batch_idx = offsets // (d1 * channel_dim)
        pos_in_d1_d2 = offsets % (d1 * channel_dim)
        d1_idx = pos_in_d1_d2 // channel_dim
        channel_idx = pos_in_d1_d2 % channel_dim
        
        # Calculate corresponding input coordinates
        # In flattened tensor, flatten(2) gives [batch, channels, d1*d2]
        # So input coordinate for this output is [batch_idx, channel_idx, pos_in_d1_d2]
        input_pos_in_d1_d2 = pos_in_d1_d2
        input_d2_idx = input_pos_in_d1_d2 % d2
        input_d1_idx = input_pos_in_d1_d2 // d2
        
        # Calculate input index
        input_idx = (batch_idx * channel_dim + channel_idx) * (d1 * d2) + input_d1_idx * d2 + input_d2_idx
        
        # Load data and store directly in output order
        input_val = tl.load(input_ptr + input_idx, mask=mask)
        tl.store(output_ptr + offsets, input_val, mask=mask)

@torch.fx.wrap
def optimized_flatten_transpose(x):
    """Optimized function that combines flatten(2) and transpose(1, 2) into a single kernel"""
    batch_size, channel_dim, d1, d2 = x.shape
    
    # Output shape: [batch_size, d1*d2, channel_dim]
    d1_d2 = d1 * d2
    output_shape = (batch_size, d1_d2, channel_dim)
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    total_elements = batch_size * d1_d2 * channel_dim
    
    # Launch kernel
    BLOCK_SIZE = 256
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_flatten_transpose_kernel[grid_size](
        x, output,
        batch_size, channel_dim, d1, d2,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_flatten_transpose