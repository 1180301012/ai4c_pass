import torch
import triton
import triton.language as tl

def pattern(in_4):
    """Match reshape operation that can be optimized"""
    return in_4.reshape(1, 256, 16, 16)

def replacement_args(in_4):
    """Extract arguments for the optimized reshape"""
    return (in_4,)

@triton.jit
def optimized_reshape_kernel(
    input_ptr,
    output_ptr,
    original_shape,
    new_shape,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized reshape kernel that handles [4, C, H] -> [1, 4*C, H//8, 8] pattern"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Directly store to output position (since we're just rearranging data layout)
    tl.store(output_ptr + offsets, input_data, mask=mask)

@torch.fx.wrap
def optimized_reshape(in_4):
    """Optimized reshape that handles the specific input->output pattern"""
    original_shape = in_4.shape
    
    # Determine target shape based on input dimensions
    if len(original_shape) == 3 and original_shape[0] == 4:
        # [4, C, H] -> [1, 4*C, H//8, 8] pattern
        total_channels = original_shape[1]
        spatial_dim = original_shape[2]
        
        if spatial_dim % 8 == 0:
            new_height = spatial_dim // 8
            new_width = 8
        else:
            new_height = 8
            new_width = spatial_dim // 8
            
        new_shape = (1, 4 * total_channels, new_height, new_width)
    elif len(original_shape) == 4:
        # Already in correct format
        new_shape = original_shape
    else:
        # Fallback to standard reshape
        return in_4.reshape(1, -1, 8, 8)
    
    n_elements = in_4.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty(new_shape, dtype=in_4.dtype, device=in_4.device)
    
    # Launch kernel only if we need actual reshaping
    if new_shape != original_shape:
        optimized_reshape_kernel[(num_programs,)](
            in_4,
            output,
            original_shape,
            new_shape,
            n_elements,
            BLOCK_SIZE,
        )
    else:
        # No actual reshaping needed
        output = in_4
    
    return output

def replacement_func():
    """Return optimized reshape function"""
    return optimized_reshape