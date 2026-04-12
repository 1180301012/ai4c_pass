import torch
import triton
import triton.language as tl

@triton.jit
def reshape_kernel(
    x_ptr,
    out_ptr,
    input_elements,
    total_output_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized Triton kernel for reshape operation"""
    pid = tl.program_id(0)
    
    # Each program handles a block of elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid output indices
    output_mask = offsets < total_output_elements
    
    # Create mask for valid input indices
    input_mask = offsets < input_elements
    
    # Combined mask - only process valid output positions
    combined_mask = output_mask & input_mask
    
    # Load input element
    x = tl.load(x_ptr + offsets, mask=combined_mask, other=0.0)
    
    # Store to output  
    tl.store(out_ptr + offsets, x, mask=combined_mask)

@torch.fx.wrap
def optimized_reshape(x):
    """Optimized reshape function using Triton kernel"""
    # Calculate input and output element counts
    input_elements = x.numel()
    
    # For the pattern: reshape(1, -1, 16, 9) -> reshape(-1, 8, 9)
    # We can go directly from input to final shape
    # The original transform changes shape from [1, groups, 16, 9] to [groups*2, 8, 9]
    # where groups*16*9 = input_elements/1
    
    # Calculate groups from input shape (after first reshape)
    # Groups would be input_elements / (1 * 16 * 9) = input_elements / 144
    groups = input_elements // (1 * 16 * 9)
    final_shape = (groups * 2, 8, 9)  # groups*2, 8, 9
    
    # Create output tensor
    out = torch.empty(final_shape, dtype=x.dtype, device=x.device)
    
    # Use Triton kernel for the transformation
    total_output_elements = out.numel()
    BLOCK_SIZE = 1024
    grid_size = (total_output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Directly copy data (optimized reshape)
    reshape_kernel[(grid_size,)](
        x_ptr=x,
        out_ptr=out,
        input_elements=input_elements,
        total_output_elements=total_output_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def pattern(x):
    # Match consecutive reshape operations similar to the computation
    tmp_4 = x.reshape(1, -1, 16, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 8, 9])
    return tmp_5

def replacement_args(x):
    return (x,)

def replacement_func():
    return optimized_reshape