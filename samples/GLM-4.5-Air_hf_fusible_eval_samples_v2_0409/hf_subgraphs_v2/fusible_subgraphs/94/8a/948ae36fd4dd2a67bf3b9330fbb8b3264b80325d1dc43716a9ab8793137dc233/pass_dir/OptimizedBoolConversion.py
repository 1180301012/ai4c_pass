import torch
from torch import device
import triton
import triton.language as tl

# Pattern matching function
def pattern(input_tensor):
    """Matches input_tensor.to(device=device(type='cuda', index=0), dtype=torch.bool)"""
    return input_tensor.to(device=device(type='cuda', index=0), dtype=torch.bool)

# Argument extraction function
def replacement_args(input_tensor):
    """Extract the input tensor for the dtype conversion"""
    return (input_tensor,)

# Optimized triton kernel for bool conversion
@triton.jit
def bool_conversion_kernel(
    input_ptr,
    output_ptr,
    num_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Convert int64 tensor to bool tensor efficiently"""
    # Each program processes one block of elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Calculate indices for this program
    indices = block_start + tl.arange(0, BLOCK_SIZE)
    mask = indices < num_elements
    
    # Load input values (int64)
    input_vals = tl.load(input_ptr + indices, mask=mask, other=0)
    
    # Convert to bool: non-zero becomes True, zero becomes False
    output_vals = (input_vals != 0).to(output_ptr.type())
    
    # Store results
    tl.store(output_ptr + indices, output_vals, mask=mask)

@torch.fx.wrap
def optimized_bool_conversion(input_tensor):
    """Optimized bool conversion implementation using Triton"""
    # Get tensor properties
    num_elements = input_tensor.numel()
    device = input_tensor.device
    
    # Since input is already on GPU, we can skip device transfer
    # Create output tensor with bool dtype
    output = torch.empty(input_tensor.shape, dtype=torch.bool, device=device)
    
    # Determine block size and grid size
    BLOCK_SIZE = 1024
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch the optimized kernel
    bool_conversion_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        num_elements=num_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return optimized_bool_conversion