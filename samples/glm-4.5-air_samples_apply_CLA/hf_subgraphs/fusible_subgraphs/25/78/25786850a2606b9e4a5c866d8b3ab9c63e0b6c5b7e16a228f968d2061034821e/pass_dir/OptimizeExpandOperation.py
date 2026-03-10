import torch
import triton
import triton.language as tl

# Pattern matching function - matches the cls token expansion
def pattern(in_3):
    tmp_8 = in_3.expand(1, -1, -1)
    return tmp_8

# Argument extraction function
def replacement_args(in_3):
    return (in_3,)

# Triton kernel for optimized tensor expansion
@triton.jit
def expand_kernel(
    input_ptr,
    output_ptr,
    original_size,
    target_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < target_size[-1]  # Only mask along the last dimension
    
    # Load input data (broadcast across first two dimensions)
    input_vals = tl.load(input_ptr + offsets % original_size[-1], mask=mask, other=0.0)
    
    # Write output data at corresponding positions
    tl.store(output_ptr + offsets, input_vals, mask=mask)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def optimized_expand(tensor, target_shape):
    """
    Optimized expand operation that handles broadcasting efficiently.
    """
    original_shape = tensor.shape
    
    # Create output tensor
    output = torch.empty(target_shape, dtype=tensor.dtype, device=tensor.device)
    
    # For the specific case [1, 1, 768] -> [1, 14, 768] in our pattern
    # The first dimension stays 1, second dimension expands from 1 to 14, third stays 768
    
    # If it's just broadcasting the second dimension
    if len(original_shape) == 3 and len(target_shape) == 3 and \
       original_shape[0] == target_shape[0] == 1 and \
       original_shape[2] == target_shape[2]:
        
        # Simple broadcast case: clone and repeat along the second dimension
        expanded = tensor.repeat(1, target_shape[1], 1)
        output.copy_(expanded)
        return output
    else:
        # For other cases, copy input to output (simple fallback)
        output.copy_(tensor)
        return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_expand