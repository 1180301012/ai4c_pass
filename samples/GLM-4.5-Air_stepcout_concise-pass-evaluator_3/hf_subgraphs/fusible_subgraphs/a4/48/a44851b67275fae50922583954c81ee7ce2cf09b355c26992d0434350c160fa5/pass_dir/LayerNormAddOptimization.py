import torch
import triton
import triton.language as tl

# Pattern matching function for the main addition in layer norm
def pattern(in_2, in_3):
    tmp_2 = in_2 + in_3
    return tmp_2

# Argument extraction function
def replacement_args(in_2, in_3):
    return (in_2, in_3)

# Triton kernel for main tensor addition
@triton.jit
def main_add_kernel(
    x1_ptr, x2_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Compute start index for this program
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs with proper bounds checking
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
    
    # Addition - optimized for large tensors
    out = x1 + x2
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def main_optimized_add(x1, x2):
    """Optimized addition for main computation tensors (large tensors)"""
    # Only apply optimization for larger tensors where it makes sense
    if x1.numel() < 10000:  # Skip for very small tensors
        return x1 + x2
        
    n_elements = x1.numel()
    
    # For large tensors, use optimized GPU path
    try:
        # Ensure both tensors are on GPU
        if x1.device.type != 'cuda':
            x1 = x1.to('cuda:0')
        if x2.device.type != 'cuda':
            x2 = x2.to('cuda:0')
            
        out = torch.empty_like(x1)
        
        # Use larger block size for better utilization
        BLOCK_SIZE = 2048  # Larger for better occupancy
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        main_add_kernel[(num_programs,)](
            x1, x2, out, n_elements, BLOCK_SIZE
        )
        
        return out
        
    except Exception:
        # Fallback to regular addition
        return x1 + x2

# Replacement function (returns function reference)  
def replacement_func():
    return main_optimized_add