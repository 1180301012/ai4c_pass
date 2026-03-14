import torch
import triton
import triton.language as tl

# Pattern matching function for simple addition
def pattern(in_2, in_3):
    tmp_2 = in_2 + in_3
    return tmp_2

# Argument extraction function
def replacement_args(in_2, in_3):
    return (in_2, in_3)

# Triton kernel for optimized addition
@triton.jit
def simple_add_kernel(
    x1_ptr, x2_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    
    # Compute start index for this program
    start_idx = pid * BLOCK_SIZE
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x1 = tl.load(x1_ptr + offsets, mask=mask, other=0.0)
    x2 = tl.load(x2_ptr + offsets, mask=mask, other=0.0)
    
    # Addition
    out = x1 + x2
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap  
def simple_optimized_add(x1, x2):
    # Try to move both tensors to GPU for optimal performance
    try:
        # If either tensor is on CPU, move it to GPU if possible
        device = 'cuda:0'
        if x1.device.type != 'cuda':
            x1 = x1.to(device)
        if x2.device.type != 'cuda':
            x2 = x2.to(device)
        
        n_elements = x1.numel()
        out = torch.empty_like(x1)
        
        BLOCK_SIZE = 1024
        num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        simple_add_kernel[(num_programs,)](
            x1, x2, out, n_elements, BLOCK_SIZE
        )
        
        return out
        
    except Exception:
        # Fallback to regular addition if GPU operations fail
        return x1 + x2

# Replacement function (returns function reference)
def replacement_func():
    return simple_optimized_add