import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(tmp_2):
    const = 128
    size = torch.sym_sum([const, tmp_2])
    return torch.ones((size,), dtype=torch.float32, device='cuda')

# Argument extraction function
def replacement_args(tmp_2):
    # Compute the actual size value (128 + tmp_2)
    size_val = 128 + tmp_2.item()
    return (size_val,)

# Triton kernel for initializing tensor to 1.0
@triton.jit
def ones_kernel(out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate block offset
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Store 1.0 to output
    tl.store(out_ptr + offsets, 1.0, mask=mask)

# Kernel wrapper for Triton execution
@torch.fx.wrap
def kernel_wrapper(size_val):
    # Allocate output tensor
    out = torch.empty((size_val,), dtype=torch.float32, device='cuda')
    
    # Configure Triton grid
    n_elements = size_val
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    ones_kernel[(num_programs,)](out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

# Replacement function
def replacement_func():
    return kernel_wrapper