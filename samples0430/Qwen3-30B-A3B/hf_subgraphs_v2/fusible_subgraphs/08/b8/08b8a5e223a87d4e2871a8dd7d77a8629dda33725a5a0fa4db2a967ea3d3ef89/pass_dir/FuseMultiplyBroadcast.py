import torch
import triton
import triton.language as tl

# Pattern matching for element-wise multiply with broadcasting

def pattern(a, b):
    return a * b

# Extract arguments for replacement

def replacement_args(a, b):
    return (a, b)

# Triton kernel for element-wise multiplication with broadcasting
@triton.jit
def triton_multiply_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    last_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(a_ptr + offsets, mask=mask)
    # Compute index into b using broadcasted dimension
    b_index = offsets % last_dim
    b = tl.load(b_ptr + b_index, mask=mask)
    out = a * b
    tl.store(out_ptr + offsets, out, mask=mask)

# Kernel wrapper for element-wise multiplication
@torch.fx.wrap
@torch.fx.wrap
def multiply_kernel(a, b):
    n_elements = a.numel()
    last_dim = b.shape[0]  # Length of the broadcasted dimension
    
    # Block size optimized for GPU
    BLOCK_SIZE = 1024
    
    # Grid size
    grid = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output
    out = torch.empty_like(a)
    
    # Launch kernel
    triton_multiply_kernel[grid](
        a, b, out,
        n_elements,
        last_dim,
        BLOCK_SIZE
    )
    
    return out

# Replacement function

def replacement_func():
    return multiply_kernel