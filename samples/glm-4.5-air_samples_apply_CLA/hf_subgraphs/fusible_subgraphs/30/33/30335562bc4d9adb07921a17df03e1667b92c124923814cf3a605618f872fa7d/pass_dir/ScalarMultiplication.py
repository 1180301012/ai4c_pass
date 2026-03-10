import torch
import triton
import triton.language as tl

# Pattern matching: scalar multiplication with 0.0625
def pattern(x):
    return 0.0625 * x

# Argument extraction
def replacement_args(x):
    return (x,)

# Optimized kernel for scalar multiplication
@triton.jit
def scalar_mul_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    scalar_value,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input value
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Apply scalar multiplication
    out = scalar_value * x
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_scalar_mul(x):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Use hardcoded scalar value 0.0625 from the original computation
    scalar_value = 0.0625
    
    out = torch.empty_like(x)
    
    scalar_mul_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        scalar_value=scalar_value,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_scalar_mul