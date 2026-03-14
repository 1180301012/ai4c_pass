import torch
import triton
import triton.language as tl

@triton.jit
def simple_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap  
def simple_triton_add(x, y):
    if x.shape != y.shape:
        # For simplicity, just return x + y if shapes don't match
        return x + y
    
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    
    simple_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

# Pattern matching function - must exactly match the model  
def pattern(x, y):
    return x + y

# Argument extraction function
def replacement_args(x, y):
    return (x, y)

# Replacement function - returns the optimized kernel
def replacement_func():
    return simple_triton_add