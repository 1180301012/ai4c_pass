import torch
import triton
import triton.language as tl

def pattern(x, y):
    return x+y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def optimized_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized Triton kernel for element-wise addition"""
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform addition
    out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_add(x, y):
    """Optimized addition function using Triton"""
    # Handle case where y might be a constant (scalar)
    if not isinstance(y, torch.Tensor):
        # Convert constant to tensor with same properties as x
        y = torch.full_like(x, fill_value=y)
    
    # Check if tensors are on the same device
    if x.device != y.device:
        y = y.to(x.device)
    
    # Ensure both tensors have compatible shapes
    if x.shape != y.shape:
        # Handle broadcasting
        y_broadcast = y.expand_as(x)
    else:
        y_broadcast = y
    
    # Perform element-wise addition using optimized kernel
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    optimized_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y_broadcast,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_add