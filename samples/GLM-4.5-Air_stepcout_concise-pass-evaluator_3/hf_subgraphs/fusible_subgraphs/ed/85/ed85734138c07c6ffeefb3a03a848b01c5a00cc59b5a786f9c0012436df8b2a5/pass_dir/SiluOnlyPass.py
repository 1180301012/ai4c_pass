import torch
import triton
import triton.language as tl

# Pattern - match exactly as in the model (with inplace=True)
def pattern(x):
    return torch.nn.functional.silu(x, inplace=True)

# Extract args
def replacement_args(x):
    return (x,)

# Optimized kernel
@triton.jit
def silu_kernel_impl(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    result = x * sigmoid
    
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_silu(a):
    n_elements = a.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(a)
    silu_kernel_impl[(num_programs,)](
        a, out, n_elements, BLOCK_SIZE,
    )
    return out

def replacement_func():
    return optimized_silu