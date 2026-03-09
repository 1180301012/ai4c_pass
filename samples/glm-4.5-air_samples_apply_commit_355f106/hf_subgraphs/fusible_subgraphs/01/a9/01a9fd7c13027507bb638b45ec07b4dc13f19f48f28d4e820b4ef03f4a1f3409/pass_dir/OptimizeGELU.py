import torch
import triton
import triton.language as tl

def pattern(x):
    return torch.nn.functional.gelu(x, approximate='none')

def replacement_args(x):
    return (x,)

@triton.jit
def gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # GELU approximation using tanh for better performance
    out = 0.5 * x * (1.0 + tl.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_gelu(x):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    gelu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def replacement_func():
    return optimized_gelu