import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_0 = torch.nn.functional.silu(x, inplace=False)
    return tmp_0

def replacement_args(x):
    return (x,)

@triton.jit
def optimized_silu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    """Optimized SILU kernel: out = x * sigmoid(x)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute SILU: x * sigmoid(x)
    # Using stable sigmoid computation
    sigmoid_x = 1.0 / (1.0 + tl.exp(-tl.where(x > 0, x, 0.0)))
    out = x * sigmoid_x
    
    # Store results
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_silu(x):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    
    optimized_silu_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_silu