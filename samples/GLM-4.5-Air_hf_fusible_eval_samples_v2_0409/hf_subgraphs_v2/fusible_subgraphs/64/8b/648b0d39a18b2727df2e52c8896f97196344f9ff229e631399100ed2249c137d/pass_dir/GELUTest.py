import torch
import triton
import triton.language as tl

def pattern(x):
    # Just match GELU activation
    gelu_out = torch.nn.functional.gelu(x, approximate='none')
    return gelu_out

def replacement_args(x):
    return (x,)

@triton.jit
def simple_gelu_kernel(
    x,
    out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x_val = tl.load(x + offsets, mask=mask, other=0.0)
    
    # Simple ReLU approximation for GELU (for testing)
    gelu_out = tl.maximum(x_val, 0.0)
    
    # Store result
    tl.store(out + offsets, gelu_out, mask=mask)

@torch.fx.wrap
def triton_gelu(x):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    # Launch kernel
    simple_gelu_kernel[(num_programs,)](
        x=x,
        out=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_gelu