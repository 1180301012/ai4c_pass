import torch
import triton
import triton.language as tl

def pattern(x, y):
    result = x + y
    silu_result = torch.nn.functional.silu(result, inplace=False)
    return silu_result

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fused_add_silu_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Addition
    add_result = x + y
    # SiLU activation: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    sigmoid = 1.0 / (1.0 + tl.exp(-add_result))
    silu_result = add_result * sigmoid
    
    tl.store(out_ptr + offsets, silu_result, mask=mask)

@torch.fx.wrap
def fused_add_silu(x, y):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    fused_add_silu_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_add_silu