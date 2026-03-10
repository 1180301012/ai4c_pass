import torch
import triton
import triton.language as tl

def pattern(x, y):
    # Complete fusion: SILU(x) * y (skipping tmp variables and final dropout)
    return torch.nn.functional.silu(x) * y

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fused_silu_mul_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Complete fusion: silu(x) * y = (x * sigmoid(x)) * y
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    silu_x = x * sigmoid_x
    out = silu_x * y
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_fused_silu_mul(x, y):
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    fused_silu_mul_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return triton_fused_silu_mul