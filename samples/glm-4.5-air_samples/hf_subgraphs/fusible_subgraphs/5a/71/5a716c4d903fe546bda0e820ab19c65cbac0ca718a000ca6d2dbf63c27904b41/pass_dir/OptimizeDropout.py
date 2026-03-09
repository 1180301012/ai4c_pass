import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_5 = torch.nn.functional.dropout(x, 0.1, False, False)
    return tmp_5

def replacement_args(x):
    return (x,)

@triton.jit
def simple_dropout_kernel(x_ptr, out_ptr, n_elements, scale: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input and apply scaling
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    result = x * scale
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_dropout(x):
    # When training=False, dropout is just scaling by (1-0.1) = 0.9
    dropout_scale = 0.9
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    simple_dropout_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        scale=dropout_scale,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_dropout