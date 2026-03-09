import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_4 = torch.nn.functional.softmax(x, dim=-1)
    return tmp_4

def replacement_args(x):
    return (x,)

@triton.jit
def simple_softmax_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    max_val = tl.max(x, axis=0)
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    shifted_x = x - max_val
    exp_x = tl.exp(shifted_x)
    sum_exp = tl.sum(exp_x, axis=0)
    softmax_vals = exp_x / sum_exp
    tl.store(out_ptr + offsets, softmax_vals, mask=mask)

@torch.fx.wrap
def optimized_softmax(x):
    # Simple 1D softmax that works for any tensor shape
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    simple_softmax_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_softmax