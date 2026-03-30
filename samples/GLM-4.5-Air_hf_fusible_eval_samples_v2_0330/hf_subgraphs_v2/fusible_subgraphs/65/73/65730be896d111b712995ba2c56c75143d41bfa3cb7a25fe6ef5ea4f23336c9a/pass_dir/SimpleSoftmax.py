import torch
import triton
import triton.language as tl

def pattern(x):
    """Simple pattern: just softmax"""
    return torch.nn.functional.softmax(x, dim=-1)

def replacement_args(x):
    return (x,)

@triton.jit
def simple_softmax_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple optimized softmax kernel"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Simple softmax with numerical stability
    max_val = tl.max(x, mask=mask)
    max_val = tl.broadcast_to(max_val, x.shape)
    shifted = x - max_val
    exp_val = tl.exp(shifted, mask=mask)
    exp_sum = tl.sum(exp_val, mask=mask)
    exp_sum = tl.broadcast_to(exp_sum, exp_val.shape)
    out = exp_val / exp_sum
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def simple_softmax_func(x):
    """Simple softmax function"""
    x_flat = x.reshape(-1)
    out = torch.empty_like(x)
    out_flat = out.reshape(-1)
    
    N = x_flat.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    simple_softmax_kernel[(num_programs,)](
        x_ptr=x_flat,
        out_ptr=out_flat,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return simple_softmax_func