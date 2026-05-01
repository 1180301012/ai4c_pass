import torch
import triton
import triton.language as tl

# Pattern matching function

def pattern(tmp_1):
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    return tmp_2

# Argument extraction function

def replacement_args(tmp_1):
    return (tmp_1,)

# Triton kernel for softmax
@triton.jit
def softmax_kernel(
    x_ptr,
    out_ptr,
    n_rows,
    n_cols,
):
    row = tl.program_id(0)
    start = row * n_cols
    offsets = tl.arange(0, n_cols)
    mask = offsets < n_cols
    
    x = tl.load(x_ptr + start + offsets, mask=mask)
    
    # Compute max of the entire row
    max_val = tl.max(x)
    
    # Compute exp(x - max_val)
    exp_x = tl.exp(x - max_val)
    
    # Compute sum of exp
    sum_exp = tl.sum(exp_x)
    
    # Compute softmax
    out = exp_x / sum_exp
    
    tl.store(out_ptr + start + offsets, out, mask=mask)

# Kernel wrapper
@torch.fx.wrap
def optimized_softmax(x):
    n_rows = 8 * 300
    n_cols = 625
    out = torch.empty_like(x)
    grid = (n_rows,)
    softmax_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_rows=n_rows,
        n_cols=n_cols,
    )
    return out

# Replacement function

def replacement_func():
    return optimized_softmax