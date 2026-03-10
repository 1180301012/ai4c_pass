import torch
import triton
import triton.language as tl

@triton.jit
def add_gelu_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Addition followed by GELU
    sum_val = x + y
    # GELU approximation using tanh for better performance
    gelu_val = sum_val * 0.5 * (1.0 + tl.tanh(sum_val * 0.7978845608028654 * (1.0 + 0.044715 * sum_val * sum_val)))
    
    # Store result
    tl.store(out_ptr + offsets, gelu_val, mask=mask)

@torch.fx.wrap
def triton_add_gelu(x, y):
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape")
    
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    add_gelu_kernel[(num_programs,)](
        x=x,
        y=y, 
        out=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def pattern(x, y):
    # Pattern matches: x += y (in-place add), then gelu(x)
    x += y
    tmp_5 = torch.nn.functional.gelu(x, approximate='none')
    return tmp_5

def replacement_args(x, y):
    return (x, y)

def replacement_func():
    return triton_add_gelu