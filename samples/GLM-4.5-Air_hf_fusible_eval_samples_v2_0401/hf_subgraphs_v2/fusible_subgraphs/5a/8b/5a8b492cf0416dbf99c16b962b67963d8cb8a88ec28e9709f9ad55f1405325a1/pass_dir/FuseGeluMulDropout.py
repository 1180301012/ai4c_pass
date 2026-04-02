import math
import torch
import triton
import triton.language as tl

def pattern(x, y):
    """Pattern matches GELU + multiplication + dropout"""
    tmp_0 = torch.nn.functional.gelu(x, approximate='none')
    tmp_1 = tmp_0 * y
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.1, False, False)
    return tmp_2

def replacement_args(x, y):
    return (x, y)

@triton.jit
def gelu_mul_dropout_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    dropout_p: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """Fused GELU + multiplication + kernel with dropout scaling"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # GELU approximation using sigmoid
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    gelu = x * sigmoid
    
    # Multiply + dropout scaling (training=False)
    scale = 1.0 - dropout_p
    out = gelu * y * scale
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_gelu_mul_dropout(x, y):
    """Wrapper function for fused GELU + multiplication + dropout"""
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    
    gelu_mul_dropout_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        dropout_p=0.1,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def replacement_func():
    return fused_gelu_mul_dropout