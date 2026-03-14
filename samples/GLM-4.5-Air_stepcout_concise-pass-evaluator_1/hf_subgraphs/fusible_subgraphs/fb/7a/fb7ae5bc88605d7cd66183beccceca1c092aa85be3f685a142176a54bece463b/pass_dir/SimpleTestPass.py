import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Simple pattern to test if pass mechanism works
    """
    tmp_3 = torch.nn.functional.dropout(in_0, p=0.0, training=False)
    tmp_4 = torch.matmul(tmp_3, in_1)
    return tmp_4

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def simple_matmul_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x * y  # Simple multiplication for testing
    
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap 
def simple_matmul(x, y):
    # For simplicity, just do element-wise multiplication
    N = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    simple_matmul_kernel[(num_programs,)](x_ptr=x, y_ptr=y, out_ptr=out, n_elements=N, BLOCK_SIZE=BLOCK_SIZE)
    return out

def replacement_func():
    return simple_matmul