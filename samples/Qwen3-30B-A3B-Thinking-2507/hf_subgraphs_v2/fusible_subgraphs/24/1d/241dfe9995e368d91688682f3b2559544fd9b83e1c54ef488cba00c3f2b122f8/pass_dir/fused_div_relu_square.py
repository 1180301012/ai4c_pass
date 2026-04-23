import torch
import triton
import triton.language as tl

@triton.jit
def fused_div_relu_square_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    x_float = x.to(tl.float32)
    
    x_div = x_float / 11.313708498984761
    x_relu = tl.maximum(x_div, 0.0)
    x_sq = x_relu * x_relu
    
    out = x_sq.to(x.dtype)
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_div_relu_square(x):
    n = x.numel()
    BLOCK_SIZE = 256
    num_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    fused_div_relu_square_kernel[(num_blocks,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def pattern(x):
    tmp0 = x / 11.313708498984761
    tmp1 = torch.nn.functional.relu(tmp0)
    tmp2 = torch.square(tmp1)
    return tmp2

def replacement_args(x):
    return (x,)

def replacement_func():
    return fused_div_relu_square