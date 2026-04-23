import torch
import triton
import triton.language as tl

@triton.jit
def silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    start = block_id * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    exp_neg_x = tl.exp(-x)
    sigmoid_x = 1 / (1 + exp_neg_x)
    result = x * sigmoid_x
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_silu(x):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    silu_kernel[(num_blocks,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def pattern(in_0):
    tmp_0 = torch.nn.functional.silu(in_0, inplace = True)
    return tmp_0

def replacement_args(in_0):
    return (in_0,)

def replacement_func():
    return optimized_silu