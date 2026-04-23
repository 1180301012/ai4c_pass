import torch
import triton
import triton.language as tl

def pattern(in_1):
    tmp_1 = torch.nn.functional.silu(in_1, inplace=True)
    return tmp_1

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    sigmoid = tl.math.sigmoid(x)
    out = x * sigmoid
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def silu(x):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    silu_kernel[(num_blocks,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def replacement_func():
    return silu