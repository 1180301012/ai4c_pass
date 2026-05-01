import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    return tmp_0

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def silu_kernel(
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sigmoid = 1.0 / (1.0 + tl.exp(-x))
    y = x * sigmoid
    tl.store(x_ptr + offsets, y, mask=mask)

@torch.fx.wrap
def silu_inplace(tensor):
    N = tensor.numel()
    BLOCK_SIZE = 128
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    silu_kernel[(num_blocks,)](
        x_ptr=tensor,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return tensor

def replacement_func():
    return silu_inplace