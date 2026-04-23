import torch
import triton
import triton.language as tl

def pattern(in_1):
    tmp_0 = in_1 * 0.1767766952966369
    return tmp_0

def replacement_args(in_1):
    return (in_1,)

@triton.jit
def scale_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    scalar: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = x * scalar
    tl.store(y_ptr + offsets, y, mask=mask)

@torch.fx.wrap
def scale_wrapper(x):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    scale_kernel[(num_blocks,)](
        x,
        out,
        n_elements,
        0.1767766952966369,
        BLOCK_SIZE
    )
    return out

def replacement_func():
    return scale_wrapper