import torch
import triton
import triton.language as tl

# Constant: 1.0 / 11.313708498984761
SCALE = 0.08838834764831844

@triton.jit
def fused_op_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    x = x * SCALE  # Faster than division
    x = tl.max(x, 0.0)  # ReLU
    x = x * x  # Square
    tl.store(out_ptr + offsets, x, mask=mask)

@torch.fx.wrap
def fused_op(x):
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    out = torch.empty_like(x)
    fused_op_kernel[(num_blocks,)](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def pattern(x):
    # Match exactly: x / constant → relu → square
    t0 = x / 11.313708498984761
    t1 = torch.nn.functional.relu(t0)
    t2 = torch.square(t1)
    return t2

def replacement_args(x):
    return (x,)

def replacement_func():
    return fused_op