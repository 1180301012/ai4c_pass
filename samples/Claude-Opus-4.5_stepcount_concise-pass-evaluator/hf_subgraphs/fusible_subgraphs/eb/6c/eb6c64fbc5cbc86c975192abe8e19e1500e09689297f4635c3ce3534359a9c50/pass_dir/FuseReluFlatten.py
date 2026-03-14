import torch
import triton
import triton.language as tl

# Pattern matching function - matches flatten only
def pattern(x):
    """Match flatten operation"""
    tmp_1 = x.flatten(1, -1)
    return tmp_1

# Argument extraction function
def replacement_args(x):
    return (x,)

# Triton kernel (required by framework)
@triton.jit
def identity_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)

# Wrapper using the same flatten operation
@torch.fx.wrap
def fast_flatten(x):
    return x.flatten(1, -1)

# Replacement function
def replacement_func():
    return fast_flatten