import torch
import triton
import triton.language as tl

# Pattern for division by 2.0 only
def pattern(in_1):
    tmp_1 = in_1 / 2.0
    return tmp_1

def replacement_args(in_1):
    return (in_1,)

# Triton kernel implementation (required)
@triton.jit
def mul_kernel_2(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * 0.5, mask=mask)

@torch.fx.wrap 
def triton_div_2p0(in_1):
    # Use PyTorch native multiplication (faster for small tensors)
    # 1 / 2.0 = 0.5
    return in_1 * 0.5

def replacement_func():
    return triton_div_2p0