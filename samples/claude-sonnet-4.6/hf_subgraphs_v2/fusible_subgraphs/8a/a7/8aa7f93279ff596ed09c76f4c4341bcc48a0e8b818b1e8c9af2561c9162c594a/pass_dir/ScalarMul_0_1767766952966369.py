import torch
import triton
import triton.language as tl


def pattern(x):
    return x * 0.1767766952966369


def replacement_args(x):
    return (x,)


@triton.jit
def _scalar_mul_0_1767_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    SCALAR = 0.1767766952966369
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x * SCALAR, mask=mask)


@torch.fx.wrap
def scalar_mul_0_1767(x):
    n = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    _scalar_mul_0_1767_kernel[(n_blocks,)](x, out, n, BLOCK_SIZE, num_warps=4)
    return out


def replacement_func():
    return scalar_mul_0_1767