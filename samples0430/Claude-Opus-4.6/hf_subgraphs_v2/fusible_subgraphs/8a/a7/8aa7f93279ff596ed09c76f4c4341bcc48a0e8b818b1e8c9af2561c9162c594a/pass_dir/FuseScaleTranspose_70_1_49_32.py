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
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask)
    out = x * 0.1767766952966369
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def triton_scale(x):
    out = torch.empty_like(x)
    scale_kernel[(14,)](x, out, 109760, BLOCK_SIZE=8192, num_warps=4)
    return out


def replacement_func():
    return triton_scale