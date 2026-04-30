import operator
import torch
import triton
import triton.language as tl


@triton.jit
def _relu_kernel(
    in_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_relu(x):
    N = x.numel()
    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    out = torch.empty_like(x)
    _relu_kernel[grid](x, out, n_elements=N, BLOCK_SIZE=BLOCK_SIZE)
    return out


def pattern(x):
    return torch.ops.aten.relu.default(x)


def replacement_args(x):
    return (x,)


def replacement_func():
    return fused_relu