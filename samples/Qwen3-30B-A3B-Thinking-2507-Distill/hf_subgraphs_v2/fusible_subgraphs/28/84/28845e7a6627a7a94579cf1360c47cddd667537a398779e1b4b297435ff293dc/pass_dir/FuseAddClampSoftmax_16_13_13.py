import torch
import triton
import triton.language as tl


@triton.jit
def _triton_add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


@torch.fx.wrap
def _triton_add(x, y):
    N = x.numel()
    out = torch.empty_like(x)
    _triton_add_kernel[(N // 1024 + 1,)](x, y, out, N, BLOCK_SIZE=1024)
    return out


def pattern(x, y):
    return x + y


def replacement_args(x, y):
    return (x, y)


def replacement_func():
    return _triton_add