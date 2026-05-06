import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < n
    x = tl.load(x_ptr + i, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + i, tl.sigmoid(x).to(out_ptr.dtype.element_ty), mask=mask)


@torch.fx.wrap
def fused_test_sigmoid(x):
    out = torch.empty_like(x)
    n = x.numel()
    grid = (n + 1023) // 1024
    _sigmoid_kernel[(grid,)](x, out, n, BLOCK=1024)
    return out


def pattern(x):
    return torch.sigmoid(x)


def replacement_args(x):
    return (x,)


def replacement_func():
    return fused_test_sigmoid