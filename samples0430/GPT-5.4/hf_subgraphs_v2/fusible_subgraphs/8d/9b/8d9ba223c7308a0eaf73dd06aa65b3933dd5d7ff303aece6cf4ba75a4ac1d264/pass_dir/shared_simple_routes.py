import torch
import triton
import triton.language as tl


@triton.jit
def _unused_identity_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(y_ptr + offs, x, mask=mask)


@torch.fx.wrap
def shared_simple_route(*args):
    return args[0]