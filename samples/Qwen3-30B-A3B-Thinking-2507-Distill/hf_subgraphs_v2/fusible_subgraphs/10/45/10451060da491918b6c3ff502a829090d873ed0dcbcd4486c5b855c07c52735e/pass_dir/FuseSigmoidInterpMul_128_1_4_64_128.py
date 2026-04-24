import torch
import triton
import triton.language as tl


def pattern(x, y):
    """Match the final element-wise multiply."""
    return x * y


def replacement_args(x, y):
    return (x, y)


@triton.jit
def _mul_kernel(
    x_ptr, y_ptr, out_ptr,
    N_ELEMS,
    BLOCK: tl.constexpr,
):
    pid     = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask    = offsets < N_ELEMS
    x   = tl.load(x_ptr  + offsets, mask=mask, other=0.0)
    y   = tl.load(y_ptr  + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x * y, mask=mask)


@torch.fx.wrap
def triton_mul(x, y):
    n   = x.numel()
    out = torch.empty_like(x)
    BLOCK = 1024
    grid  = ((n + BLOCK - 1) // BLOCK,)
    _mul_kernel[grid](x, y, out, n, BLOCK=BLOCK, num_warps=4, num_stages=1)
    return out


def replacement_func():
    return triton_mul