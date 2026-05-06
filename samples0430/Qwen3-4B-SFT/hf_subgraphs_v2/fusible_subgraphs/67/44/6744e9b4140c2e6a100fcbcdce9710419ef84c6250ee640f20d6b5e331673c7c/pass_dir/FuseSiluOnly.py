import torch
import triton
import triton.language as tl


def pattern(x):
    tmp_0 = torch.nn.functional.silu(x, inplace=True)
    return tmp_0


def replacement_args(x):
    return (x,)


@triton.jit
def _silu_kernel(
    x_ptr,
    out_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x * tl.sigmoid(x)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def _silu_wrapper(x):
    N = x.numel()
    out = torch.empty_like(x)
    BLOCK = 1024
    grid = ((N + BLOCK - 1) // BLOCK,)
    _silu_kernel[grid](x_ptr=x, out_ptr=out, N=N, BLOCK=BLOCK)
    return out


def replacement_func():
    return _silu_wrapper