import torch
import triton
import triton.language as tl


def pattern(x):
    return x.flatten(1, -1)


def replacement_args(x):
    return (x,)


@triton.jit
def flatten_copy_kernel(
    x_ptr,
    out_ptr,
    n,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, x, mask=mask)


@torch.fx.wrap
def fused_relu_avg(x):
    out = torch.empty((x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]), device=x.device, dtype=x.dtype)
    n = out.numel()
    flatten_copy_kernel[((n + 255) // 256,)](x, out, n, BLOCK=256)
    return out


def replacement_func():
    return fused_relu_avg