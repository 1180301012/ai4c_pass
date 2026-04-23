import torch
import triton
import triton.language as tl


def pattern(in_0, x):
    tmp_5 = in_0.exp()
    tmp_6 = tmp_5 * x
    return tmp_6


def replacement_args(in_0, x):
    return (in_0, x)


@triton.jit
def _exp_mul_512_kernel(
    scalar_ptr,
    x_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    scale = tl.exp(tl.load(scalar_ptr).to(tl.float32))
    x = tl.load(x_ptr + offs).to(tl.float32)
    y = x * scale
    tl.store(out_ptr + offs, y)


@torch.fx.wrap
def fused_exp_mul_512(in_0, x):
    out = torch.empty_like(x)
    _exp_mul_512_kernel[(1,)](
        in_0,
        x,
        out,
        BLOCK=512,
        num_warps=1,
    )
    return out


def replacement_func():
    return fused_exp_mul_512