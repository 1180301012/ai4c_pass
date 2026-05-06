import torch
import triton
import triton.language as tl


def pattern(in_0):
    return torch.nn.functional.gelu(in_0)


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def gelu_flat_kernel(
    input_ptr,
    output_ptr,
    N,
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    x   = tl.load(input_ptr + offs, mask=mask, other=0.0)
    xf  = x.to(tl.float32)
    gelu = 0.5 * xf * (1.0 + tl.math.erf(xf * 0.7071067811865476))

    tl.store(output_ptr + offs, gelu.to(x.dtype), mask=mask)


@torch.fx.wrap
def triton_gelu(x):
    nElems = x.numel()
    BLOCK  = 2048
    out    = torch.empty_like(x)
    gelu_flat_kernel[((nElems + BLOCK - 1) // BLOCK,)](
        x, out, nElems,
        BLOCK=BLOCK,
        num_warps=8,
    )
    return out


def replacement_func():
    return triton_gelu