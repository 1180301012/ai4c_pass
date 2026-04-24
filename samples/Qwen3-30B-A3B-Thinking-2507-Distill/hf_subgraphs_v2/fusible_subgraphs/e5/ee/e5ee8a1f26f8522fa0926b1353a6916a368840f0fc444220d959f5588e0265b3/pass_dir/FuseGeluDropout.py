import torch
import triton
import triton.language as tl


@triton.jit
def _gelu_fwd_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute GELU using exact erf: 0.5 * x * (1 + erf(x / sqrt(2)))
    x_f32 = x.to(tl.float32)
    gelu_f32 = 0.5 * x_f32 * (1.0 + tl.math.erf(x_f32 * 0.7071067811865476))
    gelu = gelu_f32.to(x.dtype)

    tl.store(out_ptr + offsets, gelu, mask=mask)


@torch.fx.wrap
def triton_gelu_dropout(x):
    n = x.numel()
    out = torch.empty_like(x)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    _gelu_fwd_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)
    return out


def pattern(x):
    tmp = torch.nn.functional.gelu(x)
    out = torch.nn.functional.dropout(tmp, 0.0, False, False)
    return out


def replacement_args(x):
    return (x,)


def replacement_func():
    return triton_gelu_dropout