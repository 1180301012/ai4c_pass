import torch
import triton
import triton.language as tl


def pattern(g, x, denom):
    tmp_6 = x / denom
    tmp_7 = tmp_6 * g
    return tmp_7


def replacement_args(g, x, denom):
    return (x, denom, g)


@triton.jit
def _div_mul_kernel(
    x_ptr,
    denom_ptr,
    g_ptr,
    out_ptr,
    n_elements,
    K,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    row = offs // K
    denom = tl.load(denom_ptr + row, mask=mask, other=1.0)
    g = tl.load(g_ptr)

    # Preserve eager-like low-precision behavior for fp16/bf16 by rounding after div
    # before the subsequent multiply.
    tmp = (x / denom).to(x.dtype)
    y = (tmp * g).to(x.dtype)
    tl.store(out_ptr + offs, y, mask=mask)


@torch.fx.wrap
def fused_truediv_mul_scalar(x, denom, g):
    # x: [N, C, K] contiguous from flatten
    # denom: [N, C, 1] contiguous from clamp
    # g: [1]
    K = x.shape[2]
    n_elements = x.numel()

    out = torch.empty_like(x)
    block = 1024
    grid = ((n_elements + block - 1) // block,)
    _div_mul_kernel[grid](
        x,
        denom,
        g,
        out,
        n_elements,
        K,
        BLOCK_SIZE=block,
    )
    return out


def replacement_func():
    return fused_truediv_mul_scalar