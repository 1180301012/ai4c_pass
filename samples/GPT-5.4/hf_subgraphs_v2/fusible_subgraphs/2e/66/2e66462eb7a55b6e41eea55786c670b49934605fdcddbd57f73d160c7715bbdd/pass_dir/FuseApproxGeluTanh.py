import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = 0.5 * in_0
    tmp_1 = torch.pow(in_0, 3.0)
    tmp_2 = 0.044715 * tmp_1
    tmp_3 = in_0 + tmp_2
    tmp_4 = 0.7978845608028654 * tmp_3
    tmp_5 = torch.tanh(tmp_4)
    tmp_6 = 1.0 + tmp_5
    tmp_7 = tmp_0 * tmp_6
    return tmp_7


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_approx_gelu_tanh_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = x * x
    x3 = x2 * x
    inner = 0.7978845608028654 * (x + 0.044715 * x3)

    abs_inner = tl.abs(inner)
    e = tl.exp(-2.0 * abs_inner)
    tanh_abs = (1.0 - e) / (1.0 + e)
    tanh_inner = tl.where(inner >= 0.0, tanh_abs, -tanh_abs)

    out = 0.5 * x * (1.0 + tanh_inner)
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_approx_gelu_tanh(in_0):
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)
    block_size = 2048
    grid = (triton.cdiv(n_elements, block_size),)
    fused_approx_gelu_tanh_kernel[grid](
        in_0,
        out,
        n_elements,
        BLOCK_SIZE=block_size,
        num_warps=8,
        num_stages=2,
    )
    return out



def replacement_func():
    return fused_approx_gelu_tanh