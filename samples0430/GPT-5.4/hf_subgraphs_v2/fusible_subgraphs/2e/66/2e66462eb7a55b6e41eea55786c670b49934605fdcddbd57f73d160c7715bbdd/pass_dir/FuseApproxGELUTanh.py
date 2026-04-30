import torch
import triton
import triton.language as tl


# Pattern matching function
# Matches: 0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x^3)))
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


# Argument extraction function
def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_approx_gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    offsets = tl.max_contiguous(tl.multiple_of(offsets, BLOCK_SIZE), BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    x2 = x_f32 * x_f32
    x3 = x2 * x_f32
    inner = 0.7978845608028654 * (x_f32 + 0.044715 * x3)

    # tanh(z) = 2 / (1 + exp(-2z)) - 1
    tanh_inner = 2.0 / (1.0 + tl.exp(-2.0 * inner)) - 1.0
    out = 0.5 * x_f32 * (1.0 + tanh_inner)

    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_approx_gelu(in_0):
    out = torch.empty_like(in_0)
    n_elements = in_0.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    fused_approx_gelu_kernel[grid](
        in_0,
        out,
        n_elements,
        BLOCK_SIZE=2048,
        num_warps=8,
        num_stages=2,
    )
    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_approx_gelu