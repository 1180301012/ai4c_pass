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
def gelu_tanh_kernel(
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

    # Compute GELU with tanh approximation
    # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x_fp32 = x.to(tl.float32)
    x_cubed = x_fp32 * x_fp32 * x_fp32
    inner = 0.7978845608028654 * (x_fp32 + 0.044715 * x_cubed)
    tanh_val = tl.extra.cuda.libdevice.tanh(inner)
    result = 0.5 * x_fp32 * (1.0 + tanh_val)
    result = result.to(x.dtype)

    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_gelu_tanh(x):
    n_elements = x.numel()
    out = torch.empty_like(x)

    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    gelu_tanh_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=3,
    )

    return out


def replacement_func():
    return fused_gelu_tanh