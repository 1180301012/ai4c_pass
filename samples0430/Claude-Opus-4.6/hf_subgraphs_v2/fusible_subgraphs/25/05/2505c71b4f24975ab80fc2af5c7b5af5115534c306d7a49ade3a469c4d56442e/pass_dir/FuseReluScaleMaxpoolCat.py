import torch
import triton
import triton.language as tl
import operator


def pattern(in_0, in_1, tmp_2):
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    return tmp_4


def replacement_args(in_0, in_1, tmp_2):
    return (in_0, in_1, tmp_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_mul_add_kernel(
    in_ptr, scale_ptr, bias_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    scale = tl.load(scale_ptr)
    bias = tl.load(bias_ptr)

    x = tl.load(in_ptr + offsets, mask=mask)
    x = x * scale + bias

    tl.store(out_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def fused_mul_add(in_0, in_1, tmp_2):
    n_elements = tmp_2.numel()
    out = torch.empty_like(tmp_2)

    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    fused_mul_add_kernel[grid](
        tmp_2, in_1, in_0, out,
        n_elements,
    )

    return out


def replacement_func():
    return fused_mul_add