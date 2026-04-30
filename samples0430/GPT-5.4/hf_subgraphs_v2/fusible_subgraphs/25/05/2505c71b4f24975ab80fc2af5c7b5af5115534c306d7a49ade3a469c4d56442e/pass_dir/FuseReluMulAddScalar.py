import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, relu_out):
    tmp_3 = in_1 * relu_out
    tmp_4 = tmp_3 + in_0
    return tmp_4


def replacement_args(in_0, in_1, relu_out):
    return (in_0, in_1, relu_out)




@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit

def fused_relu_mul_add_scalar_kernel(
    bias_ptr,
    scale_ptr,
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
    zero = tl.zeros([BLOCK_SIZE], dtype=x.dtype)
    x = tl.maximum(x, zero)
    scale = tl.load(scale_ptr)
    bias = tl.load(bias_ptr)
    out = x * scale + bias
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_relu_mul_add_scalar(in_0, in_1, in_2):
    out = torch.empty_like(in_2)
    n_elements = in_2.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_relu_mul_add_scalar_kernel[grid](
        in_0,
        in_1,
        in_2,
        out,
        n_elements,
    )
    return out


def replacement_func():
    return fused_relu_mul_add_scalar