import torch
import triton
import triton.language as tl


def pattern(x):
    tmp_0 = x / 11.313708498984761
    tmp_1 = torch.nn.functional.relu(tmp_0)
    tmp_2 = tmp_1 * tmp_1
    return tmp_2


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_div_relu_square_kernel_v2(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = x * 0.08838834764831843
    x = tl.maximum(x, 0.0)
    x = x * x
    tl.store(out_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def fused_div_relu_square_v2(x):
    n_elements = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    fused_div_relu_square_kernel_v2[grid](x, out, n_elements)
    return out


def replacement_func():
    return fused_div_relu_square_v2