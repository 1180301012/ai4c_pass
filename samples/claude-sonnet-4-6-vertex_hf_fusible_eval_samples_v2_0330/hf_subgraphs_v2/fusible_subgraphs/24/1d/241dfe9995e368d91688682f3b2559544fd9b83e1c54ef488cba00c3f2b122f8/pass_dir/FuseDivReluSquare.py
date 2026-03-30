import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_2 = torch.square(in_0)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_div_relu_square_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)

    # Just square: relu(x)*relu(x) = x*x for x>=0 (relu's output is always >=0)
    x = x * x

    tl.store(out_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def fused_div_relu_square(in_0):
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)

    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    fused_div_relu_square_kernel[grid](
        in_0, out, n_elements,
    )

    return out


def replacement_func():
    return fused_div_relu_square