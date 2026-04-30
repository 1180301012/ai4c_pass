import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['n_elements'],
)
@triton.jit
def _fused_div_relu_square_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # divide by 11.313708498984761
    x = x / 11.313708498984761
    # relu
    x = tl.where(x > 0.0, x, 0.0)
    # square
    x = x * x

    tl.store(out_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def fused_div_relu_square(x):
    n_elements = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    _fused_div_relu_square_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
    )
    return out


def pattern(in_0):
    tmp_0 = in_0 / 11.313708498984761
    tmp_1 = torch.nn.functional.relu(tmp_0)
    tmp_2 = torch.square(tmp_1)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return fused_div_relu_square