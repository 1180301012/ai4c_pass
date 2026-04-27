import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_1 = torch.sigmoid(in_0)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 64},   num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def _sigmoid_kernel(
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
    # Sigmoid: 1 / (1 + exp(-x))
    out = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def triton_sigmoid(in_0):
    n_elements = in_0.numel()
    out = torch.empty_like(in_0)

    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    _sigmoid_kernel[grid](
        x_ptr=in_0,
        out_ptr=out,
        n_elements=n_elements,
    )
    return out


def replacement_func():
    return triton_sigmoid