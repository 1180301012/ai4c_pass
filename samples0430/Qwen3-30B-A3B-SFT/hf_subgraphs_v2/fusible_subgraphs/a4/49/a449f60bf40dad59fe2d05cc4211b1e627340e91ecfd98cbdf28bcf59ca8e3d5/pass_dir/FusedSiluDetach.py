import torch
import triton
import triton.language as tl


def pattern(x):
    return x.detach()


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # SiLU: x * sigmoid(x)
    sigmoid_x = tl.sigmoid(x)
    out = x * sigmoid_x

    tl.store(out_ptr + offsets, out.to(x.dtype), mask=mask)


@torch.fx.wrap
def triton_silu(x):
    n_elements = x.numel()
    out = torch.empty_like(x)

    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    _silu_kernel[grid](
        x, out, n_elements
    )

    return out


def replacement_func():
    return triton_silu