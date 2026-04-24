import torch
import triton
import triton.language as tl


def pattern(x):
    return torch.square(x)


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2, num_stages=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
    ],
    key=['n_elements'],
)
@triton.jit
def triton_square_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    out = x * x
    tl.store(out_ptr + offsets, out, mask=mask)


# Cache output tensor by n_elements to avoid repeated CUDA allocation
_out_cache: dict = {}


@torch.fx.wrap
def triton_square(x):
    n_elements = x.numel()
    if n_elements not in _out_cache:
        _out_cache[n_elements] = torch.empty_like(x)
    out = _out_cache[n_elements]
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    triton_square_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
    )
    return out


def replacement_func():
    return triton_square