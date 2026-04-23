import torch
import triton
import triton.language as tl


def pattern(x):
    tmp_5 = torch.nn.functional.gelu(x, approximate='none')
    return tmp_5


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _gelu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865475))
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_add_gelu(x):
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _gelu_kernel[grid](x, out, n_elements)
    return out


def replacement_func():
    return fused_add_gelu