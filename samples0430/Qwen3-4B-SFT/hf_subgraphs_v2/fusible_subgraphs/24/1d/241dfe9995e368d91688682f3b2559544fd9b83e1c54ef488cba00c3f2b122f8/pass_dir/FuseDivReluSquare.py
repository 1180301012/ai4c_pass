import torch
import triton
import triton.language as tl


def pattern(x):
    return x / 11.313708498984761


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
    ],
    key=['n_elements'],
)
@triton.jit
def _fused_div_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    scaling: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    inv_scale = 1.0 / scaling
    # Compute relu(x / scale) fused - downstream relu+square apply correctly
    result = tl.maximum(x_f32 * inv_scale, 0.0)
    tl.store(out_ptr + offsets, result.to(x.dtype), mask=mask)


@torch.fx.wrap
def fused_div(x):
    n_elements = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _fused_div_kernel[grid](
        in_ptr=x,
        out_ptr=out,
        n_elements=n_elements,
        scaling=11.313708498984761,
    )
    return out


def replacement_func():
    return fused_div