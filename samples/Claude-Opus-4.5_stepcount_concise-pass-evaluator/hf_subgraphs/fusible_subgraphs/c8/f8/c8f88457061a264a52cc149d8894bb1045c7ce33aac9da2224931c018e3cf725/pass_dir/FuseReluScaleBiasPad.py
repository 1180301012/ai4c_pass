import torch
import triton
import triton.language as tl


def pattern(bias, scale, x):
    tmp_3 = scale * x
    tmp_4 = tmp_3 + bias
    return tmp_4


def replacement_args(bias, scale, x):
    return (bias, scale, x)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_scale_bias_kernel(
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
    
    # Load scalars once
    bias = tl.load(bias_ptr)
    scale = tl.load(scale_ptr)
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # scale + bias (fused multiply-add)
    out = x * scale + bias
    
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_scale_bias(bias, scale, x):
    out = torch.empty_like(x)
    n_elements = x.numel()
    
    def grid(meta):
        return ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    fused_scale_bias_kernel[grid](bias, scale, x, out, n_elements)
    
    return out


def replacement_func():
    return fused_scale_bias