"""
Simple broadcast multiply pattern: matches in_2 * in_1 where in_1 is a scale vector.
Targets rtmpose-l's elementwise scale operation.
Uses Triton for efficient broadcast multiply.
"""
import torch
import triton
import triton.language as tl


def pattern(in_1, in_2):
    return in_2 * in_1


def replacement_args(in_1, in_2):
    return (in_1, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=2),
    ],
    key=['n_elements', 'n_scale'],
)
@triton.jit
def broadcast_mul_kernel(
    x_ptr, scale_ptr, out_ptr,
    n_elements, n_scale,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute out[i] = x[i] * scale[i % n_scale]  (broadcast multiply)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + offsets % n_scale, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x * scale, mask=mask)


@torch.fx.wrap
def triton_broadcast_mul(in_1, in_2):
    """
    Compute in_2 * in_1 with Triton broadcast multiply.
    in_1: scale vector [N] (last dimension of in_2)
    in_2: input tensor [*, N]
    """
    in_2c = in_2.contiguous()
    in_1c = in_1.contiguous()
    out = torch.empty_like(in_2c)
    n_elements = in_2c.numel()
    n_scale = in_1c.numel()
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
    broadcast_mul_kernel[grid](
        in_2c, in_1c, out,
        n_elements, n_scale,
    )
    return out


def replacement_func():
    return triton_broadcast_mul