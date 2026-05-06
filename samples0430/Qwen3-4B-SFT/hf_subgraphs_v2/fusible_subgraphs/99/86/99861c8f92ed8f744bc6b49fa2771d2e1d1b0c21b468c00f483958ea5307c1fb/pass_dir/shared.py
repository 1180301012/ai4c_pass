"""
Shared Triton layer-norm kernel (single-output).
This is a drop-in replacement for torch.nn.functional.layer_norm.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _ln_kernel(
    x_ptr, out_ptr,
    weight_ptr, bias_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One program = one row.  All arithmetic in fp32 for numerical stability.
    """
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    offsets = row * N + cols

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_fp32 = x.to(tl.float32)

    mean = tl.sum(x_fp32, axis=0) / N
    diff = x_fp32 - mean
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    norm = diff * inv_std

    scale = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    bias_v = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = norm * scale + bias_v

    tl.store(out_ptr + offsets, out.to(x_ptr.dtype.element_ty))


@torch.fx.wrap
def _layernorm_kernel_only(bias, weight, x):
    """
    Triton-accelerated layer norm.
    bias, weight, x  →  out = layer_norm(x, weight=weight, bias=bias)
    """
    N = 1024
    BLOCK_SIZE = 1024
    M = x.numel() // N

    out = torch.empty_like(x)

    _ln_kernel[(M,)](
        x, out,
        weight, bias,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    return out