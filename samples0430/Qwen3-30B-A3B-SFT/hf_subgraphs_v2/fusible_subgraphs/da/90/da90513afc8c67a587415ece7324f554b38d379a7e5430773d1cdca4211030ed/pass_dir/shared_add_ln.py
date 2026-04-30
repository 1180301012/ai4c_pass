"""Shared Triton kernels and dispatch function for fused add + layer norm.

All pass files import `dispatch_fused_add_ln` from here so they share the
exact same Python function object, satisfying the replacement_func_limit.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _kern_add_ln_768(
    x1_ptr, x2_ptr, w_ptr, b_ptr, out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused add + layer-norm kernel for hidden dim = 768."""
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    base = row * N

    x1 = tl.load(x1_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    z = x1 + x2

    mean = tl.sum(z, axis=0) / N
    diff = tl.where(mask, z - mean, 0.0)
    var  = tl.sum(diff * diff, axis=0) / N
    rstd = tl.math.rsqrt(var + 1e-05)
    z_hat = diff * rstd

    w = tl.load(w_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + base + offsets, z_hat * w + b, mask=mask)


@triton.jit
def _kern_add_ln_1024(
    x1_ptr, x2_ptr, w_ptr, b_ptr, out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused add + layer-norm kernel for hidden dim = 1024."""
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    base = row * N

    x1 = tl.load(x1_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    z = x1 + x2

    mean = tl.sum(z, axis=0) / N
    diff = tl.where(mask, z - mean, 0.0)
    var  = tl.sum(diff * diff, axis=0) / N
    rstd = tl.math.rsqrt(var + 1e-05)
    z_hat = diff * rstd

    w = tl.load(w_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + base + offsets, z_hat * w + b, mask=mask)


@triton.jit
def _kern_add_ln_16(
    x1_ptr, x2_ptr, w_ptr, b_ptr, out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused add + layer-norm kernel for hidden dim = 16."""
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    base = row * N

    x1 = tl.load(x1_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(x2_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    z = x1 + x2

    mean = tl.sum(z, axis=0) / N
    diff = tl.where(mask, z - mean, 0.0)
    var  = tl.sum(diff * diff, axis=0) / N
    rstd = tl.math.rsqrt(var + 1e-05)
    z_hat = diff * rstd

    w = tl.load(w_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + base + offsets, z_hat * w + b, mask=mask)


def _run_768(bias, weight, x1, x2):
    total_rows = x1.numel() // 768
    out = torch.empty_like(x1)
    _kern_add_ln_768[(total_rows,)](
        x1, x2, weight, bias, out,
        N=768, BLOCK_SIZE=1024, num_warps=4,
    )
    return out


def _run_1024(bias, weight, x1, x2):
    total_rows = x1.numel() // 1024
    out = torch.empty_like(x1)
    _kern_add_ln_1024[(total_rows,)](
        x1, x2, weight, bias, out,
        N=1024, BLOCK_SIZE=1024, num_warps=4,
    )
    return out


def _run_16(bias, weight, x1, x2):
    total_rows = x1.numel() // 16
    out = torch.empty_like(x1)
    _kern_add_ln_16[(total_rows,)](
        x1, x2, weight, bias, out,
        N=16, BLOCK_SIZE=16, num_warps=1,
    )
    return out


@torch.fx.wrap
def dispatch_fused_add_ln(bias, weight, x1, x2, route):
    """Single dispatch function shared across all pattern passes."""
    if route == "768":
        return _run_768(bias, weight, x1, x2)
    elif route == "1024":
        return _run_1024(bias, weight, x1, x2)
    elif route == "16":
        return _run_16(bias, weight, x1, x2)
    else:
        return _run_768(bias, weight, x1, x2)