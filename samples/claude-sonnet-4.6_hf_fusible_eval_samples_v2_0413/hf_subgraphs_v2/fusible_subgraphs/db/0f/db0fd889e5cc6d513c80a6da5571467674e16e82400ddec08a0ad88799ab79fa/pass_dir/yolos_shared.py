"""
Shared Triton kernels and dispatch wrapper for YOLOS optimizations.
Imported by all individual pass files.

Routes:
  "interp_same_size" : a = input 4D tensor [B, 32, 15, 15], b = a (ignored)
                       bicubic interpolate with same input/output size is identity;
                       replace with a fast Triton contiguous-copy kernel.
  "add_dropout"      : a = x, b = y
                       dropout with training=False is identity; fuse add + dropout
                       into a single Triton add kernel.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: Copy a (possibly non-contiguous) 4D [B, 32, 15, 15] tensor to
#            a fresh contiguous buffer of the same shape.
# ---------------------------------------------------------------------------
@triton.jit
def _copy_4d_interp_kernel(
    src_ptr, dst_ptr,
    s0, s1, s2, s3,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    D3: tl.constexpr = 15
    D2: tl.constexpr = 15
    D1: tl.constexpr = 32

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    w = offsets % D3
    h = (offsets // D3) % D2
    c = (offsets // (D2 * D3)) % D1
    b = offsets // (D1 * D2 * D3)

    src_off = b * s0 + c * s1 + h * s2 + w * s3
    val = tl.load(src_ptr + src_off, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, val, mask=mask)


def _run_interp_identity(x):
    B = x.shape[0]
    total = B * 32 * 15 * 15
    out = torch.empty(B, 32, 15, 15, dtype=x.dtype, device=x.device)
    s = x.stride()
    BLOCK_SIZE = 256
    grid = ((total + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _copy_4d_interp_kernel[grid](x, out, s[0], s[1], s[2], s[3], total, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ---------------------------------------------------------------------------
# Kernel 2: Element-wise add — replaces add + eval-dropout (dropout no-op)
# ---------------------------------------------------------------------------
@triton.jit
def _add_kernel(
    x_ptr, y_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def _run_add(x, y):
    out = torch.empty_like(x)
    N = x.numel()
    BLOCK_SIZE = 4096
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _add_kernel[grid](x, y, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ---------------------------------------------------------------------------
# Shared dispatch wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def yolos_dispatch(a, b, route):
    if route == "interp_same_size":
        return _run_interp_identity(a)
    else:
        # "add_dropout": Triton add (dropout with training=False is identity)
        return _run_add(a, b)