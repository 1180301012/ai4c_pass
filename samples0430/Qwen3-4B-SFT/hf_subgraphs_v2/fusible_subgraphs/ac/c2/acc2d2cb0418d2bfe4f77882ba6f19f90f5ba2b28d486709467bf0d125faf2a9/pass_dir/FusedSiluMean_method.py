"""
AI4C Pass: Fuse SiLU (inplace method) + spatial mean into one Triton kernel.
Tests if the model's FX graph uses call_method('silu_', ...) for the inplace silu.
If this matches, we get a 2-op fused kernel for better speedup.
"""

import torch
import triton
import triton.language as tl


# Try torch.ops.aten.silu.default (non-inplace aten form)
# The model stores call_function; this is the aten-level op torchdynamo may emit.
def pattern(x):
    silu_out = torch.ops.aten.silu.default(x)
    mean_out = silu_out.mean((2, 3), keepdim=True)
    return silu_out, mean_out


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8),
    ],
    key=['HW', 'C'],
)
@triton.jit
def _silu_mean_kernel2(
    x_ptr,
    out_ptr,
    mean_ptr,
    HW, C,
    BLOCK_HW: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)
    c = pid_bc % C
    hw_off = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask   = hw_off < HW
    base   = pid_bc * HW
    xv     = tl.load(x_ptr + base + hw_off, mask=mask, other=0.0).to(tl.float32)
    s      = xv * tl.sigmoid(xv)
    tl.store(out_ptr + base + hw_off, s.to(xv.dtype), mask=mask)
    tl.atomic_add(mean_ptr + pid_bc, tl.sum(tl.where(mask, s, 0.0), axis=0))


@triton.jit
def _cast_mean_kernel2(
    src_ptr,
    dst_ptr,
    n,
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(dst_ptr + offs,
             tl.load(src_ptr + offs, mask=mask, other=0.0).
             to(dst_ptr.dtype.element_ty), mask=mask)


@torch.fx.wrap
def fused_silu_mean2(x):
    B, C, H, W = x.shape
    HW   = H * W
    n_bc = B * C
    out        = torch.empty_like(x)
    mean_f32   = torch.empty((n_bc,), dtype=torch.float32, device=x.device)
    mean_out   = torch.empty((B, C, 1, 1), dtype=x.dtype, device=x.device)
    grid = lambda meta: (n_bc, triton.cdiv(HW, meta['BLOCK_HW']))
    _silu_mean_kernel2[grid](
        x_ptr=x, out_ptr=out, mean_ptr=mean_f32,
        HW=HW, C=C,
    )
    BLOCK_C = 256
    if x.dtype != torch.float32:
        _cast_mean_kernel2[(triton.cdiv(n_bc, BLOCK_C),)](
            src_ptr=mean_f32, dst_ptr=mean_out, n=n_bc, BLOCK=BLOCK_C,
        )
    return out, mean_out


def replacement_func():
    return fused_silu_mean2