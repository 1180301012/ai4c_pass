"""
Shared Triton kernels + unified dispatch function.

Key design:
- _bn_flat_kernel / _avgpool_flat_kernel : individual ops (kept for reference)
- _combined_bn_pool_kernel               : single-launch BN + AvgPool
  First  ceil(total_bn   / BLOCK) programs → BatchNorm
  Remaining ceil(total_pool / BLOCK) programs → AvgPool2d

All integer divisions use tl.constexpr denominators (SPATIAL, SOUT, W_OUT)
so the Triton compiler substitutes fast magic-multiplies (~1 cycle).
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Parameter cache
# ---------------------------------------------------------------------------
_param_gpu_cache: dict = {}


def _to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    if t.device == device:
        return t
    key = (t.data_ptr(), str(device))
    cached = _param_gpu_cache.get(key)
    if cached is None:
        cached = t.to(device)
        _param_gpu_cache[key] = cached
    return cached


# ---------------------------------------------------------------------------
# Combined BN + AvgPool2d kernel  (ONE launch for both ops)
#
# Grid layout  (1-D, size = bn_blocks + pool_blocks):
#   pid ∈ [0,          bn_blocks)  → BatchNorm on x_bn
#   pid ∈ [bn_blocks,  total)      → AvgPool2d  on x_pool
#
# All division denominators are tl.constexpr → magic-multiply optimisation.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 4096}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK': 2048}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK': 1024}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK': 512},  num_warps=4),
    ],
    key=['total_bn', 'total_pool'],
)
@triton.jit
def _combined_bn_pool_kernel(
    # BN tensors
    x_bn_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, bn_out_ptr,
    C_bn, total_bn,
    # AvgPool tensors
    x_pool_ptr, pool_out_ptr,
    H_in, W_in, total_pool,
    # Constexpr shapes (fast division)
    SPATIAL: tl.constexpr,    # H_bn * W_bn
    H_OUT:   tl.constexpr,    # H_pool // 2
    W_OUT:   tl.constexpr,    # W_pool // 2
    SOUT:    tl.constexpr,    # H_OUT * W_OUT
    BLOCK:   tl.constexpr,
):
    pid = tl.program_id(0)
    # Ceiling-divide total_bn by BLOCK (BLOCK constexpr → fast)
    bn_blocks = (total_bn + BLOCK - 1) // BLOCK

    if pid < bn_blocks:
        # ── BatchNorm path ──────────────────────────────────────────
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total_bn

        c = (offs // SPATIAL) % C_bn          # SPATIAL constexpr → fast

        mean = tl.load(mean_ptr   + c, mask=mask).to(tl.float32)
        var  = tl.load(var_ptr    + c, mask=mask).to(tl.float32)
        w    = tl.load(weight_ptr + c, mask=mask).to(tl.float32)
        b    = tl.load(bias_ptr   + c, mask=mask).to(tl.float32)

        scale  = w * tl.rsqrt(var + 1e-5)
        offset = b - mean * scale

        x   = tl.load(x_bn_ptr + offs, mask=mask)
        out = x.to(tl.float32) * scale + offset
        tl.store(bn_out_ptr + offs, out.to(x.dtype), mask=mask)

    else:
        # ── AvgPool2d path ─────────────────────────────────────────
        adj  = pid - bn_blocks
        offs = adj * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total_pool

        nc    = offs // SOUT               # SOUT constexpr → fast
        s     = offs %  SOUT
        h_out = s    // W_OUT              # W_OUT constexpr → fast
        w_out = s    %  W_OUT
        h_in  = h_out * 2
        w_in  = w_out * 2

        base = nc * H_in * W_in
        v00 = tl.load(x_pool_ptr + base + h_in       * W_in + w_in,     mask=mask, other=0.0)
        v01 = tl.load(x_pool_ptr + base + h_in       * W_in + w_in + 1, mask=mask, other=0.0)
        v10 = tl.load(x_pool_ptr + base + (h_in + 1) * W_in + w_in,     mask=mask, other=0.0)
        v11 = tl.load(x_pool_ptr + base + (h_in + 1) * W_in + w_in + 1, mask=mask, other=0.0)

        avg = (v00.to(tl.float32) + v01.to(tl.float32) +
               v10.to(tl.float32) + v11.to(tl.float32)) * 0.25
        tl.store(pool_out_ptr + offs, avg.to(v00.dtype), mask=mask)


def _do_combined_bn_pool(x_bn, mean, var, weight, bias, x_pool):
    """Run BN + AvgPool in a single Triton kernel launch."""
    device = x_bn.device
    mean   = _to_device(mean,   device)
    var    = _to_device(var,    device)
    weight = _to_device(weight, device)
    bias   = _to_device(bias,   device)

    N_bn, C_bn, H_bn, W_bn = x_bn.shape
    SPATIAL  = H_bn * W_bn
    total_bn = N_bn * C_bn * SPATIAL
    bn_out   = torch.empty_like(x_bn)

    N_p, C_p, H_p, W_p = x_pool.shape
    H_OUT      = H_p // 2
    W_OUT      = W_p // 2
    SOUT       = H_OUT * W_OUT
    total_pool = N_p * C_p * SOUT
    pool_out   = torch.empty((N_p, C_p, H_OUT, W_OUT), dtype=x_pool.dtype, device=device)

    grid = lambda meta: (
        triton.cdiv(total_bn,   meta['BLOCK']) +
        triton.cdiv(total_pool, meta['BLOCK']),
    )
    _combined_bn_pool_kernel[grid](
        x_bn, mean, var, weight, bias, bn_out,
        C_bn=C_bn, total_bn=total_bn,
        x_pool_ptr=x_pool, pool_out_ptr=pool_out,
        H_in=H_p, W_in=W_p, total_pool=total_pool,
        SPATIAL=SPATIAL, H_OUT=H_OUT, W_OUT=W_OUT, SOUT=SOUT,
    )
    return pool_out, bn_out


# ---------------------------------------------------------------------------
# Individual BN / AvgPool kernels (used by fallback passes)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 2048}, num_warps=8),
        triton.Config({'BLOCK': 1024}, num_warps=8),
        triton.Config({'BLOCK': 512},  num_warps=4),
    ],
    key=['total_bn'],
)
@triton.jit
def _bn_flat_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    C, total_bn,
    SPATIAL: tl.constexpr,
    BLOCK:   tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_bn
    c = (offs // SPATIAL) % C
    mean = tl.load(mean_ptr   + c, mask=mask).to(tl.float32)
    var  = tl.load(var_ptr    + c, mask=mask).to(tl.float32)
    w    = tl.load(weight_ptr + c, mask=mask).to(tl.float32)
    b    = tl.load(bias_ptr   + c, mask=mask).to(tl.float32)
    scale  = w * tl.rsqrt(var + 1e-5)
    offset = b - mean * scale
    x   = tl.load(x_ptr + offs, mask=mask)
    out = x.to(tl.float32) * scale + offset
    tl.store(out_ptr + offs, out.to(x.dtype), mask=mask)


def _do_bn(x, mean, var, weight, bias):
    device = x.device
    mean   = _to_device(mean,   device)
    var    = _to_device(var,    device)
    weight = _to_device(weight, device)
    bias   = _to_device(bias,   device)
    N, C, H, W = x.shape
    spatial  = H * W
    total_bn = N * C * spatial
    out      = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(total_bn, meta['BLOCK']),)
    _bn_flat_kernel[grid](x, mean, var, weight, bias, out,
                          C=C, total_bn=total_bn, SPATIAL=spatial)
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=8),
        triton.Config({'BLOCK': 512},  num_warps=4),
        triton.Config({'BLOCK': 256},  num_warps=4),
    ],
    key=['total_pool'],
)
@triton.jit
def _avgpool_flat_kernel(
    x_ptr, out_ptr,
    H_in, W_in, total_pool,
    H_OUT: tl.constexpr,
    W_OUT: tl.constexpr,
    SOUT:  tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total_pool
    nc    = offs // SOUT
    s     = offs %  SOUT
    h_out = s    // W_OUT
    w_out = s    %  W_OUT
    h_in  = h_out * 2
    w_in  = w_out * 2
    base = nc * H_in * W_in
    v00 = tl.load(x_ptr + base + h_in       * W_in + w_in,     mask=mask, other=0.0)
    v01 = tl.load(x_ptr + base + h_in       * W_in + w_in + 1, mask=mask, other=0.0)
    v10 = tl.load(x_ptr + base + (h_in + 1) * W_in + w_in,     mask=mask, other=0.0)
    v11 = tl.load(x_ptr + base + (h_in + 1) * W_in + w_in + 1, mask=mask, other=0.0)
    avg = (v00.to(tl.float32) + v01.to(tl.float32) +
           v10.to(tl.float32) + v11.to(tl.float32)) * 0.25
    tl.store(out_ptr + offs, avg.to(v00.dtype), mask=mask)


def _do_avgpool(x):
    N, C, H, W = x.shape
    H_out, W_out = H // 2, W // 2
    sout       = H_out * W_out
    total_pool = N * C * sout
    out = torch.empty((N, C, H_out, W_out), dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(total_pool, meta['BLOCK']),)
    _avgpool_flat_kernel[grid](x, out, H_in=H, W_in=W, total_pool=total_pool,
                               H_OUT=H_out, W_OUT=W_out, SOUT=sout)
    return out


# ---------------------------------------------------------------------------
# Unified dispatch for fallback passes (BN-only / AvgPool-only)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_dispatch(route, a, b=None, c=None, d=None, e=None):
    if route == "bn":
        return _do_bn(a, b, c, d, e)
    return _do_avgpool(a)