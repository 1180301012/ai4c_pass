"""
Shared Triton kernel + dispatch wrapper for fused Add + LayerNorm.

Design:
  One kernel function shared by all routes. Kernel is compiled fresh for
every unique (BLOCK_SIZE, DTYPE) pair chosen at runtime.

  BLOCK_SIZE=N:  exact-match tile.  No OOB loads. No masking overhead.
  For N=768/16/1024 we choose BLOCK_SIZE=N so the column-vector from
  tl.arange(0, N_CONST) is always fully in-bounds.

  Loops:  Phase 1 (mean via E[x]), Phase 2 (var via E[x²]-E[x]²),
                    Phase 3 (normalise + store).
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Shared dispatch wrapper  (@torch.fx.wrap → opaque to FX tracing)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def dispatch_fused_add_layernorm(in_0, in_1, in_2, in_3, route):
    """
    in_0 : LN bias   [N]
    in_1 : LN weight [N]
    in_2, in_3 : hidden states  [..., N]
    route : int – selects which kernel variant (N)
            16  → N=16
            768 → N=768
            1024→ N=1024
    """
    dtype = in_2.dtype
    if dtype == torch.bfloat16:
        d = 0
    elif dtype == torch.float16:
        d = 1
    else:
        d = 2

    if route == 16:
        N, NBS = 16, 16
    elif route == 768:
        N, NBS = 768, 1024
    else:   # 1024
        N, NBS = 1024, 1024

    num_rows   = in_2.numel() // N
    out        = torch.empty_like(in_2)
    NUM_BLOCKS = 1

    if route == 16:
        _kernel_16[(num_rows, NUM_BLOCKS)](
            in_2, in_3, in_1, in_0, out, N, 1e-5, DTYPE=d, NBS=16,    num_warps=1)
    elif route == 768:
        _kernel_768[(num_rows, NUM_BLOCKS)](
            in_2, in_3, in_1, in_0, out, N, 1e-5, DTYPE=d, NBS=1024,  num_warps=4)
    else:
        _kernel_1024[(num_rows, NUM_BLOCKS)](
            in_2, in_3, in_1, in_0, out, N, 1e-5, DTYPE=d, NBS=1024,  num_warps=4)
    return out


# ---------------------------------------------------------------------------
# Three specialised kernels, chosen from the dispatch at JIT time:
#  _kernel_16    : N=16,   BLOCK_SIZE=16   (exact, no masking)
#  _kernel_768   : N=768,  BLOCK_SIZE=1024 (masked)
#  _kernel_1024  : N=1024, BLOCK_SIZE=1024 (exact, no masking)
# Each has its own @triton.jit symbol; no cross-constexpr pollution.
# ---------------------------------------------------------------------------

@triton.jit
def _kernel_16(
    x_ptr, y_ptr, w_ptr, b_ptr, out_ptr,
    N,
    eps,
    DTYPE: tl.constexpr,
    NBS: tl.constexpr,     # == N == 16 → tl.arange(0,16) works
):
    row = tl.program_id(0)
    xpb = x_ptr + row * N;  ypb = y_ptr + row * N;  opb = out_ptr + row * N
    cols = tl.arange(0, NBS)
    nm   = cols < N
    xv = tl.load(xpb + cols, mask=nm, other=0.0).to(tl.float32)
    yv = tl.load(ypb + cols, mask=nm, other=0.0).to(tl.float32)
    _s = tl.sum(xv + yv, axis=0);  mean = _s / N
    _ss = tl.sum((xv + yv) * (xv + yv), axis=0)
    var = _ss / N - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    wv = tl.load(w_ptr + cols, mask=nm, other=1.0).to(tl.float32)
    bv = tl.load(b_ptr + cols, mask=nm, other=0.0).to(tl.float32)
    res = (xv + yv - mean) * inv_std * wv + bv
    d32 = tl.full([NBS], 0.0, tl.float32)
    if DTYPE == 0:
        tl.store(opb + cols, tl.where(nm, res.to(tl.bfloat16), tl.full([NBS], 0.0, tl.float16)), mask=nm)
    elif DTYPE == 1:
        tl.store(opb + cols, tl.where(nm, res.to(tl.float16), tl.full([NBS], 0.0, tl.float16)), mask=nm)
    else:
        tl.store(opb + cols, tl.where(nm, res, d32), mask=nm)


@triton.jit
def _kernel_768(
    x_ptr, y_ptr, w_ptr, b_ptr, out_ptr,
    N,
    eps,
    DTYPE: tl.constexpr,
    NBS: tl.constexpr,     # = 1024 → tl.arange(0,1024) works
):
    row  = tl.program_id(0)
    xpb  = x_ptr   + row * N
    ypb  = y_ptr   + row * N
    opb  = out_ptr + row * N
    cols = NBS + tl.arange(0, NBS)   # start at 1024
    nm   = cols < N
    xv = tl.load(xpb + cols, mask=nm, other=0.0).to(tl.float32)
    yv = tl.load(ypb + cols, mask=nm, other=0.0).to(tl.float32)
    _s = tl.sum(xv + yv, axis=0);  mean = _s / N
    _ss = tl.sum((xv + yv) * (xv + yv), axis=0)
    var = _ss / N - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    wv = tl.load(w_ptr + cols, mask=nm, other=1.0).to(tl.float32)
    bv = tl.load(b_ptr + cols, mask=nm, other=0.0).to(tl.float32)
    res = (xv + yv - mean) * inv_std * wv + bv
    d32 = tl.full([NBS], 0.0, tl.float32)
    if DTYPE == 0:
        tl.store(opb + cols, tl.where(nm, res.to(tl.bfloat16), tl.full([NBS], 0.0, tl.float16)), mask=nm)
    elif DTYPE == 1:
        tl.store(opb + cols, tl.where(nm, res.to(tl.float16), tl.full([NBS], 0.0, tl.float16)), mask=nm)
    else:
        tl.store(opb + cols, tl.where(nm, res, d32), mask=nm)


@triton.jit
def _kernel_1024(
    x_ptr, y_ptr, w_ptr, b_ptr, out_ptr,
    N,
    eps,
    DTYPE: tl.constexpr,
    NBS: tl.constexpr,     # == N == 1024 → tl.arange(0,1024) works
):
    row  = tl.program_id(0)
    xpb  = x_ptr   + row * N
    ypb  = y_ptr   + row * N
    opb  = out_ptr + row * N
    cols = tl.arange(0, NBS)
    nm   = cols < N   # always True (N==NBS)==1024
    xv = tl.load(xpb + cols).to(tl.float32)
    yv = tl.load(ypb + cols).to(tl.float32)
    _s = tl.sum(xv + yv, axis=0);  mean = _s / N
    _ss = tl.sum((xv + yv) * (xv + yv), axis=0)
    var = _ss / N - mean * mean
    inv_std = 1.0 / tl.sqrt(var + eps)
    wv = tl.load(w_ptr + cols).to(tl.float32)
    bv = tl.load(b_ptr + cols).to(tl.float32)
    res = (xv + yv - mean) * inv_std * wv + bv
    if DTYPE == 0:
        tl.store(opb + cols, res.to(tl.bfloat16), mask=nm)
    elif DTYPE == 1:
        tl.store(opb + cols, res.to(tl.float16),  mask=nm)
    else:
        tl.store(opb + cols, res,                  mask=nm)