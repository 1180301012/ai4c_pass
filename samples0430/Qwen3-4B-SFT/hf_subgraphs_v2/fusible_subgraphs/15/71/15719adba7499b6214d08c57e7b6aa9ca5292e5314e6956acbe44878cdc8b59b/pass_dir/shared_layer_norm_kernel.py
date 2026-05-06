"""
Shared Triton layer normalization kernels used by both small-C (16) and large-C (96)
fused passes.  Each compiled variant is specialised on C and BLOCK_C at JIT time.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _ln_fwd_c16(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N,
    eps,
    BLOCK_C: tl.constexpr,
):
    """Layer-norm for rows of width C=16, BLOCK_C=next_pow2(16)=16."""
    row     = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_C)

    # Load input row and promote to fp32 for accuracy
    x   = tl.load(x_ptr + row * 16 + offsets).to(tl.float32)
    # Computation mask (all elements valid since BLOCK_C == 16 == C)
    mask = offsets < 16

    # Mean
    mean = tl.sum(x.to(tl.float32), axis=0) / 16

    # Diff (scalar mask broadcast)
    diff = (x.to(tl.float32) - mean) * mask.to(tl.float32)

    # Variance
    var  = tl.sum(diff * diff, axis=0) / 16
    rstd = 1.0 / tl.sqrt(var + eps)

    # Weight / bias
    w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Normalise, scale, shift
    y = diff * rstd * w + b

    tl.store(out_ptr + row * 16 + offsets, y.to(x.dtype), mask=mask)


@triton.jit
def _ln_fwd_c96(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N,
    eps,
    BLOCK_C: tl.constexpr,
):
    """Layer-norm for rows of width C=96, BLOCK_C=128 (next_pow2).

    Multiple Triton specialisations will be compiled via @triton.autotune,
    one per BLOCK_C configuration; autotune selects the fastest.
    """
    row     = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_C)

    # Clamp to valid range via mask
    mask    = offsets < 96
    x_raw   = tl.load(x_ptr + row * 96 + offsets, mask=mask, other=0.0)
    x       = x_raw.to(tl.float32)

    # Mean (masked values are 0 so they do not contribute)
    mean    = tl.sum(x, axis=0) / 96

    # Centre by subtracting mean; 0.0 keeps padding zero
    diff    = (x - mean) * mask.to(tl.float32)

    # Variance + reciprocal std
    var     = tl.sum(diff * diff, axis=0) / 96
    rstd    = 1.0 / tl.sqrt(var + eps)

    # Scale / shift
    w       = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b       = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    y = diff * rstd * w + b

    tl.store(out_ptr + row * 96 + offsets, y.to(x_raw.dtype), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 128}, num_warps=8),
        triton.Config({'BLOCK_C': 128}, num_warps=16),
        triton.Config({'BLOCK_C': 256}, num_warps=4),
        triton.Config({'BLOCK_C': 256}, num_warps=8),
        triton.Config({'BLOCK_C': 256}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def _ln_fwd_c96_multi(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C,
    eps,
    BLOCK_C: tl.constexpr,
):
    """Layer-norm for rows of width C=96 using a generalisation that
    supports C < BLOCK_C and multiple rows via the N dimension.
    Grid axis 0 = row (0 .. N-1),  axis 1 = row chunk (0 .. N // CHUNK_ROWS - 1)."""
    pid    = tl.program_id(0)
    chunky = tl.program_id(1)
    CHUNK_ROWS = 8
    row    = pid // CHUNK_ROWS + chunky * CHUNK_ROWS
    offsets = tl.arange(0, BLOCK_C)
    mask   = offsets < C

    x   = tl.load(x_ptr + row * C + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / C
    diff = (x - mean) * mask.to(tl.float32)
    var  = tl.sum(diff * diff, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    y = diff * rstd * w + b
    tl.store(out_ptr + row * C + offsets, y.to(x.dtype), mask=mask)


def run_layer_norm(x, weight, bias, eps=1e-5):
    """Generic dispatcher that calls the right specialised kernel."""
    assert x is not None and weight is not None and bias is not None
    ndim        = x.ndim
    last_dim    = x.shape[-1]
    num_rows    = x.numel() // last_dim
    out         = torch.empty_like(x)

    if last_dim == 16:
        # BLOCK_C must be passed explicitly (it's a tl.constexpr)
        _ln_fwd_c16[(num_rows,)](x, weight, bias, out, num_rows, eps, BLOCK_C=16)
    elif last_dim == 96:
        # BLOCK_C=128 (next power-of-2 >= 96); masked loads/stores handle it
        _ln_fwd_c96[(num_rows,)](x, weight, bias, out, num_rows, eps, BLOCK_C=128)
    else:
        raise ValueError(f"Unsupported last_dim: {last_dim}")

    return out


# ---------------------------------------------------------------------------
# New: direct window-partition kernels (replaces old _ln_fwd_*'
# Instead of calling view→pad→view→permute, these scatter-write the LN
# result *directly* into the window-partitioned tensor layout.
# ---------------------------------------------------------------------------

@triton.jit
def _w_part16_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    eps,
):
    """C=16 tiny model: scatter layer-norm output into [1,8,2,8,2,16].
    N=256 patches, 2×2 windows of 8 each → 8×8 window grid."""
    row  = tl.program_id(0)
    o    = tl.arange(0, 16)
    x    = tl.load(x_ptr + row * 16 + o).to(tl.float32)
    mean = tl.sum(x, axis=0) / 16.0
    d    = (x - mean) * tl.full([16], 1.0, dtype=tl.float32)
    var  = tl.sum(d * d, axis=0) / 16.0
    rstd = 1.0 / tl.sqrt(var + eps)
    wv   = tl.load(w_ptr + o).to(tl.float32)
    bv   = tl.load(b_ptr + o).to(tl.float32)
    y    = d * rstd * wv + bv
    ph   = row // 16
    pw   = row %  16
    w1   = (ph >> 1) + (pw >> 1)   # 0..7
    w0   = (ph &  1) * 8 + (pw &  1)
    p1   = ph - w1 * 2 + w1 *  2
    p0   = pw - w0 * 2 + w0 *  2
    win_idx = w1 * 4096 + w0 * 256 + p1 * 32 + p0
    tl.store(out_ptr + win_idx + o, y.to(x.dtype))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 128}, num_warps=8),
        triton.Config({'BLOCK_C': 128}, num_warps=16),
        triton.Config({'BLOCK_C': 256}, num_warps=4),
        triton.Config({'BLOCK_C': 256}, num_warps=8),
        triton.Config({'BLOCK_C': 256}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def _w_part96_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C, eps,
    BLOCK_C: tl.constexpr,
):
    """C=96 arocr: scatter layer-norm output into [1,32,8,32,8,96].
    N=256 patches, 8×8 windows, window_size=8."""
    row  = tl.program_id(0)
    blk  = tl.arange(0, BLOCK_C)
    mask = blk < 96
    x    = tl.load(x_ptr + row * 96 + blk, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / 96.0
    d    = (x - mean) * mask.to(tl.float32)
    var  = tl.sum(d * d, axis=0) / 96.0
    rstd = 1.0 / tl.sqrt(var + eps)
    wv   = tl.load(w_ptr + blk, mask=mask, other=0.0).to(tl.float32)
    bv   = tl.load(b_ptr + blk, mask=mask, other=0.0).to(tl.float32)
    y    = d * rstd * wv + bv
    ph   = row // 32
    pw   = row %  32
    w1   = (ph // 8) + (pw // 8)
    w0   = (ph %  8) * 8 + (pw %  8)
    p1   = ph - w1 * 8 + w1 * 8
    p0   = pw - w0 * 8 + w0 * 8
    win_idx = w1 * 24576 + w0 * 6144 + p1 * 96 + p0 * 96
    tl.store(out_ptr + win_idx + blk, y.to(x.dtype), mask=mask)


def _run_win_partition16(x, weight, bias):
    out_ln = torch.empty_like(x)
    s0  = torch.empty((1, 16, 16, 16), dtype=x.dtype, device=x.device)
    s1  = torch.empty((1, 8, 2, 8, 2, 16), dtype=x.dtype, device=x.device)
    N   = x.numel() // 16
    _w_part16_kernel[(N,)](x, weight, bias, out_ln, 1e-5)
    _w_part16_kernel[(N,)](x, weight, bias, out_ln, 1e-5)
    return out_ln.view(1, 16, 16, 16), s1


def _run_win_partition96(x, weight, bias):
    out_ln = torch.empty_like(x)
    s0  = torch.empty((1, 256, 256, 96), dtype=x.dtype, device=x.device)
    s1  = torch.empty((1, 32, 8, 32, 8, 96), dtype=x.dtype, device=x.device)
    N   = x.numel() // 96
    _w_part96_kernel[(N,)](x, weight, bias, out_ln, N, 96, 1e-5)
    _w_part96_kernel[(N,)](x, weight, bias, out_ln, N, 96, 1e-5)
    return out_ln.view(1, 256, 256, 96), s1



# ---------------------------------------------------------------------------
# @torch.fx.wrap dispatch hub
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_dispatch(x, weight, bias, route):
    # Both routes (C=16 and C=96) use the same Triton layer-normalisation kernel.
    return run_layer_norm(x, weight, bias)


def replacement_func():
    return fused_dispatch