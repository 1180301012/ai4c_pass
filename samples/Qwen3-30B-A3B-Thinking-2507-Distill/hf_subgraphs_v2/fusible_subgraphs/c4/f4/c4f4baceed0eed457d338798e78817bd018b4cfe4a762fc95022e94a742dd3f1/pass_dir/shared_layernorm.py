"""
Shared Triton kernels and dispatch functions used by all FuseAddLayernorm_*
passes.  Importing this module ensures all passes return the *same* function
object from replacement_func(), satisfying the replacement_func_limit
constraint.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel 1: standalone layer-norm  (used as fallback when full-chain doesn't
#            match for some reason, or by passes that only replace layernorm)
# ---------------------------------------------------------------------------

@triton.jit
def _ln_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    C, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """One program = one row of C elements."""
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < C
    base = row * C

    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / C
    x_c  = tl.where(mask, x - mean, 0.0)
    var  = tl.sum(x_c * x_c, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    out = x_c * rstd * w + b
    tl.store(out_ptr + base + offs, out, mask=mask)


@torch.fx.wrap
def dispatch_layernorm(x, weight, bias, route):
    """
    Standalone layer-norm dispatch.  Route selects the channel variant.
    Fixed BLOCK_SIZE/num_warps — no autotune overhead.
    """
    if route == "C384":
        C = 384
        BLOCK_SIZE = 512
        NW = 4
    elif route == "C192":
        C = 192
        BLOCK_SIZE = 256
        NW = 4
    else:           # "C96"
        C = 96
        BLOCK_SIZE = 128
        NW = 2

    N   = x.numel() // C
    out = torch.empty_like(x)

    _ln_kernel[(N,)](
        x, weight, bias, out,
        C, 1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NW,
    )

    return out


# ---------------------------------------------------------------------------
# Kernel 2: fused add + layer-norm  (input already in [1, N, C] layout)
#   Replaces: view(1,N,C) → add → layer_norm
#   Used when the window-partition output is already contiguous.
# ---------------------------------------------------------------------------

@triton.jit
def _add_ln_kernel(
    x_ptr, y_ptr, w_ptr, b_ptr, out_ptr,
    C, eps,
    BLOCK_SIZE: tl.constexpr,
):
    """One program = one row of C elements."""
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < C
    base = row * C

    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    added = x + y
    tl.store(out_ptr + base + offs, added, mask=mask)

    mean = tl.sum(added, axis=0) / C
    x_c  = tl.where(mask, added - mean, 0.0)
    var  = tl.sum(x_c * x_c, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = x_c * rstd * w + b
    tl.store(out_ptr + base + offs, out, mask=mask)


@torch.fx.wrap
def dispatch_fused_view_add_ln(x, y, weight, bias, route):
    """
    Fused add+layernorm for already-partitioned input x [1,N,C] + y [1,N,C].
    Returns (added, ln_out) matching model outputs (tmp_8, tmp_9).
    """
    if route == "C384":
        C, BLOCK_SIZE, NW = 384, 512, 4
    elif route == "C192":
        C, BLOCK_SIZE, NW = 192, 256, 4
    else:               # "C96"
        C, BLOCK_SIZE, NW = 96, 128, 2

    N    = x.numel() // C
    added  = torch.empty_like(x)
    ln_out = torch.empty_like(x)

    _add_ln_kernel[(N,)](
        x, y, weight, bias, added,
        C, 1e-5,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=NW,
    )

    return (added, ln_out)


# ---------------------------------------------------------------------------
# Kernel 3: fused gather-add-layernorm  (for the full chain: 6D input → add → LN)
#   Fuses: contiguous → view → roll → slice → contiguous → view → add → LN
# ---------------------------------------------------------------------------

@triton.jit
def _fused_gather_add_ln_kernel(
    x_ptr, in2_ptr, w_ptr, b_ptr, out_added_ptr, out_ln_ptr,
    N_h_in, N_w_in, H_in, W_in, N_h_out, N_w_out, C,
    BLOCK_C: tl.constexpr,
):
    """
    One program per output row (n_out_row).
    Computes window-partition gather → add → layer-norm for that row.
    """
    n_out_row = tl.program_id(0)

    # Decompose output row into (window-index, local-row)
    n_win_h   = n_out_row // N_h_out
    out_local = n_out_row % N_h_out

    # Cyclic roll: input row = (out_local + SHIFT) % WIN_H
    SHIFT = 3
    WIN_H = 7
    WIN_W = 7

    in_row  = (out_local + SHIFT) % WIN_H
    n_win_w = n_win_h % N_w_in         # which window column
    n_win_h_ = n_win_h // N_w_in       # which window row

    in_row_global = n_win_h_ * WIN_H + in_row

    # Base pointer for this input row (6-D tensor viewed as flat [H*W, C])
    in_col_base = (n_win_w * WIN_W + (out_local + SHIFT) % WIN_W) * C
    in_row_base = (in_row_global * N_w_in + n_win_w * WIN_W) * C

    col_offs = tl.arange(0, BLOCK_C)
    mask_c   = col_offs < C

    # Gather x[n_win_h_, in_row, n_win_w_, (out_local+SHIFT)%WIN_W, col]
    x_idx  = in_row_base + in_col_base + col_offs
    xv     = tl.load(x_ptr + x_idx, mask=mask_c, other=0.0).to(tl.float32)

    # Load in_2 (contiguous [1, N_h_out*N_w_out, C])
    out_row_base = n_out_row * C
    yv = tl.load(in2_ptr + out_row_base + col_offs, mask=mask_c, other=0.0).to(tl.float32)

    # Fused add
    added = xv + yv

    # Store added result
    tl.store(out_added_ptr + out_row_base + col_offs, added, mask=mask_c)

    # Layer-norm on added
    mean = tl.sum(added, axis=0) / C
    xc   = tl.where(mask_c, added - mean, 0.0)
    var  = tl.sum(xc * xc, axis=0) / C
    rstd = 1.0 / tl.sqrt(var + 1e-5)

    wv = tl.load(w_ptr + col_offs, mask=mask_c, other=1.0).to(tl.float32)
    bv = tl.load(b_ptr + col_offs, mask=mask_c, other=0.0).to(tl.float32)

    out = xc * rstd * wv + bv
    tl.store(out_ln_ptr + out_row_base + col_offs, out, mask=mask_c)


@torch.fx.wrap
def dispatch_fused(x6d, in2, weight, bias, route):
    """
    Fused window-partition + add + layer-norm dispatch.
    x6d   : in_3 before contiguous  [1, N_h_win, WIN_H, N_w_win, WIN_W, C]
    in2   : in_2                    [1, N_out, C]
    weight, bias : layernorm params  [C]
    route : "C384" | "C192" | "C96"
    Returns (added, ln_out)
    """
    if route == "C384":
        C = 384
    elif route == "C192":
        C = 192
    else:               # "C96"
        C = 96

    # Shape of 6-D tensor
    sh = x6d.shape       # e.g. [1, 5, 7, 5, 7, 384]
    N_h_in  = sh[1]
    N_w_in  = sh[3]
    N_h_out = N_h_in - 1   # (win_h - 1) windows fit in the output
    N_w_out = N_w_in - 1
    H_in    = sh[1] * sh[2]   # N_h_win * win_h
    W_in    = sh[3] * sh[4]   # N_w_win * win_w
    N_out   = N_h_out * N_w_out  # number of output rows

    added = torch.empty_like(in2)
    ln_out = torch.empty_like(in2)

    _fused_gather_add_ln_kernel[(N_out,)](
        x6d, in2, weight, bias, added, ln_out,
        N_h_in, N_w_in, H_in, W_in, N_h_out, N_w_out, C,
        BLOCK_C=512,
    )

    return (added, ln_out)