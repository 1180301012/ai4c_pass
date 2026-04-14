"""
Shared Triton kernels and dispatch wrapper imported by all pass files.
Using the routing technique so all passes return the same @torch.fx.wrap function.
"""
import torch
import triton
import triton.language as tl


# ── Layer-Norm kernel for hidden_size=384  (BLOCK_SIZE must be power-of-2) ────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16, num_stages=2),
    ],
    key=['N_rows', 'N_cols'],
)
@triton.jit
def _ln_fwd_384(x_ptr, w_ptr, b_ptr, o_ptr,
                N_rows, N_cols, eps,
                BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_cols
    x = tl.load(x_ptr + row * N_cols + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N_cols
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    xn = (x - mean) * rstd
    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(o_ptr + row * N_cols + cols, xn * w + b, mask=mask)


# ── Layer-Norm kernel for hidden_size=768 ─────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=32, num_stages=2),
    ],
    key=['N_rows', 'N_cols'],
)
@triton.jit
def _ln_fwd_768(x_ptr, w_ptr, b_ptr, o_ptr,
                N_rows, N_cols, eps,
                BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N_cols
    x = tl.load(x_ptr + row * N_cols + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N_cols
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    xn = (x - mean) * rstd
    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(o_ptr + row * N_cols + cols, xn * w + b, mask=mask)


# ── Layer-Norm kernel for hidden_size=32 (2-D batched, no autotune) ───────────
# Fixed BLOCK_ROWS=256 → 1 GPU program for N_rows=236, minimal launch overhead.
# Safe row clamping prevents out-of-bounds access when BLOCK_ROWS > N_rows.

@triton.jit
def _ln_fwd_32(x_ptr, w_ptr, b_ptr, o_ptr,
               N_rows, N_cols, eps,
               BLOCK_ROWS: tl.constexpr, BLOCK_COLS: tl.constexpr):
    base_row = tl.program_id(0) * BLOCK_ROWS
    row_offs = base_row + tl.arange(0, BLOCK_ROWS)
    col_offs = tl.arange(0, BLOCK_COLS)
    row_mask = row_offs < N_rows
    col_mask = col_offs < N_cols
    full_mask = row_mask[:, None] & col_mask[None, :]

    # Clamp row indices to last valid row to prevent out-of-bounds memory access
    # (Triton may speculatively access masked addresses; clamping makes them safe)
    safe_rows = tl.where(row_mask, row_offs, N_rows - 1)

    x = tl.load(
        x_ptr + safe_rows[:, None] * N_cols + col_offs[None, :],
        mask=full_mask, other=0.0
    ).to(tl.float32)

    mean = tl.sum(x, axis=1) / N_cols
    xc = tl.where(col_mask[None, :], x - mean[:, None], 0.0)
    var = tl.sum(xc * xc, axis=1) / N_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    xn = (x - mean[:, None]) * rstd[:, None]

    w = tl.load(w_ptr + col_offs, mask=col_mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + col_offs, mask=col_mask, other=0.0).to(tl.float32)
    out = xn * w[None, :] + b[None, :]

    tl.store(
        o_ptr + safe_rows[:, None] * N_cols + col_offs[None, :],
        out, mask=full_mask
    )


# ── Unified dispatch wrapper (shared by all pass files) ───────────────────────

@torch.fx.wrap
def _dispatch(arg0, arg1, arg2, route):
    """
    Route-based dispatch for layer-norm operations.
      route == "ln_384" → layer_norm(arg2, (384,), arg1, arg0, 1e-12)
      route == "ln_768" → layer_norm(arg2, (768,), arg1, arg0, 1e-12)
      route == "ln_32"  → layer_norm(arg2, (32,),  arg1, arg0, 1e-12)
    arg0=bias, arg1=weight, arg2=input_tensor
    """
    if route == "ln_384":
        bias, weight, x = arg0, arg1, arg2
        N_cols = x.shape[-1]
        N_rows = x.shape[0] * x.shape[1]
        out = torch.empty_like(x)
        _ln_fwd_384[(N_rows,)](x, weight, bias, out, N_rows, N_cols, 1e-12)
        return out

    elif route == "ln_768":
        bias, weight, x = arg0, arg1, arg2
        N_cols = x.shape[-1]
        N_rows = x.shape[0] * x.shape[1]
        out = torch.empty_like(x)
        _ln_fwd_768[(N_rows,)](x, weight, bias, out, N_rows, N_cols, 1e-12)
        return out

    else:  # route == "ln_32"
        bias, weight, x = arg0, arg1, arg2
        N_cols = x.shape[-1]
        N_rows = x.shape[0] * x.shape[1]
        out = torch.empty_like(x)
        # Fixed BLOCK_ROWS=256, BLOCK_COLS=32: for N_rows=236 → 1 GPU program
        num_programs = (N_rows + 255) // 256
        _ln_fwd_32[(num_programs,)](
            x, weight, bias, out, N_rows, N_cols, 1e-12,
            BLOCK_ROWS=256, BLOCK_COLS=32
        )
        return out