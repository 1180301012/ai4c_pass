"""Shared Triton kernels and dispatch function for Add+LayerNorm fusion passes.

All pattern pass files import `fused_dispatch` from here so that
replacement_func() returns the same function object across every pass,
satisfying output_pass_replacement_func_limit=1.
"""
import torch
import triton
import triton.language as tl


# ── Triton kernel (works for any BLOCK_SIZE power of 2) ─────────────────────
@triton.jit
def _fused_add_ln_kernel(
    x_ptr, y_ptr, w_ptr, b_ptr,
    out_add_ptr, out_ln_ptr,
    eps,
    N_ROWS,
    BLOCK_SIZE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)

    # Load and upcast to fp32 for stable computation
    x_f32 = tl.load(x_ptr + row * BLOCK_SIZE + offsets).to(tl.float32)
    y_f32 = tl.load(y_ptr + row * BLOCK_SIZE + offsets).to(tl.float32)
    add_f32 = x_f32 + y_f32

    # Write add result (dropout p=0.1, training=False is identity)
    tl.store(out_add_ptr + row * BLOCK_SIZE + offsets, add_f32.to(INPUT_DTYPE))

    # ── Layer-norm ────────────────────────────────────────────────────────────
    mean    = tl.sum(add_f32, axis=0) / BLOCK_SIZE
    diff    = add_f32 - mean
    var     = tl.sum(diff * diff, axis=0) / BLOCK_SIZE
    inv_std = 1.0 / tl.sqrt(var + eps)
    norm    = diff * inv_std

    w      = tl.load(w_ptr + offsets).to(tl.float32)
    b      = tl.load(b_ptr + offsets).to(tl.float32)
    ln_out = norm * w + b

    tl.store(out_ln_ptr + row * BLOCK_SIZE + offsets, ln_out.to(INPUT_DTYPE))


# ── Single dispatch wrapper shared by ALL pass files ─────────────────────────
@torch.fx.wrap
def fused_dispatch(x, y, weight, bias, route):
    """route: '1024' or '2048' – selects the hidden dimension."""
    H      = 1024 if route == '1024' else 2048
    N_ROWS = x.numel() // H

    if x.dtype == torch.float32:
        INPUT_DTYPE = tl.float32
    elif x.dtype == torch.float16:
        INPUT_DTYPE = tl.float16
    else:                          # bfloat16
        INPUT_DTYPE = tl.bfloat16

    out_add = torch.empty_like(x)
    out_ln  = torch.empty_like(x)

    _fused_add_ln_kernel[(N_ROWS,)](
        x, y, weight, bias,
        out_add, out_ln,
        eps=1e-5,
        N_ROWS=N_ROWS,
        BLOCK_SIZE=H,
        INPUT_DTYPE=INPUT_DTYPE,
        num_warps=16,
    )

    return (out_add, out_ln)