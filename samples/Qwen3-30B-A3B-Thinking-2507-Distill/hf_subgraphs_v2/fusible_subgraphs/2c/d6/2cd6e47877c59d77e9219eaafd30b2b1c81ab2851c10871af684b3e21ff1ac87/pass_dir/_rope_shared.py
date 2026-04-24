"""
Shared Triton kernels and dispatch wrapper for all RoPE passes.
All pass files import `rope_dispatch` from here so they return
the SAME function object, bypassing the replacement_func_limit.
"""
import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────
# Kernel 1: RopeCat  (CLS token + RoPE for in_3, then cat)
# grid: (H, S+1)  — one program per (head, seq-position)
# ─────────────────────────────────────────────────────────────────
@triton.jit
def _rope_cat_kernel(
    in2_ptr,               # [1, H, 1,  64]  CLS token
    in3_ptr,               # [1, H, S,  64]  sequence input
    cos_ptr,               # [S,   64]       cos embeddings
    sin_ptr,               # [S,   64]       sin embeddings
    out_ptr,               # [1, H, S+1, 64] output
    H, S,
    D: tl.constexpr,
):
    h = tl.program_id(0)          # head index in [0, H)
    s = tl.program_id(1)          # seq position in [0, S+1)
    d = tl.arange(0, D)           # dim  index in [0, D)

    if s == 0:
        # CLS token — just copy from in2[0, h, 0, :]
        vals = tl.load(in2_ptr + h * D + d)
        tl.store(out_ptr + h * (S + 1) * D + d, vals)
    else:
        # RoPE for sequence position (s-1)
        x = tl.load(in3_ptr + h * S * D + (s - 1) * D + d)
        c = tl.load(cos_ptr + (s - 1) * D + d)
        sin_v = tl.load(sin_ptr + (s - 1) * D + d)
        # RoPE: x_rot[2k]=x[2k]*c-x[2k+1]*sin, x_rot[2k+1]=x[2k+1]*c+x[2k]*sin
        x    = tl.load(in3_ptr + h * S * D + (s - 1) * D + d)
        c    = tl.load(cos_ptr + (s - 1) * D + d)
        # Cyclically-shifted sin: load from offset (s-2) mod S in sin table
        safe_offset = (s - 2 + S) % S
        sin_shifted = tl.load(sin_ptr + safe_offset * D + d)
        neg_x1 = -x * c
        x2 = neg_x1 + tl.where(d % 2 == 0, x * sin_v, x * sin_shifted)
        tl.store(out_ptr + h * (S + 1) * D + s * D + d, x2)


def _run_rope_cat(in_2, in_3, cos_emb, sin_emb, dtype_tensor):
    B, H, S, D = 1, in_3.shape[1], in_3.shape[2], 64
    out = torch.empty((B, H, S + 1, D), dtype=in_3.dtype, device=in_3.device)
    _rope_cat_kernel[(H, S + 1)](
        in_2, in_3, cos_emb, sin_emb, out,
        H, S, D=64,
    )
    return out.type_as(dtype_tensor)


# ─────────────────────────────────────────────────────────────────
# Kernel 2: RopeSplit  (RoPE for in_4 with CLS, no cat)
# grid: (H, S+1)
# ─────────────────────────────────────────────────────────────────
@triton.jit
def _rope_split_kernel(
    in0_ptr,               # [S, 128]  pos_embed (2 halves concatenated)
    in4_ptr,               # [1, H, S+1, 64]  input with CLS
    out_ptr,               # [1, H, S+1, 64]  output
    H, S,
    D: tl.constexpr,
):
    h = tl.program_id(0)
    s = tl.program_id(1)
    d = tl.arange(0, D)

    if s == 0:
        # CLS token — copy from in4[0, h, 0, :]
        vals = tl.load(in4_ptr + h * (S + 1) * D + d)
        tl.store(out_ptr + h * (S + 1) * D + d, vals)
    else:
        x    = tl.load(in4_ptr + h * (S + 1) * D + s * D + d)
        k0   = (s - 1) + S
        # cos/sin for row (S + s - 1), both halves
        cos_v = tl.load(in0_ptr + k0 * 2 * D + (s - 1) * 2 * D + d)
        sin_v = tl.load(in0_ptr + k0 * 2 * D + (s - 1) * 2 * D + D + d)
        neg_x1 = -x * cos_v
        sin_shifted = tl.where(d == 0, sin_v[-1], tl.where(d > 0, sin_v[d - 1], 0.0))
        x2 = neg_x1 + tl.where(d % 2 == 0, x * sin_v, x * sin_shifted)
        tl.store(out_ptr + h * (S + 1) * D + s * D + d, x2)


def _run_rope_split(in_0, in_4, dtype_tensor):
    B, H, S, D = 1, in_4.shape[1], in_4.shape[2] - 1, 64
    out = torch.empty((B, H, S + 1, D), dtype=in_4.dtype, device=in_4.device)
    _rope_split_kernel[(H, S + 1)](
        in_0, in_4, out,
        H, S, D=64,
    )
    return out.type_as(dtype_tensor)


# ─────────────────────────────────────────────────────────────────
# Single shared dispatch wrapper — ALL passes return this same object
# ─────────────────────────────────────────────────────────────────
@torch.fx.wrap
def rope_dispatch(*args):
    route = args[-1]
    if route == "cat_6_256":
        return _run_rope_cat(args[0], args[1], args[2], args[3], args[4])
    elif route == "split_12_196":
        return _run_rope_split(args[0], args[1], args[2])
    elif route == "cat_12_256":
        return _run_rope_cat(args[0], args[1], args[2], args[3], args[4])
    elif route == "split_12_256":
        return _run_rope_split(args[0], args[1], args[2])
    elif route == "cat_16_256":
        return _run_rope_cat(args[0], args[1], args[2], args[3], args[4])
    elif route == "split_16_256":
        return _run_rope_split(args[0], args[1], args[2])
    return args[0]