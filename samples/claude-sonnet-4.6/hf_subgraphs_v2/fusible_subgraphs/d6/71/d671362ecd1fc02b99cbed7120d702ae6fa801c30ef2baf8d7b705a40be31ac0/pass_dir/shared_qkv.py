"""
Shared Triton kernels and single dispatch wrapper used by both
FuseSplitPermuteTranspose, FusePermute0213, and FuseQKTV passes.

Using the route-string dispatch pattern so all passes return the
SAME replacement_func object, satisfying output_pass_replacement_func_limit=1.
"""
import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Kernel 0: Full QK^TV from contiguous tmp_4 (B, 49, 8, 192)
#
# out[b,h,d,s]  = x[b,s,h,32+d]  → K^T  (B,8,32,49)
# out[b,h,s,d]  = x[b,s,h,d]     → Q    (B,8,49,32)
# out[b,h,s,dv] = x[b,s,h,64+dv] → V    (B,8,49,128)
#
# Input strides passed at runtime (accommodates any batch size).
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _qktv_kernel(
    x_ptr,
    q_ptr,
    kt_ptr,
    v_ptr,
    B,
    stride_b,   # x.stride(0)
    stride_s,   # x.stride(1)
    stride_h,   # x.stride(2)
    BLOCK_S: tl.constexpr,   # 64
    S:       tl.constexpr,   # 49
    H:       tl.constexpr,   # 8
    D_QK:    tl.constexpr,   # 32
    D_V:     tl.constexpr,   # 128
):
    pid = tl.program_id(0)
    b   = pid // H
    h   = pid %  H

    s_range = tl.arange(0, BLOCK_S)
    s_mask  = s_range < S
    d_qk    = tl.arange(0, D_QK)
    d_v     = tl.arange(0, D_V)

    x_bh = b * stride_b + h * stride_h

    # ── Load Q: x[b, s, h, 0:D_QK] → (BLOCK_S, D_QK) ──────────────────────
    q_offs = x_bh + s_range[:, None] * stride_s + d_qk[None, :]
    q_vals = tl.load(x_ptr + q_offs, mask=s_mask[:, None], other=0.0)

    # ── Load K in transposed layout (D_QK, BLOCK_S) ─────────────────────────
    k_offs = x_bh + s_range[None, :] * stride_s + (D_QK + d_qk[:, None])
    k_vals = tl.load(x_ptr + k_offs, mask=s_mask[None, :], other=0.0)

    # ── Load V: x[b, s, h, 2*D_QK:] → (BLOCK_S, D_V) ──────────────────────
    v_offs = x_bh + s_range[:, None] * stride_s + (2 * D_QK + d_v[None, :])
    v_vals = tl.load(x_ptr + v_offs, mask=s_mask[:, None], other=0.0)

    # ── Store Q: q[b,h,s,d] ─────────────────────────────────────────────────
    q_bh = b * (H * S * D_QK) + h * (S * D_QK)
    tl.store(q_ptr + q_bh + s_range[:, None] * D_QK + d_qk[None, :],
             q_vals, mask=s_mask[:, None])

    # ── Store K^T: kt[b,h,d,s] ──────────────────────────────────────────────
    kt_bh = b * (H * D_QK * S) + h * (D_QK * S)
    tl.store(kt_ptr + kt_bh + d_qk[:, None] * S + s_range[None, :],
             k_vals, mask=s_mask[None, :])

    # ── Store V: v[b,h,s,dv] ────────────────────────────────────────────────
    v_bh = b * (H * S * D_V) + h * (S * D_V)
    tl.store(v_ptr + v_bh + s_range[:, None] * D_V + d_v[None, :],
             v_vals, mask=s_mask[:, None])


# ──────────────────────────────────────────────────────────────────────────────
# Kernel 1: K^T  — out[b,h,d,s] = x[b,s,h,d]
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _kt_kernel(
    x_ptr,
    kt_ptr,
    B,
    stride_b,
    stride_s,
    stride_h,
    BLOCK_S: tl.constexpr,
    S:       tl.constexpr,
    H:       tl.constexpr,
    D:       tl.constexpr,
):
    # One program per batch element b; loop over all H heads (constexpr → unrolled)
    b = tl.program_id(0)

    s_range = tl.arange(0, BLOCK_S)
    s_mask  = s_range < S
    d_range = tl.arange(0, D)

    for h in range(H):
        x_bh = b * stride_b + h * stride_h
        x_offs = x_bh + s_range[None, :] * stride_s + d_range[:, None]
        vals   = tl.load(x_ptr + x_offs, mask=s_mask[None, :], other=0.0)

        kt_bh  = b * (H * D * S) + h * (D * S)
        kt_offs = kt_bh + d_range[:, None] * S + s_range[None, :]
        tl.store(kt_ptr + kt_offs, vals, mask=s_mask[None, :])


# ──────────────────────────────────────────────────────────────────────────────
# Kernel 2: permute(0,2,1,3) — out[b,h,s,d] = x[b,s,h,d]
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _permute_0213_kernel(
    x_ptr,
    out_ptr,
    B,
    D,
    stride_b,
    stride_s,
    stride_h,
    BLOCK_S:  tl.constexpr,
    BLOCK_D:  tl.constexpr,
    S:        tl.constexpr,
    H:        tl.constexpr,
):
    # One program per batch element b; loop over all H heads (constexpr → unrolled)
    b = tl.program_id(0)

    s_range = tl.arange(0, BLOCK_S)
    s_mask  = s_range < S
    d_range = tl.arange(0, BLOCK_D)
    d_mask  = d_range < D
    mask_2d = s_mask[:, None] & d_mask[None, :]

    for h in range(H):
        x_bh = b * stride_b + h * stride_h
        x_offs = x_bh + s_range[:, None] * stride_s + d_range[None, :]
        vals   = tl.load(x_ptr + x_offs, mask=mask_2d, other=0.0)

        out_bh  = b * (H * S * D) + h * (S * D)
        out_offs = out_bh + s_range[:, None] * D + d_range[None, :]
        tl.store(out_ptr + out_offs, vals, mask=mask_2d)


# ──────────────────────────────────────────────────────────────────────────────
# Shared dispatch — single @torch.fx.wrap function for ALL passes.
# route="qktv"    → full Q/K^T/V from contiguous tmp_4 (FuseQKTV)
# route="kt"      → K^T from split slice (FuseSplitPermuteTranspose)
# route="permute" → permute from split slice (FusePermute0213)
# ──────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def shared_dispatch(x, route):
    B        = x.shape[0]
    S        = 49
    H        = 8
    stride_b = x.stride(0)
    stride_s = x.stride(1)
    stride_h = x.stride(2)

    if route == "qktv":
        # x = tmp_4: (B, 49, 8, 192) contiguous
        D_QK = 32
        D_V  = 128
        q  = torch.empty((B, H, S, D_QK), dtype=x.dtype, device=x.device)
        kt = torch.empty((B, H, D_QK, S), dtype=x.dtype, device=x.device)
        v  = torch.empty((B, H, S, D_V),  dtype=x.dtype, device=x.device)
        _qktv_kernel[(B * H,)](
            x, q, kt, v, B,
            stride_b, stride_s, stride_h,
            BLOCK_S=64, S=S, H=H, D_QK=D_QK, D_V=D_V,
        )
        return q, kt, v

    elif route == "kt":
        # x = split[1]: K slice (B, 49, 8, 32) non-contiguous
        D  = 32
        kt = torch.empty((B, H, D, S), dtype=x.dtype, device=x.device)
        _kt_kernel[(B,)](
            x, kt, B,
            stride_b, stride_s, stride_h,
            BLOCK_S=64, S=S, H=H, D=D,
        )
        return kt

    else:
        # route == "permute": x = split[0] or split[2], Q or V slice
        D   = x.shape[3]
        out = torch.empty((B, H, S, D), dtype=x.dtype, device=x.device)
        if D <= 32:
            _permute_0213_kernel[(B,)](
                x, out, B, D,
                stride_b, stride_s, stride_h,
                BLOCK_S=64, BLOCK_D=32, S=S, H=H,
            )
        else:
            _permute_0213_kernel[(B,)](
                x, out, B, D,
                stride_b, stride_s, stride_h,
                BLOCK_S=64, BLOCK_D=128, S=S, H=H,
            )
        return out