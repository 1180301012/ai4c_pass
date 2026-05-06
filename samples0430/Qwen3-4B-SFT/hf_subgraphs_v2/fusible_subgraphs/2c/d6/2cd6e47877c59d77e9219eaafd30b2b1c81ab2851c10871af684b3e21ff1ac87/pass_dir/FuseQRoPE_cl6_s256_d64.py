"""
Pass: FuseQRoPE_cl6_s256_d64
Matches the QRoPE+cat+type_as subgraph where reshape shape is (1, 6, 256, 64).
Target: eva02_small graphs (H=6, S=256, D=64).
"""

# ── imports ──────────────────────────────────────────────────────────────────
import torch
import triton
import triton.language as tl

# ── Triton kernel ─────────────────────────────────────────────────────────────
@triton.jit
def _rope_kernel_q(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_q_ptr,
    H,
    S,
    BLOCK_D: tl.constexpr,
):
    """
    One program handles (h, s) pair.
    Loads the query block [BLOCK_D], applies RoPE, writes to even/odd halves.
    Handles batch=1 assumed: out layout [H, S, D], stride every dim is D.
    """
    h = tl.program_id(0)
    s = tl.program_id(1)
    d_start = tl.program_id(2) * BLOCK_D

    d_offs = tl.arange(0, BLOCK_D)
    d_global = d_start + d_offs

    # Load cos, sin for position s (shape [D])
    cos_val = tl.load(cos_ptr + s * BLOCK_D + d_offs)
    sin_val = tl.load(sin_ptr + s * BLOCK_D + d_offs)

    # Load the query block
    x = tl.load(x_ptr + h * S * BLOCK_D + s * BLOCK_D + d_global)

    # --- RoPE math (QRoPE: x_even*sinf + x_odd*cosf ; x_odd*cosf - x_even*sinf) ---
    x_even = tl.load(x_ptr + h * S * BLOCK_D + s * BLOCK_D + d_global,
                     mask=d_global % 2 == 0, other=0.0)
    x_odd  = tl.load(x_ptr + h * S * BLOCK_D + s * BLOCK_D + d_global,
                     mask=d_global % 2 == 1, other=0.0)

    rope_q1 = x_even * sin_val + x_odd * cos_val   # contributes to even output slots
    rope_q2 = x_odd  * cos_val - x_even * sin_val   # contributes to odd output slots

    # --- Store: out[h,s,d] = rope_q1 if d even, rope_q2 if d odd ---
    out_even_offs = tl.arange(0, BLOCK_D // 2)
    out_odd_offs  = tl.arange(0, BLOCK_D // 2)
    output_base   = h * S * BLOCK_D + s * BLOCK_D

    tl.store(out_q_ptr + output_base + out_even_offs,
             rope_q1,  mask=out_even_offs < BLOCK_D // 2)
    tl.store(out_q_ptr + output_base + out_even_offs + BLOCK_D // 2,
             rope_q2,  mask=out_even_offs < BLOCK_D // 2)

    # For D=64: output_base + s*64 + out_even_offs == output_base + out_even_offs*2
    old_even = tl.load(x_ptr + h * S * BLOCK_D + s * BLOCK_D + out_even_offs * 2)
    old_odd  = tl.load(x_ptr + h * S * BLOCK_D + s * BLOCK_D + out_even_offs * 2 + 1)
    # Already computed above via rope_q1/q2


# ── pattern / replacement_args ────────────────────────────────────────────────
def pattern(in_2, in_3, in_5, in_6):
    tmp_1 = in_3 * in_1
    tmp_2 = in_3[(Ellipsis, slice(1, None, 2))]
    tmp_3 = -tmp_2
    tmp_4 = in_3[(Ellipsis, slice(None, None, 2))]
    tmp_5 = torch.stack([tmp_3, tmp_4], -1)
    tmp_6 = tmp_5.reshape((1, 6, 256, 64))
    tmp_7 = tmp_6 * in_5
    tmp_8 = tmp_1 + tmp_7
    tmp_9 = torch.cat([in_2, tmp_8], dim=2)
    tmp_10 = tmp_9.type_as(in_6)
    return tmp_10


def replacement_args(in_2, in_3, in_5, in_6):
    return (in_2, in_3, in_5, in_6)


# ── wrapper (must be @torch.fx.wrap) ─────────────────────────────────────────
@torch.fx.wrap
def _qr_t6s256(x, cos, sin, cls_tk, ref):
    """
    x         : [1, H, S, D]  -- query tokens (in_3)
    cos       : [N, D]        -- cos rope embedding (in_1)
    sin       : [N, D]        -- sin rope embedding (in_5)
    cls_tk    : [1, H, 1, D]  -- cls token (in_2)
    ref       : [1, H, S+1, D] -- dtype reference (in_6)
    returns   : [1, H, S+1, D]
    """
    H = x.shape[1]
    S = x.shape[2]
    D = x.shape[3]

    rope_out = torch.empty((1, H, S, D), dtype=ref.dtype, device=ref.device)
    out      = torch.empty((1, H, S + 1, D), dtype=ref.dtype, device=ref.device)

    # flatten batch dim (B=1) so layout is [H, S, D] for contiguous access
    x_flat   = x.view(H, S, D)
    out_flat = out.view(H, S + 1, D)

    grid = (H, S, triton.cdiv(D, 64))
    _rope_kernel_q[grid](
        x_flat, cos, sin, rope_out,
        H, S,
        BLOCK_D=64,
    )

    # Cat cls token with rope output
    out[:, :, 0, :] = cls_tk.view(H, D)
    out[:, :, 1:, :] = rope_out.view(H, S, D)
    return out


def replacement_func():
    return _qr_t6s256