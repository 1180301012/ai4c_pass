"""
Fused Rotary Relative-Position Attention kernel for 64-token attention sequences.
Handles the full pattern:
  matmul(Q, W) -> rotate_bias + add(in2) -> add(in0) -> softmax -> matmul(V) -> transpose
where Q=[4,8,8,128], W=[128,15], in2=[4,8,8,8,8], V=[4,64,128].
"""
import torch
import triton
import triton.language as tl


# Grid: (B*Hp*Hg, D//Ho) = (4*8*8, 128//8) = (256, 4)
@triton.jit
def _fused_rot_attn_v64(
    in_1_ptr,   # Q:  [4, 8, 8, 128]  strides [32768, 4096, 512, 1]
    in_3_ptr,   # W:  [128, 15]
    in_2_ptr,   # rel_bias [4, 8, 8, 8, 8] strides [4096, 512, 64, 8, 1]
    in_4_ptr,   # V_mat   [4, 64, 128]  strides [16384, 128, 1]
    out_ptr,    # [4, 128, 64] contiguous strides [8192, 64, 1]
):
    pid_batch = tl.program_id(0)   # 0..255
    pid_n     = tl.program_id(1)   # 0..3

    b = pid_batch // 64          # batch [0,4)
    h = (pid_batch // 8) % 8     # head  [0,8)
    g = pid_batch % 8            # group  [0,8)

    col_j = pid_n * 8 + tl.arange(0, 8)    # [0,8) flat-key indices

    # ── Load Q[b,h,:,:]: [8,128] ────────────────────────────────────────────
    q_base = b * 32768 + h * 512
    q_off  = (tl.arange(0, 8)[:, None] + pid_n) * 128 + tl.arange(0, 128)[None, :]
    q = tl.load(in_1_ptr + q_base + q_off).to(tl.float32)    # [8,128]

    # ── Load W[128,8]: row j from in_3[128,15] (cols 0–7 valid) ─────────────
    cw = tl.arange(0, 128)[:, None]
    cs = tl.arange(0, 8)[None, :]
    w  = tl.load(in_3_ptr + cw * 8 + cs).to(tl.float32)     # [128,8]

    # ── QW: [8,8] = Q @ W ───────────────────────────────────────────────────
    attn = tl.dot(q, w)   # [8,8]

    # ── Rel-bias in_2[b,h,g,n1i,n1j]: strides [4096,512,64,8,1], dim1=Hg*Hp*S=512 ─
    bi    = b * 4096 + h * 512 + g * 64
    ni    = tl.arange(0, 8)
    n1i   = ni // 8           # 0..7
    n1j   = ni % 8            # 0..7
    hg    = tl.where(h < 8, 8, 0)
    off   = bi + n1i * 512 + hg * 512 + g * 8 + n1j
    bias  = tl.load(in_2_ptr + off).to(tl.float32)           # [8]

    attn = attn + bias[None, :]    # broadcast

    # ── Additive bias in_0[b, n1i*8+g*8+n1j+col_j, col_j] ──────────────────
    # in_0 [4,64,64]: offset = b*64*64 + (i*8+g*8+n1i+n1j)*8 + col_j
    attn = attn + (b * 4096 + n1i * 512 + g * 64 + ni * 8 + col_j)[None, :]

    # ── Softmax (float32) ────────────────────────────────────────────────────
    attn_mask = (n1i >= 8) | (n1j >= 8)
    attn      = tl.where(attn_mask, -float('inf'), attn)
    amax      = tl.max(attn, axis=1, keep_dims=True)
    attn      = attn - amax
    et        = tl.exp(attn)
    et        = tl.where(attn_mask, 0.0, et)
    et        = et / tl.sum(et, axis=1, keep_dims=True)

    # ── Load V_slice in_4[b, col_j*8+g*8 : +8, :]: [8,128] ─────────────────
    v_off  = tl.arange(0, 8)[:, None] * 128 + tl.arange(0, 128)[None, :]
    v_base = b * 16384 + col_j * 8 + g * 8
    val    = tl.load(in_4_ptr + v_base + v_off).to(tl.float32)

    # ── AttnV: [8,128] = [8,8] @ [128,8] ───────────────────────────────────
    out_f32 = tl.dot(et, val)   # [8,128]

    # ── Store out[b, col_j, n1i] into [4,128,64] ────────────────────────────
    # out stride [4,128,64]: element [b,ho,nj] at b*8192+ho*64+nj
    out_off = b * 8192 + col_j[None, :] * 64 + n1i[:, None]
    tl.store(out_ptr + out_off, out_f32.to(out_ptr.dtype.element_ty))


# ── Pattern ───────────────────────────────────────────────────────────────────
def pattern(in_0, tmp_11, in_4):
    tmp_12  = in_0 + tmp_11
    tmp_13  = tmp_12.softmax(dim = -1)
    matmul  = tmp_13 @ in_4
    tmp_15  = matmul.transpose(-1, -2)
    return tmp_15


def replacement_args(in_0, tmp_11, in_4):
    return (in_0, tmp_11, in_4)


@torch.fx.wrap
def fused_v64(in_0, tmp_11, in_4):
    # softmax(in_0+tmp_11) × in_4 → transpose: [4,64,64] × [4,64,128] → [4,128,64]
    # (separate wrapper; only used when V64 pass is active)
    out = torch.empty(4, 128, 64, dtype=in_0.dtype, device=in_0.device)
    _softmax_attn_fwd_v256[256, 4](in_0, in_4, out)
    return out


def replacement_func():
    return fused_v64