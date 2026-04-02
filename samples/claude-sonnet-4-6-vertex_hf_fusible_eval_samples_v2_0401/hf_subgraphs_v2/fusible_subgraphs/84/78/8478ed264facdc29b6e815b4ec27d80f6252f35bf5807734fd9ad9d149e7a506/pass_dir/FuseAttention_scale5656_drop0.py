import torch
import triton
import triton.language as tl


def _next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p


# ── Flash Attention kernel (fused QK^T·scale·softmax·AV + permute) ────────
@triton.jit
def _flash_attn_fwd_5656(
    Q_ptr, KT_ptr, V_ptr, Out_ptr,
    # Q:  [B, H, SQ, DK]
    # KT: [B, H, DK, SK]  (key already transposed)
    # V:  [B, H, SK, DV]
    # Out:[B, SQ, H, DV]  (output in permuted order)
    q_sb, q_sh, q_ssq, q_sdk,
    kt_sb, kt_sh, kt_sdk, kt_ssk,
    v_sb, v_sh, v_ssk, v_sdv,
    B, H, SQ, SK, DK, DV,
    scale,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SK: tl.constexpr,
    BLOCK_DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    bid   = tl.program_id(0)
    hid   = tl.program_id(1)
    sq_id = tl.program_id(2)

    sq_off = sq_id * BLOCK_SQ + tl.arange(0, BLOCK_SQ)
    sq_mask = sq_off < SQ
    dk_off = tl.arange(0, BLOCK_DK)
    dk_mask = dk_off < DK
    dv_off = tl.arange(0, BLOCK_DV)
    dv_mask = dv_off < DV

    # Load Q tile [BLOCK_SQ, BLOCK_DK]
    q_ptrs = (Q_ptr
              + bid * q_sb + hid * q_sh
              + sq_off[:, None] * q_ssq
              + dk_off[None, :] * q_sdk)
    q = tl.load(q_ptrs, mask=sq_mask[:, None] & dk_mask[None, :], other=0.0)
    q = q.to(tl.float32) * scale

    # Running accumulators
    m_i = tl.full([BLOCK_SQ], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_SQ], dtype=tl.float32)
    o_i = tl.zeros([BLOCK_SQ, BLOCK_DV], dtype=tl.float32)

    # Iterate over SK tiles
    for sk_start in range(0, tl.cdiv(SK, BLOCK_SK)):
        sk_off = sk_start * BLOCK_SK + tl.arange(0, BLOCK_SK)
        sk_mask = sk_off < SK

        # Load KT tile [BLOCK_DK, BLOCK_SK]
        kt_ptrs = (KT_ptr
                   + bid * kt_sb + hid * kt_sh
                   + dk_off[:, None] * kt_sdk
                   + sk_off[None, :] * kt_ssk)
        kt = tl.load(kt_ptrs, mask=dk_mask[:, None] & sk_mask[None, :], other=0.0)
        kt = kt.to(tl.float32)

        # Attention scores [BLOCK_SQ, BLOCK_SK]
        s = tl.dot(q, kt)
        s = tl.where(sk_mask[None, :] & sq_mask[:, None], s, float('-inf'))

        # Online softmax
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha  = tl.exp(m_i - m_new)
        p      = tl.exp(s - m_new[:, None])
        l_i    = alpha * l_i + tl.sum(p, axis=1)
        o_i    = o_i * alpha[:, None]

        # Load V tile [BLOCK_SK, BLOCK_DV]
        v_ptrs = (V_ptr
                  + bid * v_sb + hid * v_sh
                  + sk_off[:, None] * v_ssk
                  + dv_off[None, :] * v_sdv)
        v = tl.load(v_ptrs, mask=sk_mask[:, None] & dv_mask[None, :], other=0.0)
        v = v.to(tl.float32)

        o_i += tl.dot(p, v)
        m_i = m_new

    # Normalize
    o_i = o_i / l_i[:, None]

    # Write to Out[bid, sq_off, hid, dv_off]  (permuted layout: [B,SQ,H,DV])
    out_ptrs = (Out_ptr
                + bid * (SQ * H * DV)
                + sq_off[:, None] * (H * DV)
                + hid * DV
                + dv_off[None, :])
    out_mask = sq_mask[:, None] & dv_mask[None, :]
    tl.store(out_ptrs, o_i, mask=out_mask)


@torch.fx.wrap
def fused_sdpa_5656_drop0(in_0, in_1, in_2):
    """
    in_0 : [B, H, SQ, DK]  query
    in_1 : [B, H, DK, SK]  key (already transposed)
    in_2 : [B, H, SK, DV]  value
    returns [B, SQ, H, DV] (permute(0,2,1,3).contiguous())
    """
    B, H, SQ, DK = in_0.shape
    _,  _, _,  SK = in_1.shape
    _,  _, _,  DV = in_2.shape

    # Force contiguous for strided access
    q  = in_0.contiguous()
    kt = in_1.contiguous()
    v  = in_2.contiguous()

    out = torch.empty(B, SQ, H, DV, dtype=in_0.dtype, device=in_0.device)

    BLOCK_SQ = 16
    BLOCK_SK = max(16, _next_pow2(min(SK, 256)))
    BLOCK_DK = _next_pow2(DK)
    BLOCK_DV = _next_pow2(DV)

    grid = (B, H, (SQ + BLOCK_SQ - 1) // BLOCK_SQ)
    _flash_attn_fwd_5656[grid](
        q, kt, v, out,
        q.stride(0),  q.stride(1),  q.stride(2),  q.stride(3),
        kt.stride(0), kt.stride(1), kt.stride(2), kt.stride(3),
        v.stride(0),  v.stride(1),  v.stride(2),  v.stride(3),
        B, H, SQ, SK, DK, DV,
        1.0 / 5.656854249492381,
        BLOCK_SQ=BLOCK_SQ, BLOCK_SK=BLOCK_SK,
        BLOCK_DK=BLOCK_DK, BLOCK_DV=BLOCK_DV,
        num_warps=4,
    )
    return out


# ── Pattern ──────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2):
    tmp_0 = torch.matmul(in_0, in_1)
    tmp_1 = tmp_0 / 5.656854249492381
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.0, False, False)
    tmp_4 = torch.matmul(tmp_3, in_2)
    tmp_5 = tmp_4.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_sdpa_5656_drop0