"""
Shared fused scaled dot-product attention (SDPA) kernel used by all pass files.
Fuses: Q@K^T / scale -> softmax -> @V -> permute -> contiguous -> view
into a single Triton kernel that writes directly to the [B, Sq, H*D] layout.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32}, num_stages=4, num_warps=2),
    ],
    key=['Sq', 'Sk', 'D'],
)
@triton.jit
def _fused_sdpa_kernel(
    q_ptr, kt_ptr, v_ptr, out_ptr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kt0, stride_kt1, stride_kt2, stride_kt3,
    stride_vb, stride_vh, stride_vk, stride_vd,
    stride_ob, stride_on,
    B, H, Sq, Sk, D, HD, scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Each program handles one (batch, head) pair and a BLOCK_M chunk of queries.
    Inner loop iterates over BLOCK_N chunks of the key/value sequence.
    Accumulates in float32; stores in the original input dtype.
    Output layout: [B, Sq, H*D]  (permute(0,2,1,3) + view combined)
    """
    pid_bh = tl.program_id(0)
    pid_m  = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    m_start   = pid_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    m_mask    = m_offsets < Sq
    d_offsets = tl.arange(0, 32)          # always 32 for tl.dot compatibility

    # ---- Load Q block [BLOCK_M, 32] in float32 ----
    q_base    = b * stride_qb + h * stride_qh
    q_offsets = m_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd
    q = tl.load(q_ptr + q_base + q_offsets,
                mask=m_mask[:, None], other=0.0).to(tl.float32)

    # ---- Online-softmax accumulators ----
    row_max  = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    row_sum  = tl.zeros([BLOCK_M], dtype=tl.float32)
    o_acc    = tl.zeros([BLOCK_M, 32], dtype=tl.float32)

    kt_base = b * stride_kt0 + h * stride_kt1  # K is stored as [B, H, D, Sk]

    for k_start in range(0, Sk, BLOCK_N):
        n_offsets = k_start + tl.arange(0, BLOCK_N)
        n_mask    = n_offsets < Sk

        # Load K block [32, BLOCK_N] from K^T = [B, H, D, Sk]
        kt_offsets = d_offsets[:, None] * stride_kt2 + n_offsets[None, :] * stride_kt3
        k_block = tl.load(kt_ptr + kt_base + kt_offsets,
                          mask=n_mask[None, :], other=0.0).to(tl.float32)

        # Attention scores [BLOCK_M, BLOCK_N] = Q @ K^T / scale
        scores = tl.dot(q, k_block) * (1.0 / scale)

        # Mask invalid K positions
        scores = tl.where(n_mask[None, :], scores, float('-inf'))

        # Online softmax: update running max/sum, re-normalise
        new_max   = tl.maximum(row_max, tl.max(scores, axis=1))
        exp_s     = tl.exp(scores - new_max[:, None])
        exp_s     = tl.where(n_mask[None, :], exp_s, 0.0)
        row_sum   = row_sum * tl.exp(new_max - row_max) + tl.sum(exp_s, axis=1)
        row_max   = new_max

        # Load V block [BLOCK_N, 32]
        v_base    = b * stride_vb + h * stride_vh
        v_offsets = n_offsets[:, None] * stride_vk + d_offsets[None, :] * stride_vd
        v_block   = tl.load(v_ptr + v_base + v_offsets,
                            mask=n_mask[:, None], other=0.0).to(tl.float32)

        # Weighted V accumulation
        o_acc = o_acc * tl.exp(new_max - row_max)[:, None] + tl.dot(exp_s, v_block)

    # ---- Normalise ----
    o_acc = o_acc / row_sum[:, None]

    # ---- Write output to [B, Sq, H*D] ----
    # out[b, m, h*D + d]  →  offset = b*stride_ob + m*stride_on + (h*D+d)
    out_d     = tl.arange(0, 32)
    out_base  = b * stride_ob + h * D
    out_offs  = m_offsets[:, None] * stride_on + out_d[None, :] + out_base
    out_mask  = m_mask[:, None] & (out_d[None, :] < D)

    tl.store(out_ptr + out_offs,
             o_acc.to(out_ptr.dtype.element_ty),
             mask=out_mask)


@torch.fx.wrap
def fused_attn_scale5656(in_0, in_1, in_2):
    """Replacement for scale=5.656854249492381, dropout=0.0."""
    B, H, Sq, D = in_0.shape
    Sk  = in_1.shape[3]
    HD  = H * D
    scale = 5.656854249492381  # sqrt(32)

    out = torch.empty((B, Sq, HD), dtype=in_0.dtype, device=in_0.device)

    def grid(meta):
        return (B * H, triton.cdiv(Sq, meta['BLOCK_M']))

    _fused_sdpa_kernel[grid](
        in_0, in_1, in_2, out,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        out.stride(0), out.stride(1),
        B, H, Sq, Sk, D, HD, scale,
    )
    return out


@torch.fx.wrap
def fused_attn_scale80(in_0, in_1, in_2):
    """Replacement for scale=8.0, dropout=0.0."""
    B, H, Sq, D = in_0.shape
    Sk  = in_1.shape[3]
    HD  = H * D
    scale = 8.0

    out = torch.empty((B, Sq, HD), dtype=in_0.dtype, device=in_0.device)

    def grid(meta):
        return (B * H, triton.cdiv(Sq, meta['BLOCK_M']))

    _fused_sdpa_kernel[grid](
        in_0, in_1, in_2, out,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        out.stride(0), out.stride(1),
        B, H, Sq, Sk, D, HD, scale,
    )
    return out


@torch.fx.wrap
def dispatch_fused_attn(in_0, in_1, in_2, route):
    """
    Unified dispatch wrapper shared by ALL pass files.
    The `route` string (passed via replacement_args) selects the scale.
    All kernel computations happen in float32; results are stored in the
    original input dtype.
    Output shape: [B, Sq, H*D]  (equivalent to permute+contiguous+view).
    """
    B, H, Sq, D = in_0.shape
    Sk  = in_1.shape[3]
    HD  = H * D
    out = torch.empty((B, Sq, HD), dtype=in_0.dtype, device=in_0.device)

    def grid(meta):
        return (B * H, triton.cdiv(Sq, meta['BLOCK_M']))

    if route == "5656":
        scale = 5.656854249492381   # sqrt(32)
        _fused_sdpa_kernel[grid](
            in_0, in_1, in_2, out,
            in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
            in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
            in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
            out.stride(0), out.stride(1),
            B, H, Sq, Sk, D, HD, scale,
        )
    elif route == "6928":
        scale = 6.928203230275509   # sqrt(48)
        _fused_sdpa_kernel[grid](
            in_0, in_1, in_2, out,
            in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
            in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
            in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
            out.stride(0), out.stride(1),
            B, H, Sq, Sk, D, HD, scale,
        )
    else:  # route == "80"
        scale = 8.0                   # sqrt(64)
        _fused_sdpa_kernel[grid](
            in_0, in_1, in_2, out,
            in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
            in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
            in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
            out.stride(0), out.stride(1),
            B, H, Sq, Sk, D, HD, scale,
        )
    return out


@torch.fx.wrap
def dispatch_perm_cont(in_0, route):
    """
    Simple permute(0,2,1,3) + contiguous replacement.
    Input:  [B, H, Sq, D]  (attention weights or any 4-D tensor)
    Output: [B, Sq, H*D]   (same as permute+contiguous result)
    Uses a dedicated, fast Triton transpose kernel.
    """
    B, H, Sq, D = in_0.shape
    HD = H * D
    out = torch.empty((B, Sq, HD), dtype=in_0.dtype, device=in_0.device)

    def grid(meta):
        return (B * Sq, triton.cdiv(H * D, meta['BLOCK']))

    _perm_cont_kernel[grid](
        in_0, out,
        B, H, Sq, D, HD,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        out.stride(0), out.stride(1),
        BLOCK=32,
    )
    return out


@triton.jit
def _perm_cont_kernel(
    inp_ptr, out_ptr,
    B, H, Sq, D, HD,
    s0, s1, s2, s3,
    out_s0, out_s1,
    BLOCK: tl.constexpr,
):
    """Transpose [B,H,Sq,D] -> [B,Sq,H*D] in one pass."""
    pid_bhsq = tl.program_id(0)
    pid_blk  = tl.program_id(1)

    b   = pid_bhsq // Sq
    sq  = pid_bhsq % Sq
    blk = pid_blk

    h_off  = (blk * BLOCK) // HD
    d_off  = (blk * BLOCK) % HD
    h_off  = h_off  * 32
    d_off  = d_off  * 32

    h_off2 = tl.arange(0, 32) + h_off
    d_off2 = tl.arange(0, 32) + d_off
    h_mask = h_off2 < H
    d_mask = d_off2 < D

    inp_off = b * s0 + sq * s2 + h_off2[:, None] * s1 + d_off2[None, :] * s3
    x = tl.load(inp_ptr + inp_off, mask=h_mask[:, None] & d_mask[None, :], other=0.0)

    out_off = b * out_s0 + sq * out_s1 + h_off2[:, None] * D + d_off2[None, :]
    tl.store(out_ptr + out_off, x, mask=h_mask[:, None] & d_mask[None, :])


@torch.fx.wrap
def dispatch_perm_cont(in_0, route):
    """
    Fused permute(0,2,1,3) + contiguous for attention output.
    Input : [B, H, Sq, D]
    Output: [B, Sq, H*D]
    """
    B, H, Sq, D = in_0.shape
    HD = H * D
    out = torch.empty((B, Sq, HD), dtype=in_0.dtype, device=in_0.device)

    BLOCK = 32
    grid = (B * Sq, triton.cdiv(H * D, BLOCK))

    _perm_cont_kernel[grid](
        in_0, out,
        B, H, Sq, D, HD,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        out.stride(0), out.stride(1),
        BLOCK=BLOCK,
    )
    return out


# ── Keep old names for backward compatibility ──
@torch.fx.wrap
def fused_attn_scale5656(in_0, in_1, in_2):
    return dispatch_fused_attn(in_0, in_1, in_2, "5656")

@torch.fx.wrap
def fused_attn_scale80(in_0, in_1, in_2):
    return dispatch_fused_attn(in_0, in_1, in_2, "80")

@torch.fx.wrap
def fused_attn_scale6928(in_0, in_1, in_2):
    return dispatch_fused_attn(in_0, in_1, in_2, "6928")


# ── Keep old names for backward compatibility (imported by old pass files) ──
@torch.fx.wrap
def fused_attn_scale5656(in_0, in_1, in_2):
    return dispatch_fused_attn(in_0, in_1, in_2, "5656")

@torch.fx.wrap
def fused_attn_scale80(in_0, in_1, in_2):
    return dispatch_fused_attn(in_0, in_1, in_2, "80")

@torch.fx.wrap
def fused_attn_scale6928(in_0, in_1, in_2):
    return dispatch_fused_attn(in_0, in_1, in_2, "6928")