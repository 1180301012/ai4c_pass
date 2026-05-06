import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: fuses reshape + split([32,32,128], dim=3) + permute(0,2,1,3)
#                + permute + transpose(-2,-1) for all three splits.
#
# Input  linear_out : [B, S, 1536]  contiguous CUDA tensor
# Outputs:
#   Q   : [B, H, S, 32]     (split[0] Permuted)
#   K^T : [B, H, 32, S]     (split[1] Permuted + transposed)
#   V   : [B, H, 128, S]    (split[2] Permuted)
#
# Grid : B*H programs — one per (batch, head) slice.
# Each program processes ALL S rows for one head.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 64 }, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64 }, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 32 }, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 32 }, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 64 }, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64 }, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_K': 64 }, num_warps=4, num_stages=4),
    ],
    key=['B', 'S', 'C'],
)
@triton.jit
def _fused_kernel(
    X_ptr,       # linear_out  [B, S, 3*C]  row-major
    Q_ptr,       # [B, H, S, 32]
    K_T_ptr,     # [B, H, 32, S]
    V_ptr,       # [B, H, 128, S]
    B, S, C,
    # X strides
    sx_b, sx_s, sx_k,
    # Q strides
    sq_b, sq_h, sq_s, sq_d,
    # K^T strides
    sktb_b, sktb_h, sktb_d, sktb_s,
    # V strides
    sv_b, sv_h, sv_d, sv_s,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    One program per (b, h) pair.  Grid axis 0 = B*H.
    Processes all S rows for that (b,h) in BLOCK_M-wide tiles.
    """
    pid = tl.program_id(0)          # 0 .. B*H - 1
    b   = pid // 8                  # H = 8 (fixed for this graph)
    h   = pid % 8

    m_offs   = tl.arange(0, BLOCK_M)        # sequence position tile
    m_mask   = m_offs < S
    base_row = b * S + m_offs                # flat row index in [B*S, C]

    Q_CHUNK  = 32
    K_CHUNK  = 32
    V_CHUNK  = 128
    QKV_OFF  = 96                             # start of V slice
    QKV_OFF2 = 64                             # start of K^T slice

    # ---- Q tile: columns  0 .. Q_CHUNK-1 = 0..31 ----
    acc_q = tl.zeros((BLOCK_M, Q_CHUNK), tl.float32)
    for k in range(0, 448, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        k2d    = m_mask[:, None] & (k_offs[None, :] < 448)
        x = tl.load(X_ptr + base_row[:, None] * sx_k + k_offs[None, :] * sx_k,
                    mask=k2d, other=0.)       # [BLOCK_M, BLOCK_K]
        w = tl.load(W_ptr + tl.arange(0, Q_CHUNK)[:, None] * sx_k + k_offs[None, :] * sx_k,
                    mask=(k_offs[None, :] < 448), other=0.)   # [Q_CHUNK, BLOCK_K]
        acc_q = tl.dot(x, tl.trans(w), acc_q)
    bv = tl.load(W_ptr + tl.arange(0, Q_CHUNK) * sx_k)
    acc_q += bv[None, :]

    q_ptrs = Q_ptr + b * sq_b + h * sq_h + m_offs[:, None] * sq_s + tl.arange(0, Q_CHUNK)[None, :] * sq_d
    tl.store(q_ptrs, acc_q.to(Q_ptr.dtype.element_ty), mask=m_mask[:, None])

    # ---- K^T tile: columns Q Chunk .. QKV Off = 32 .. 63 ----
    acc_k = tl.zeros((BLOCK_M, K_CHUNK), tl.float32)
    for k in range(0, 448, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        k2d    = m_mask[:, None] & (k_offs[None, :] < 448)
        x = tl.load(X_ptr + base_row[:, None] * sx_k + k_offs[None, :] * sx_k,
                    mask=k2d, other=0.)
        w = tl.load(W_ptr + (Q_CHUNK + tl.arange(0, K_CHUNK))[:, None] * sx_k + k_offs[None, :] * sx_k,
                    mask=(k_offs[None, :] < 448), other=0.)   # [K_CHUNK, BLOCK_K]
        acc_k = tl.dot(x, tl.trans(w), acc_k)
    bv = tl.load(W_ptr + (Q_CHUNK + tl.arange(0, K_CHUNK)) * sx_k)
    acc_k += bv[None, :]

    # -- K^T has layout [B,H,D,S]  so dimension index → s index ----
    kt_ptrs = K_T_ptr + b * sktb_b + h * sktb_h + tl.arange(0, K_CHUNK)[None, :] * sktb_d \
                                     + m_offs[:, None] * sktb_s
    tl.store(kt_ptrs, acc_k.to(K_T_ptr.dtype.element_ty), mask=m_mask[:, None])

    # ---- V tile: columns QKV Off2 .. end = 64 .. 191 ----
    acc_v = tl.zeros((BLOCK_M, V_CHUNK), tl.float32)
    for k in range(0, 448, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        k2d    = m_mask[:, None] & (k_offs[None, :] < 448)
        x = tl.load(X_ptr + base_row[:, None] * sx_k + k_offs[None, :] * sx_k,
                    mask=k2d, other=0.)
        w = tl.load(W_ptr + (QKV_OFF2 + tl.arange(0, V_CHUNK))[:, None] * sx_k + k_offs[None, :] * sx_k,
                    mask=(k_offs[None, :] < 448), other=0.)   # [V_CHUNK, BLOCK_K]
        acc_v = tl.dot(x, tl.trans(w), acc_v)
    bv = tl.load(W_ptr + (QKV_OFF2 + tl.arange(0, V_CHUNK)) * sx_k)
    acc_v += bv[None, :]

    # -- V has layout [B,H,128,S]  so dim index → s index ----
    v_ptrs = V_ptr + b * sv_b + h * sv_h + tl.arange(0, V_CHUNK)[None, :] * sv_d \
                                   + m_offs[:, None] * sv_s
    tl.store(v_ptrs, acc_v.to(V_ptr.dtype.element_ty), mask=m_mask[:, None])


@torch.fx.wrap
def fused_reshape_split_wrapper(linear_out):
    """
    Replaces reshape(split; permute×3, transpose×1) on linear_out [B,49,1536].

    Returns: Q   [B, 8, 49, 32],
             K_T [B, 8, 32, 49],
             V   [B, 8, 49, 128]
    """
    B, S, C = linear_out.shape   # [B, 49, 1536]
    H      = 8
    Q   = torch.empty(B, H, S, 32,    dtype=linear_out.dtype, device=linear_out.device)
    K_T = torch.empty(B, H, 32,  S,    dtype=linear_out.dtype, device=linear_out.device)
    V   = torch.empty(B, H, 128, S,    dtype=linear_out.dtype, device=linear_out.device)

    _fused_kernel[(B * H,)](
        linear_out, Q, K_T, V,
        B, S, C,
        linear_out.stride(0), linear_out.stride(1), linear_out.stride(2),
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K_T.stride(0), K_T.stride(1), K_T.stride(2), K_T.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
    )

    return Q, K_T, V


# ---------------------------------------------------------------------------
# Pattern: matches the four-tuple (reshape → split → permute → permute →
#          permute → transpose) that follows the linear output in all variants.
# `linear` is an abstract node (input to the pattern), so dtype does not matter.
# ---------------------------------------------------------------------------
def pattern(linear):
    x4d            = linear.reshape(8, 49, 8, -1)
    split_targets  = x4d.split([32, 32, 128], dim=3)
    q   = split_targets[0]
    kt  = split_targets[1]
    v   = split_targets[2]
    q1  = q.permute(0, 2, 1, 3)
    kt1 = kt.permute(0, 2, 1, 3)
    v1  = v.permute(0, 2, 1, 3)
    kt2 = kt1.transpose(-2, -1)
    return q1, kt2, v1


def replacement_args(linear):
    return (linear,)


def replacement_func():
    return fused_reshape_split_wrapper