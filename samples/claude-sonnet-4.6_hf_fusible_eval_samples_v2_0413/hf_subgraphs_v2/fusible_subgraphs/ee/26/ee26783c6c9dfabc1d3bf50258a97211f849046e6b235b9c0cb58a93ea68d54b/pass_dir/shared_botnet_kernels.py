"""
Shared Triton kernels for BotNet relative attention optimization.

Two kernels:
  blocked_fa_kernel   – N=16: processes N rows at once, tl.dot([16,16],[16,BK])
  simple_fa_kernel    – N=8:  per-row flash-attn, tl.sum (K=8 < 16 bans tl.dot)

Pattern: tmp_10.reshape → add in_0 → softmax → matmul → transpose
tmp_10 contiguous [B,N,H,N,N]: bias[b,xpid,m,jk,jr] at b*S^2+xpid*N^3+m*N^2+jk*N+jr
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Kernel A – N=16: blocked flash-attn with tl.dot (tensor cores)
#   Grid: (B, S // N, K // BK)
# ---------------------------------------------------------------------------
@triton.jit
def blocked_fa_kernel(
    in0_ptr, bias_ptr, in4_ptr, out_ptr,
    B, H: tl.constexpr, N: tl.constexpr, S: tl.constexpr,
    K: tl.constexpr, BK: tl.constexpr, IS_BF16: tl.constexpr,
):
    b    = tl.program_id(0)
    xpid = tl.program_id(1)
    kpid = tl.program_id(2)

    m_loc  = tl.arange(0, N)
    m_offs = xpid * N + m_loc
    k_offs = kpid * BK + tl.arange(0, BK)
    j_r    = tl.arange(0, N)

    m_i = tl.full([N], -1e9, dtype=tl.float32)
    l_i = tl.zeros([N],      dtype=tl.float32)
    acc = tl.zeros([N, BK],  dtype=tl.float32)

    bias_base = b * (S * S) + xpid * (N * N * N) + m_loc * (N * N)

    for jk in range(H):
        j_offs = jk * N + j_r

        qk = (
            tl.load(in0_ptr  + b*S*S + m_offs[:, None]*S + j_offs[None, :]).to(tl.float32)
          + tl.load(bias_ptr + bias_base[:, None] + j_offs[None, :]).to(tl.float32)
        )

        m_tile  = tl.max(qk, axis=1)
        m_new   = tl.maximum(m_i, m_tile)
        exp_qk  = tl.exp(qk - m_new[:, None])
        l_scale = tl.exp(m_i - m_new)

        v_tile = tl.load(
            in4_ptr + (b*S + jk*N)*K + j_r[:, None]*K + k_offs[None, :]
        ).to(tl.float32)

        acc = (acc * l_scale[:, None]
               + tl.dot(exp_qk.to(tl.float16), v_tile.to(tl.float16),
                        out_dtype=tl.float32))
        l_i = l_i * l_scale + tl.sum(exp_qk, axis=1)
        m_i = m_new

    out_v    = acc / l_i[:, None]
    out_dtype = tl.bfloat16 if IS_BF16 else tl.float16
    tl.store(
        out_ptr + b*K*S + k_offs[None, :]*S + m_offs[:, None],
        out_v.to(out_dtype)
    )


# ---------------------------------------------------------------------------
# Kernel B – N=8: per-row flash-attn with tl.sum (avoids K<16 tl.dot error)
#   Grid: (B, S, K // BK)
# ---------------------------------------------------------------------------
@triton.jit
def simple_fa_kernel(
    in0_ptr, bias_ptr, in4_ptr, out_ptr,
    B, H: tl.constexpr, N: tl.constexpr, S: tl.constexpr,
    K: tl.constexpr, BK: tl.constexpr, IS_BF16: tl.constexpr,
):
    b    = tl.program_id(0)
    i    = tl.program_id(1)
    kpid = tl.program_id(2)

    i_x    = i // H
    i_h    = i % H
    k_offs = kpid * BK + tl.arange(0, BK)
    j_r    = tl.arange(0, N)

    m_i = tl.full([1], -1e9, dtype=tl.float32)
    l_i = tl.zeros([1],      dtype=tl.float32)
    acc = tl.zeros([BK],     dtype=tl.float32)

    bias_row = b*(S*S) + i_x*(N*N*N) + i_h*(N*N)

    for jk in range(H):
        j_offs = jk * N + j_r

        qk = (
            tl.load(in0_ptr  + b*S*S + i*S + j_offs).to(tl.float32)
          + tl.load(bias_ptr + bias_row + j_offs).to(tl.float32)
        )

        m_tile  = tl.max(qk, axis=0)
        m_new   = tl.maximum(m_i, m_tile)
        exp_qk  = tl.exp(qk - m_new)
        l_scale = tl.exp(m_i - m_new)

        v_tile = tl.load(
            in4_ptr + (b*S + jk*N)*K + j_r[:, None]*K + k_offs[None, :]
        ).to(tl.float32)

        acc = acc * l_scale + tl.sum(exp_qk[:, None] * v_tile, axis=0)
        l_i = l_i * l_scale + tl.sum(exp_qk, axis=0)
        m_i = m_new

    out_v     = acc / l_i
    out_dtype = tl.bfloat16 if IS_BF16 else tl.float16
    tl.store(out_ptr + b*K*S + k_offs*S + i, out_v.to(out_dtype))


# ---------------------------------------------------------------------------
# Dispatch wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def botnet_attn_dispatch(in_0, bias, in_4, route: str):
    """
    Fused attention: softmax(in_0 + bias_reshaped) @ in_4 → [B, K, S]

    route "N16" → blocked_fa_kernel with tl.dot (tensor cores)
    route "N8"  → simple_fa_kernel  with tl.sum (K=8 < 16 bans tl.dot)
    """
    B = in_0.shape[0]
    is_bf16 = (in_0.dtype == torch.bfloat16)

    if route == "N16":
        H, N, S, K, BK = 16, 16, 256, 128, 16
        out = torch.empty((B, K, S), dtype=in_0.dtype, device=in_0.device)
        blocked_fa_kernel[(B, S // N, K // BK)](
            in_0, bias, in_4, out,
            B, H, N, S, K, BK, IS_BF16=is_bf16, num_warps=4,
        )
    else:
        H, N, S, K, BK = 8, 8, 64, 128, 32
        out = torch.empty((B, K, S), dtype=in_0.dtype, device=in_0.device)
        simple_fa_kernel[(B, S, K // BK)](
            in_0, bias, in_4, out,
            B, H, N, S, K, BK, IS_BF16=is_bf16, num_warps=4,
        )
    return out