"""
Pass: FuseMHAGetItem_seq150

Pattern: MHA(self-attn, need_weights=True) + getitem[0]   (NO dropout in pattern)
Replace: Triton Flash-Attention (avoids attention-weight computation overhead)

This is a fallback pass in case the dropout nodes were eliminated before
pattern matching reaches the full MHA+dropout pattern.
"""
import torch
import triton
import triton.language as tl
import math


# ─── Triton Flash Attention Kernel ────────────────────────────────────────────
# Grid: (bsz*num_heads, ceil(seq_len / BLOCK_M))
# Q/K/V: contiguous [bsz*num_heads, seq_len, head_dim]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16}, num_warps=4, num_stages=2),
    ],
    key=['seq_len'],
)
@triton.jit
def _flash_attn_gi(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_h, stride_s, stride_d,
    seq_len,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    head_idx = tl.program_id(0)
    m_block  = tl.program_id(1)

    m_start = m_block * BLOCK_M
    m_offs  = m_start + tl.arange(0, BLOCK_M)
    d_offs  = tl.arange(0, HEAD_DIM)

    q_ptrs = Q_ptr + head_idx * stride_h + m_offs[:, None] * stride_s + d_offs[None, :] * stride_d
    q_mask = m_offs[:, None] < seq_len
    q      = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32) * scale

    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for n_start in range(0, seq_len, BLOCK_N):
        n_offs = n_start + tl.arange(0, BLOCK_N)
        n_mask = n_offs < seq_len

        k_ptrs = K_ptr + head_idx * stride_h + n_offs[:, None] * stride_s + d_offs[None, :] * stride_d
        k      = tl.load(k_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k))
        scores = tl.where(n_mask[None, :], scores, float('-inf'))

        m_ij  = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(scores - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v_ptrs = V_ptr + head_idx * stride_h + n_offs[:, None] * stride_s + d_offs[None, :] * stride_d
        v      = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)
        acc    = acc + tl.dot(p, v)
        m_i    = m_new

    acc = acc / l_i[:, None]

    out_ptrs = Out_ptr + head_idx * stride_h + m_offs[:, None] * stride_s + d_offs[None, :] * stride_d
    out_mask = m_offs[:, None] < seq_len
    tl.store(out_ptrs, acc, mask=out_mask)


# ─── Triton matmul + bias-add kernel ──────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _linear_gi(
    A_ptr, W_ptr, bias_ptr, Out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offs = m_start + tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        a = tl.load(A_ptr + m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak,
                    mask=(m_offs[:, None] < M) & k_mask[None, :], other=0.0).to(tl.float32)
        w = tl.load(W_ptr + n_offs[:, None] * stride_wn + k_offs[None, :] * stride_wk,
                    mask=(n_offs[:, None] < N) & k_mask[None, :], other=0.0).to(tl.float32)
        acc += tl.dot(a, tl.trans(w))

    bias = tl.load(bias_ptr + n_offs, mask=n_offs < N, other=0.0).to(tl.float32)
    acc += bias[None, :]

    tl.store(Out_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on,
             acc, mask=(m_offs[:, None] < M) & (n_offs[None, :] < N))


def _triton_linear_gi(a, w, bias):
    M, K = a.shape
    N    = w.shape[0]
    out  = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 64))
    _linear_gi[grid](
        a, w, bias, out,
        M, N, K,
        a.stride(0), a.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


# ─── Wrapper ──────────────────────────────────────────────────────────────────

@torch.fx.wrap
def optimized_mha_gi(in_4, in_3, in_2, in_1, in_0):
    """
    Replacement for MHA(need_weights=True) + getitem[0].
    Uses Triton Flash Attention (no attention weights computed).

    in_4 : [seq_len, bsz, embed_dim]
    in_3 : [3*embed_dim, embed_dim]   – in_proj_weight
    in_2 : [3*embed_dim]              – in_proj_bias
    in_1 : [embed_dim, embed_dim]     – out_proj_weight
    in_0 : [embed_dim]                – out_proj_bias
    """
    seq_len, bsz, embed_dim = in_4.shape
    num_heads = 8
    head_dim  = embed_dim // num_heads   # 64

    # QKV projection
    x      = in_4.reshape(seq_len * bsz, embed_dim).contiguous()
    x_f32  = x.to(torch.float32)
    w3_f32 = in_3.to(torch.float32)
    b2_f32 = in_2.to(torch.float32)
    qkv    = _triton_linear_gi(x_f32, w3_f32, b2_f32)   # [150, 1536]

    q = qkv[:, :embed_dim].contiguous()
    k = qkv[:, embed_dim:2*embed_dim].contiguous()
    v = qkv[:, 2*embed_dim:].contiguous()

    # Reshape to [bsz*num_heads, seq_len, head_dim]
    q = q.reshape(seq_len, bsz * num_heads, head_dim).permute(1, 0, 2).contiguous()
    k = k.reshape(seq_len, bsz * num_heads, head_dim).permute(1, 0, 2).contiguous()
    v = v.reshape(seq_len, bsz * num_heads, head_dim).permute(1, 0, 2).contiguous()

    num_bh   = bsz * num_heads
    out_attn = torch.empty(num_bh, seq_len, head_dim,
                           device=in_4.device, dtype=torch.float32)
    scale    = 1.0 / math.sqrt(head_dim)

    _flash_attn_gi[
        (num_bh, triton.cdiv(seq_len, 16))
    ](
        q, k, v, out_attn,
        seq_len * head_dim, head_dim, 1,
        seq_len, scale,
        HEAD_DIM=head_dim,
    )

    # Output projection
    out_flat = out_attn.permute(1, 0, 2).contiguous().reshape(seq_len * bsz, embed_dim)
    w1_f32   = in_1.to(torch.float32)
    b0_f32   = in_0.to(torch.float32)
    output   = _triton_linear_gi(out_flat, w1_f32, b0_f32)

    return output.to(in_4.dtype).reshape(seq_len, bsz, embed_dim)


# ─── Pattern: MHA + getitem[0] ONLY (no dropout) ─────────────────────────────

def pattern(in_4, in_3, in_2, in_1, in_0):
    result = torch.nn.functional.multi_head_attention_forward(
        in_4, in_4, in_4, 512, 8, in_3, in_2, None, None, False, 0.0,
        in_1, in_0,
        training=False,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        average_attn_weights=True,
        is_causal=False,
    )
    return result[0]


def replacement_args(in_4, in_3, in_2, in_1, in_0):
    return (in_4, in_3, in_2, in_1, in_0)


def replacement_func():
    return optimized_mha_gi